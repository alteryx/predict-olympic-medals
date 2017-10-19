import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import Imputer, RobustScaler
import featuretools as ft
from featuretools import primitives as ftypes
from featuretools.selection.variance_selection import select_high_variance_features
from ml import fit_and_score, TimeSeriesSplitByDate, bin_labels
import cPickle as pck
import itertools
import os

def feature_importances_as_df(fitted_est, columns):
    return (pd.DataFrame({
        'Importance': fitted_est.steps[-1][1].feature_importances_,
        'Feature': columns
    }).sort_values(['Importance'], ascending=False))


def get_feature_importances(estimator, feature_matrix, labels, splitter,
                            n=100):
    feature_imps_by_time = {}
    for i, train_test_i in enumerate(splitter.split(None, labels.values)):
        train_i, test_i = train_test_i
        train_dates, test_dates = splitter.get_dates(i, y=labels.values)
        X = feature_matrix.values[train_i, :]
        icols_used = [i for i, c in enumerate(X.T) if not pd.isnull(c).all()]
        cols_used = feature_matrix.columns[icols_used].tolist()

        X = X[:, icols_used]
        y = labels.values[train_i]
        clf = clone(estimator)
        clf.fit(X, y)
        feature_importances = feature_importances_as_df(clf, cols_used)
        feature_imps_by_time[test_dates[-1]] = feature_importances.head(n)

    return feature_imps_by_time


def build_baseline_features(es):
    # Baseline 1
    total_num_medals = ftypes.Count(es['medals_won']['medal_id'], es['countries'])
    count_num_olympics = ftypes.NUnique(
        es['countries_at_olympic_games']['Olympic Games ID'], es['countries'])
    mean_num_medals = (
        total_num_medals / count_num_olympics).rename("mean_num_medals")

    # Baseline 2
    olympic_id = ft.Feature(es['countries_at_olympic_games']['Olympic Games ID'],
                            es['medals_won'])
    num_medals_each_olympics = [
        ftypes.Count(
            es['medals_won']['medal_id'], es['countries'],
            where=olympic_id == i).rename("num_medals_olympics_{}".format(i))
        for i in es.get_all_instances('olympic_games')
    ]
    return num_medals_each_olympics, mean_num_medals



def load_econ_indicators(econ_path='~/olympic_games_data/economic_data/',
                         since_date=None):
    country_cols = [
        'CountryCode', 'IncomeGroup', 'SystemOfTrade',
        'GovernmentAccountingConcept'
    ]
    econ_country = pd.read_csv(
        econ_path + "Country.csv", encoding='utf-8', usecols=country_cols)
    econ_country = econ_country.append(
        pd.DataFrame({
            'CountryCode': ['UNK']
        }), ignore_index=True)

    econ_indicators = pd.read_csv(
        econ_path + "Indicators.csv", encoding='utf-8')
    econ_indicators['Year'] = pd.to_datetime(
        econ_indicators['Year'], format='%Y')
    econ_indicators.drop(
        ['CountryName', 'IndicatorCode'], axis=1, inplace=True)
    econ_indicators.set_index('CountryCode', inplace=True)
    econ_indicators.set_index('Year', append=True, inplace=True)
    econ_indicators.set_index('IndicatorName', append=True, inplace=True)
    econ_indicators = (econ_indicators['Value'].unstack(level='IndicatorName')
                    .reset_index(level='Year', drop=False).reset_index(
                        drop=False))
    econ_indicators.columns.name = None
    if since_date is not None:
        econ_indicators = econ_indicators[econ_indicators['Year'] >=
                                        since_date]
    return econ_country, econ_indicators


def load_regions(region_path='~/olympic_games_data/economic_data/'):
    intermediate_regions = pd.read_csv(region_path + 'country_regions.csv')

    # Drop Antarctica
    intermediate_regions = intermediate_regions[
        intermediate_regions['ISO-alpha3 Code'] != 'ATA']
    intermediate_regions['Intermediate Region Name'] = intermediate_regions[
        'Intermediate Region Name'].fillna('')
    combined_sub_regions = (intermediate_regions[[
        'Intermediate Region Name', 'Sub-region Name'
    ]].drop_duplicates())

    # Combine Sub-region Name and Intermediate Region Name,
    # removing extraneous colons at the end if there was no
    # Intermediate Region Name
    combined_sub_regions['Subregion Name'] = (
        combined_sub_regions['Sub-region Name'].str.cat(
            combined_sub_regions['Intermediate Region Name'],
            sep=': ').str.replace(r': \Z', ''))
    combined_sub_regions.reset_index(inplace=True, drop=True)
    combined_sub_regions.index.name = 'Subregion ID'
    combined_sub_regions.reset_index(inplace=True, drop=False)
    drop_cols = [
        'Intermediate Region Name', 'Sub-region Name', 'Global Code',
        'Global Name', 'Region Code', 'Sub-region Code',
        'Intermediate Region Code', 'M49 Code'
    ]
    intermediate_regions = intermediate_regions.merge(
        combined_sub_regions,
        on=['Intermediate Region Name', 'Sub-region Name'],
        how='left').drop(
            drop_cols, axis=1)

    max_subregion = intermediate_regions['Subregion ID'].max()
    intermediate_regions = intermediate_regions.append(
        pd.DataFrame({
            'Subregion ID': [max_subregion + 1],
            'Subregion Name': ['Unknown'],
            'Region Name': ['Unknown'],
            'ISO-alpha3 Code': ['UNK'],
            'Country or Area': ['Unknown']
        }),
        ignore_index=True)
    intermediate_regions.dropna(subset=['ISO-alpha3 Code'], inplace=True)
    return intermediate_regions


def match_countries_to_regions_and_econ_data(
        countries, region_countries, econ_country=None, econ_indicators=None):
    '''
    Match country codes from the `countries` dataframe (from the Kaggle olympics dataset)
    with the codes in the `regions` dataframe (from the UN)
    and the codes in the `econ_country` and `econ_indicators` dataframes (from the Kaggle economics dataset)

    These datasets contain slightly different country codes, and we want to link them up the best we can.
    We also want to pull additional information from the UN region dataset into countries

    The strategy:

    1. Merge `countries` with `region_countries` on the country name
    2. Merge `countries` with `region_countries` on the code (`countries` code is from the IOC, `region_countries` code is `ISO-alpha3 Code`)
    3. Combine these two merges by setting the nulls on Country and Code with the values from the other dataframe
    4. Merge the rest that did not share a name or code on a hand-defined dictionary mapping country names in the two dataframes
    5. Merge `countries` with original `region_countries` to pull in region and other information
    6. Generate a manual mapping on country codes in `countries` and `econ_country`
    7. Merge `countries` with `econ_country` using the `ISO-alpha3 Code` from region_countries as well as this mapping
    8. Pull IOC code from `countries` into `econ_indicators` and remove extraneous countries not present in the Olympics dataset
    '''
    # Step 1
    matching_countries_by_country = countries[['Code', 'Country']].merge(
        region_countries[['Country or Area', 'ISO-alpha3 Code']],
        right_on='Country or Area',
        left_on='Country',
        how='left')
    matching_countries_by_country.drop(
        ['Country or Area'], axis=1, inplace=True)
    # Step 2
    matching_countries_by_code = countries[['Code', 'Country']].merge(
        region_countries[['ISO-alpha3 Code']],
        right_on='ISO-alpha3 Code',
        left_on='Code',
        how='left')

    # Step 3
    matching_countries = matching_countries_by_code
    code_mask = matching_countries_by_country['ISO-alpha3 Code'].notnull()
    matching_countries.loc[
        code_mask, 'ISO-alpha3 Code'] = matching_countries_by_country.loc[
            code_mask, 'ISO-alpha3 Code']

    # Step 4
    country_region_mapping = {
        'Brunei': 'Brunei Darussalam',
        'Burma': 'Myanmar',
        'Iran': 'Iran (Islamic Republic of)',
        'Netherlands Antilles': 'Suriname',
        'Palestine, Occupied Territories': 'State of Palestine',
        'Taiwan': 'China',
        'Tanzania': 'United Republic of Tanzania',
        'Vietnam': 'Viet Nam',
        'Virgin Islands': 'United States Virgin Islands'
    }
    region_country_index = region_countries.set_index('Country or Area')[
        'ISO-alpha3 Code']
    region_mapping_codes = {
        k: region_country_index[v]
        for k, v in country_region_mapping.items()
    }
    dict_mapping_mask = matching_countries['Country'].isin(
        country_region_mapping)
    matching_countries.loc[
        dict_mapping_mask, 'ISO-alpha3 Code'] = matching_countries.loc[
            dict_mapping_mask, 'Country'].replace(region_mapping_codes).values
    countries['ISO-alpha3 Code'] = matching_countries['ISO-alpha3 Code'].values

    # Step 5
    region_cols_to_pull = [
        'ISO-alpha3 Code', 'Subregion ID',
        'Land Locked Developing Countries (LLDC)',
        'Least Developed Countries (LDC)',
        'Small Island Developing States (SIDS)',
        'Developed / Developing Countries'
    ]

    countries = countries.merge(
        region_countries[region_cols_to_pull],
        on='ISO-alpha3 Code',
        how='left')
    unk_subregion = region_countries.loc[region_countries['Subregion Name'] ==
                                         'Unknown']['Subregion ID'].iloc[0]
    countries.loc[countries['Code'] == 'ZZX', 'Subregion ID'] = unk_subregion

    # Step 6
    region_country_mapping = {'COK': 'NZL', 'NRU': 'AUS', 'VGB': 'GBR'}
    countries['ISO-alpha3 Code'] = countries['ISO-alpha3 Code'].replace(
        region_country_mapping)

    if econ_country is not None:
        econ_country_region_country_code_mapping = {
            'ZAR': 'COD',
            'VGB': 'GBR',
            'TMP': 'TLS',
            'WBG': 'PSE',
            'ROM': 'ROU',
            'ADO': 'AND',
        }
        econ_country['CountryCodeMerge'] = econ_country['CountryCode'].replace(
            econ_country_region_country_code_mapping)

        # Step 7
        countries = countries.merge(
            econ_country,
            left_on='ISO-alpha3 Code',
            right_on='CountryCodeMerge',
            how='left').drop(
                ['CountryCodeMerge', 'ISO-alpha3 Code'], axis=1)

        # Step 8
        # make sure econ_indicators are matched with countries on Code,
        # and remove rows from econ_indicators from countries not present
        # in the countries dataframe
        econ_indicators = econ_indicators.merge(
            countries[['CountryCode', 'Code']], on='CountryCode',
            how='inner').drop(
                ['CountryCode'], axis=1)

        countries.drop(['CountryCode'], axis=1, inplace=True)
    return countries, econ_indicators

def load_entityset(data_dir='~/olympic_games_data',
                   with_econ_data=False,
                   with_region_data=False,
                   countries_known_for_subsequent_games=False,
                   econ_path='~/olympic_games_data/economic_data/',
                   region_path='~/olympic_games_data/economic_data/',
                   since_date=None):
    '''
    1. Load data on each medal won at every summer Olympic Games
    2. Load data about each country that won at least one medal through Olympic history
    3. Do some formatting
    4. Sort on Year
    5. Normalize out Olympics as separate dataframe containing a unique row for each Olympic Games
    6. Initialize Featuretools EntitySet
    7. Optionally load region and economic data
    8. Finish loading Featuretools EntitySet

    olympic_games       (regions)
      |                    |
      |   countries __________________________
      |       |                  sports        \
    countries_at_olympic_games     |     (economic_indicators)
              |               disciplines
         medals_won____________/
              |                            athletes
     medaling_athletes ____________________/

    # do a bit more probing on analysis in simple version
    # Clean up
    '''
    # Step 1
    summer = pd.read_csv(
        os.path.join(data_dir, 'summer.csv'), encoding='utf-8')
    # winter = pd.read_csv(os.path.join(data_dir, 'winter.csv'), encoding='utf-8')

    # Step 2
    countries = pd.read_csv(
        os.path.join(data_dir, 'dictionary.csv'), encoding='utf-8')
    countries.drop(['GDP per Capita', 'Population'], axis=1, inplace=True)
    # Some countries had a '*" at their end, which we need to remove
    # in order to match with economic data
    countries['Country'] = countries['Country'].str.replace('*', '')
    countries = countries.append(
        pd.DataFrame({
            'Country': ['Unknown', 'Mixed Team'],
            'Code': ['UNK', 'ZZX']
        }),
        ignore_index=True)

    # Step 3
    # Make names First Last instead of Last, First?
    # These two lines were taken from https://www.kaggle.com/ash316/great-olympians-eda/notebook
    summer['Athlete'] = summer['Athlete'].str.split(', ').str[::-1].str.join(' ')
    summer['Athlete'] = summer['Athlete'].str.title()

    # winter['Athlete']=winter['Athlete'].str.split(', ').str[::-1].str.join(' ')
    # winter['Athlete']=winter['Athlete'].str.title()
    summer['Year'] = (pd.to_datetime(summer['Year'], format="%Y") +
                      pd.offsets.MonthEnd(6)).dt.date
    # winter['Year'] = (pd.to_datetime(winter['Year'], format="%Y")).dt.date

    # Step 4
    # summer['Games Type'] = 'Summer'
    # winter['Games Type'] = 'Winter'
    # medals_won = pd.concat([summer, winter]).sort_values(['Year'])
    medals_won = summer.sort_values(['Year'])
    if since_date is not None:
        medals_won = medals_won[medals_won['Year'] >= since_date]

    # Step 5
    medals_won['Olympic Games Name'] = medals_won['City'].str.cat(
        medals_won['Year'].astype(str), sep=' ')
    medals_won['Country'].fillna("UNK", inplace=True)

    medals_won['Olympic Games ID'] = medals_won[
        'Olympic Games Name'].factorize()[0]
    medals_won['Country'].fillna("UNK", inplace=True)
    medals_won['Country Olympic ID'] = medals_won['Country'].str.cat(
        medals_won['Olympic Games ID'].astype(str)).factorize()[0]

    unique_cols = ['Year', 'Discipline', 'Event', 'Medal']
    new_medals_won = medals_won.drop_duplicates(unique_cols, keep='first').reset_index(drop=True)
    new_medals_won['medal_id'] = new_medals_won.index
    athletes_at_olympic_games = medals_won.merge(new_medals_won[unique_cols + ['medal_id']], on=unique_cols, how='left')
    athletes_at_olympic_games = athletes_at_olympic_games[['Year', 'Athlete', 'Gender', 'medal_id']]
    medals_won = new_medals_won[[c for c in new_medals_won if c not in ['Athlete', 'Gender']]]
    athletes_at_olympic_games['Athlete Medal ID'] = athletes_at_olympic_games['Athlete'].str.cat(
        athletes_at_olympic_games['medal_id'].astype(str)).factorize()[0]

    # There were 2 duplicate athlete entries in the data, get rid of them
    athletes_at_olympic_games.drop_duplicates(['Athlete Medal ID'], inplace=True)

    # Step 6
    es = ft.EntitySet("Olympic Games")
    es.entity_from_dataframe(
        "medaling_athletes",
        athletes_at_olympic_games,
        index="Athlete Medal ID",
        time_index='Year')

    es.entity_from_dataframe(
        "medals_won",
        medals_won,
        index="medal_id",
        time_index='Year')

    # es.normalize_entity(
        # base_entity_id="medals_won",
        # new_entity_id="athletes_at_olympic_games",
        # index="Athlete Olympic ID",
        # additional_variables=[
            # 'City', 'Olympic Games Name', 'Olympic Games ID',
            # 'Country Olympic ID', 'Country', 'Gender', 'Athlete'
        # ],  # , 'Games Type'],
        # make_time_index=True,
        # new_entity_time_index='Year')
    es.normalize_entity(
        base_entity_id="medaling_athletes",
        new_entity_id="athletes",
        index="Athlete",
        make_time_index=True,
        new_entity_time_index='Year of First Medal',
        additional_variables=['Gender'])
    es.normalize_entity(
        base_entity_id="medals_won",
        new_entity_id="countries_at_olympic_games",
        index="Country Olympic ID",
        make_time_index=True,
        new_entity_time_index='Year',
        additional_variables=[
            'City', 'Olympic Games Name', 'Olympic Games ID', 'Country'
        ])
    es.normalize_entity(
        base_entity_id="countries_at_olympic_games",
        new_entity_id="olympic_games",
        index="Olympic Games ID",
        make_time_index=False,
        copy_variables=['Year'],
        additional_variables=['City'])
    es.normalize_entity(
        base_entity_id="medals_won",
        new_entity_id="disciplines",
        index="Discipline",
        new_entity_time_index='Debut Year',
        additional_variables=['Sport'])
    es.normalize_entity(
        base_entity_id="disciplines",
        new_entity_id="sports",
        new_entity_time_index='Debut Year',
        index="Sport")
    if with_econ_data or with_region_data:
        with_region_data = True

        # Step 7
        region_countries = load_regions(region_path)
        if with_econ_data:
            econ_country, econ_indicators = load_econ_indicators(
                econ_path, since_date=since_date)
        else:
            econ_country = None
            econ_indicators = None

        # Match country codes from the `countries` dataframe (from the Kaggle olympics dataset)
        # with the codes in the `regions` dataframe (from the UN)
        # and the codes in the `econ_country` and `econ_indicators` dataframes (from the Kaggle economics dataset)
        countries, econ_indicators = match_countries_to_regions_and_econ_data(
            countries, region_countries, econ_country, econ_indicators)
        # Create a dataframe that's unique on intermediate regions
        # and drop columns that are now included in countries dataframe
        # Keep columns required to normalize out subregion and region
        region_cols_to_keep = ['Subregion ID', 'Subregion Name', 'Region Name']
        regions = region_countries[region_cols_to_keep].drop_duplicates(
            'Subregion ID')

        es.entity_from_dataframe('regions', regions, index='Subregion ID')

        if with_econ_data:
            es.entity_from_dataframe(
                'econ_indicators',
                econ_indicators,
                index='IndicatorId',
                make_index=True,
                time_index='Year')

    # map countries in medals_won to those in countries
    mapping = pd.DataFrame.from_records(
        [
            ('BOH', 'AUT', 'Bohemia'),
            ('ANZ', 'AUS', 'Australasia'),
            ('TTO', 'TRI', 'Trinidad and Tobago'),
            ('RU1', 'RUS', 'Russian Empire'),
            ('TCH', 'CZE', 'Czechoslovakia'),
            ('ROU', 'ROM', 'Romania'),
            ('YUG', 'SCG', 'Yugoslavia'),
            ('URS', 'RUS', 'Soviet Union'),
            ('EUA', 'GER', 'United Team of Germany'),
            ('BWI', 'ANT', 'British West Indies'),
            ('GDR', 'GER', 'East Germany'),
            ('FRG', 'GER', 'West Germany'),
            ('EUN', 'RUS', 'Unified Team'),
            ('IOP', 'SCG', 'Yugoslavia'),
            ('SRB', 'SCG', 'Serbia'),
            ('MNE', 'SCG', 'Montenegro'),
            ('SGP', 'SIN', 'Singapore'),
        ],
        columns=['NewCode', 'Code', 'Country'])
    columns_to_pull_from_similar = [
        u'Code', u'Subregion ID', u'Land Locked Developing Countries (LLDC)',
        u'Least Developed Countries (LDC)',
        u'Small Island Developing States (SIDS)',
        u'Developed / Developing Countries', u'IncomeGroup'
    ]
    similar_countries = mapping['Code']
    similar = countries.loc[countries['Code'].isin(similar_countries),
                            columns_to_pull_from_similar]
    similar = similar.merge(
        mapping, on='Code', how='outer').drop(
            ['Code'], axis=1).rename(columns={'NewCode': 'Code'})
    countries = countries.append(
        similar, ignore_index=True).reset_index(drop=True)

    # Step 8
    es.entity_from_dataframe("countries", countries, index="Code")

    relationships = [
        ft.Relationship(es['countries']['Code'],
                        es['countries_at_olympic_games']['Country']),
        ft.Relationship(es['medals_won']['medal_id'],
                        es['medaling_athletes']['medal_id']),
    ]
    if with_region_data:
        relationships.append(
            ft.Relationship(es['regions']['Subregion ID'],
                            es['countries']['Subregion ID']))
    if with_econ_data:
        relationships.append(
            ft.Relationship(es['countries']['Code'],
                            es['econ_indicators']['Code']),
        )

    es.add_relationships(relationships)

    if countries_known_for_subsequent_games:
        es['countries_at_olympic_games'].df['Year'] -= pd.Timedelta('7 days')
    es.add_interesting_values()
    return es


def plot_feature(feature_matrix,
                 labels,
                 feature_name,
                 impute_strategy='mean',
                 kde=False,
                 log=False,
                 plot_name=None,
                 bin_edges=None,
                 separate_true_false_plots=True,
                 axes=None):
    """
    Plot conditional distribution of feature_name when label
    is True, and when label is False

    Args:
        feature_matrix (pd.DataFrame) : DataFrame containing feature data (each column is the data for one feature)
        labels (pd.Series) : Series containing the labels for each instance (for each row of feature matrix)
        feature_name (str) : Name of feature to plot
        plot_name (str, optional) : Name to give title of plot

    Note:
        This function requires the seaborn and matplotlib libraries, both installable by pip.
    """
    vals = feature_matrix[[feature_name]]
    labels = labels.astype(bool)
    if vals.dropna().shape[0] == 0:
        raise ValueError("Passed vals contains only null values")
    elif separate_true_false_plots:
        if vals[labels.values].dropna().shape[0] == 0:
            raise ValueError("Passed True vals contains only null values")
        elif vals[~labels.values].dropna().shape[0] == 0:
            raise ValueError("Passed False vals contains only null values")
    vals = Imputer(
        missing_values='NaN', strategy=impute_strategy,
        axis=0).fit_transform(vals.values).flatten()

    if log:
        pos_vals = vals > 0
        neg_vals = vals < 0
        vals[pos_vals] = np.log10(vals[pos_vals])
        vals[neg_vals] = -np.log10(-vals[neg_vals])
    if bin_edges is None:
        bin_edges = np.histogram(vals, 'auto')[1]
    import seaborn as sns
    import matplotlib.pyplot as plt
    if plot_name is not None:
        title = plot_name
    else:
        title = "Feature: {}".format(feature_name[:50])
    if axes is None:
        # Set up the matplotlib figure
        if separate_true_false_plots:
            plt_title = feature_name + " vs. label"
            f, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
        else:
            plt_title = feature_name
            f, axes = plt.subplots(1, 1, figsize=(13, 6))
        if len(feature_name) > 50:
            feature_name = feature_name[:50] + "..."

        f.suptitle(title, fontsize=20, y=1.15)
        plt.title(plt_title)

    vals1 = vals
    ax = axes
    if separate_true_false_plots:
        vals1 = vals[labels.values]
        ax = axes[0]
    sns.distplot(
        vals1,
        hist=True,
        bins=bin_edges,
        color="g",
        rug=True,
        kde=kde,
        kde_kws={"shade": False},
        ax=ax)
    ax.set_title(title)
    if separate_true_false_plots:
        ax.set_title('{}: Distribution when True'.format(title))
        sns.distplot(
            vals[~labels.values],
            hist=True,
            bins=bin_edges,
            color="r",
            rug=True,
            kde=kde,
            kde_kws={"shade": False},
            ax=axes[1])
        axes[1].set_title('{}: Distribution when False'.format(title))
    plt.tight_layout()


def plot_confusion_matrix(cm,
                          classes=[0, 1],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Source:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def run_dfs_score_and_get_feature_importances(entityset,
                                              label_file=None,
                                              agg_primitives=None,
                                              trans_primitives=None,
                                              max_depth=None,
                                              seed_features=None,
                                              feature_matrix_file=None,
                                              features_file=None,
                                              score_baseline=False,
                                              scores_file=None,
                                              importances_file=None,
                                              important_features=None,
                                              sample=None):

    if agg_primitives is None:
        agg_primitives = [
            ftypes.Sum, ftypes.Std, ftypes.Max, ftypes.Min, ftypes.Mean, ftypes.Count,
            ftypes.PercentTrue, ftypes.NUnique, ftypes.Mode, ftypes.Trend, ftypes.Last
        ]
    if trans_primitives is None:
        trans_primitives = []

    # Include our baseline predictors as seed features

    es = entityset

    # Baseline 1
    total_num_medals = ftypes.Count(es['medals_won']['medal_id'], es['countries'])
    count_num_olympics = ftypes.NUnique(
        es['countries_at_olympic_games']['Olympic Games ID'], es['countries'])
    mean_num_medals = (
        total_num_medals / count_num_olympics).rename("mean_num_medals")

    # Baseline 2
    olympic_id = ft.Feature(es['countries_at_olympic_games']['Olympic Games ID'],
                            es['medals_won'])
    num_medals_each_olympics = [
        ftypes.Count(
            es['medals_won']['medal_id'], es['countries'],
            where=olympic_id == i).rename("num_medals_olympics_{}".format(i))
        for i in es.get_all_instances('olympic_games')
    ]

    features = ft.dfs(
        entityset=es,
        target_entity="countries",
        trans_primitives=trans_primitives,
        agg_primitives=agg_primitives,
        max_depth=4,
        seed_features=seed_features + num_medals_each_olympics + [mean_num_medals],
        features_only=True,
        verbose=True)
    if important_features is not None:
        features = [f for f in features if f.get_name() in important_features]
    scores, importances = calculate_fm_score_and_get_feature_importances(features=features,
        label_file=label_file,
        feature_matrix_file=feature_matrix_file,
        features_file=features_file,
        score_baseline=score_baseline,
        scores_file=scores_file,
        importances_file=importances_file,
        sample=sample)
    return features, scores, importances


def calculate_fm_score_and_get_feature_importances(features=None,
        important_features=None,
        label_file=None,
        feature_matrix_file=None,
        load_feature_matrix=False,
        features_file=None,
        load_features=False,
        score_baseline=False,
        scores_file=None,
        importances_file=None,
        sample=None,
        entityset=None):
    if label_file is None:
        label_file = "/Users/bschreck/olympic_games_data/num_medals_by_country_labels.csv"
    label_df = pd.read_csv(label_file,
        parse_dates=['Olympics Date'],
        encoding='utf-8')
    if sample is not None:
        label_df = label_df.sample(sample)
    cutoff_times = label_df[['Country', 'Olympics Date']].rename(
        columns={'Country': 'instance_id',
                 'Olympics Date': 'time'})

    time = pd.Timestamp.now().isoformat()
    if feature_matrix_file is None:
        feature_matrix_file = "~/olympic_games_data/{}_feature_matrix_by_country_encoded.p".format(time)

    if not load_feature_matrix:
        feature_matrix = ft.calculate_feature_matrix(
            features=features,
            cutoff_time=cutoff_times,
            verbose=True)

        hv_fm, hv_f = select_high_variance_features(
            feature_matrix, features, cv_threshold=0, categorical_nunique_ratio=0)
        feature_matrix_encoded, features_encoded = ft.encode_features(hv_fm, hv_f)

        feature_matrix_encoded.to_pickle(os.path.expanduser(feature_matrix_file))
    else:
        feature_matrix_encoded = pd.read_pickle(os.path.expanduser(feature_matrix_file))
    if features_file is None:
        features_file = "~/olympic_games_data/{}_features_encoded.p".format(time)
    if not load_features:
        ft.save_features(features_encoded, os.path.expanduser(features_file))
    else:
        features_encoded = ft.load_features(os.path.expanduser(features_file), entityset)

    if important_features:
        features_encoded = [f for f in features_encoded if f.get_name() in important_features]
        feature_matrix_encoded = feature_matrix_encoded[[f.get_name() for f in features_encoded]]

    if score_baseline:
        just_baseline = feature_matrix_encoded[[
            f for f in feature_matrix_encoded
            if f.startswith("num_medals_olympics_")
        ]]
    pipeline_preprocessing = [("imputer", Imputer(
        missing_values='NaN', strategy="most_frequent", axis=0)),
                              ("scaler", RobustScaler(with_centering=True))]

    reg = RandomForestRegressor(
        max_features=0.05, min_samples_split=3, n_estimators=100, n_jobs=-1)
    #clf = RandomForestClassifier(min_samples_leaf=4, n_estimators=100, n_jobs=-1)
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

    estimator_clf = Pipeline(pipeline_preprocessing + [("rf", clf)])
    estimator_reg = Pipeline(pipeline_preprocessing + [("rf", reg)])

    labels = label_df.sort_values(['Olympics Date', 'Country'])
    dates = labels['Olympics Date']
    labels = labels['Number of Medals']
    binned_labels, bins = bin_labels(labels, [2, 6, 10, 50])
    binary_labels = (labels >= 10).astype(int)

    splitter = TimeSeriesSplitByDate(
        dates=dates, earliest_date=pd.Timestamp('1/1/1960'))

    X = feature_matrix_encoded.values
    y = labels.values
    binary_y = binary_labels.values
    binned_y = binned_labels.values

    problems_and_metrics = {
        "binary": (X, binary_y, splitter, estimator_clf, ['roc_auc', 'f1']),
        #"regression": (X, y, splitter, estimator_reg, ['r2', 'mse']),
        #"binned": (X, binned_y, splitter, estimator_clf, ['f1_micro']),
    }
    if score_baseline:
        baseline_X = just_baseline.values
        problems_and_metrics['binary_baseline'] = (baseline_X, binary_y, splitter, estimator_clf, ['roc_auc', 'f1'])
        problems_and_metrics['binned_baseline'] = (baseline_X, binned_y, splitter, estimator_clf, ['f1_micro'])
        problems_and_metrics['regression_baseline'] = (baseline_X, y, splitter, estimator_reg, ['r2', 'mse'])

    scores_by_problem = {}
    importances_by_problem = {}
    for problem_name, problem_args in problems_and_metrics.items():
        scores = fit_and_score(*problem_args)
        scores_by_problem[problem_name] = scores
        metrics = problem_args[-1]
        print "working on problem {}".format(problem_name)
        for m in metrics:
            print "   -- scoring metric {}".format(m)

        print "   -- computing feature importances"
        feature_imps = get_feature_importances(problem_args[3],
                                               feature_matrix_encoded,
                                               pd.Series(problem_args[1]),
                                               splitter)
        importances_by_problem[problem_name] = feature_imps

    if importances_file is None:
        importances_file = "~/olympic_games_data/{}_importances.p".format(time)
    with open(os.path.expanduser(importances_file), 'wb') as f:
        pck.dump(importances_by_problem, f)

    if scores_file is None:
        scores_file = "~/olympic_games_data/{}_scores.p".format(time)
    with open(os.path.expanduser(scores_file), 'wb') as f:
        pck.dump(scores_by_problem, f)
    return scores_by_problem, importances_by_problem
