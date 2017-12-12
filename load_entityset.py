import pandas as pd
import featuretools as ft
import os


def load_entityset(data_dir='~/olympic_games_data',
                   countries_known_for_subsequent_games=False,
                   since_date=None):
    '''
    1. Load data on each medal won at every summer Olympic Games
    2. Load data about each country that won at least one medal through Olympic history
    3. Do some formatting
    4. Sort on Year
    5. Add in a column representing a unique row for each Olympic Games
    6. Separate team medals from individual medals
    7. Create Featuretools EntitySet

    olympic_games
      |
      |   countries
      |       |                  sports
    countries_at_olympic_games     |
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

    # Step 6
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

    # Step 7
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

    es.entity_from_dataframe("countries", countries, index="Code")

    relationships = [
        ft.Relationship(es['countries']['Code'],
                        es['countries_at_olympic_games']['Country']),
        ft.Relationship(es['medals_won']['medal_id'],
                        es['medaling_athletes']['medal_id']),
    ]

    es.add_relationships(relationships)

    if countries_known_for_subsequent_games:
        es['countries_at_olympic_games'].df['Year'] -= pd.Timedelta('7 days')
    es.add_interesting_values()
    return es



