# %% """Standard dependancies"""
import pandas as pd
from prettytable import PrettyTable

# %% The files have been setup in Github. Url which will help access the data
url = 'https://raw.githubusercontent.com/karthikiyer365/Football-Scout/Visualizations/'
def get_table(df):
    # Defining col_headers for all tables
    df = df
    headers = df.columns.tolist()
    headers.insert(0, " Data Descriptions")

    # Creating table for passed Dataframe
    x = PrettyTable(headers)
    for index, row in df.iterrows():
        item = row.tolist()
        item.insert(0, index)
        x.add_row(item)

    return x

def fetch_clean_df(gender, year):
    # For Male Datasets
    file_str_ending = 'male_players.csv'
    if gender == 'M':
        df_id = f'{url}FIFA{year}_{file_str_ending}'
        df = pd.read_csv(df_id)
        # Dropping NaN values columns, unecessary columns etc.
        df = df[['short_name', 'player_positions', 'overall', 'potential', 'value_eur', 'age', 'club_name',
                 'league_level', 'nationality_name', 'preferred_foot', 'pace', 'shooting', 'passing', 'dribbling',
                 'defending', 'player_face_url',
                 'physic', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                 'goalkeeping_positioning',
                 'goalkeeping_reflexes']]
        # Adding Year column for identification
        df['year'] = '01/01/' + '20' + str(year)
        df['sex'] = 'M'
        df['player_positions'] = df['player_positions'].str.split(', ')
        df = df.explode('player_positions')
        df.reset_index(inplace=True, drop=True)

    # For Female Datasets
    elif gender == 'F':
        df_id = f'{url}FIFA{year}_fe{file_str_ending}'
        df = pd.read_csv(df_id)
        df = df[['short_name', 'player_positions', 'overall', 'potential', 'value_eur', 'age', 'club_name',
                 'league_level', 'nationality_name', 'preferred_foot', 'pace', 'shooting', 'passing', 'dribbling',
                 'defending', 'player_face_url',
                 'physic', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                 'goalkeeping_positioning',
                 'goalkeeping_reflexes']]
        df['year'] = '01/01/' + '20' + str(year)
        df['sex'] = 'F'
        df['player_positions'] = df['player_positions'].str.split(', ')
        df = df.explode('player_positions')
        df.reset_index(inplace=True, drop=True)

    return df

def form_data():
    player_datasets = []
    for i in range(15, 23):
        df = fetch_clean_df('M', i)
        player_datasets.append(df)
    male_players = pd.concat(player_datasets, axis=0)
    male_players.reset_index(inplace=True)
    male_players['year'] = pd.DatetimeIndex(male_players['year']).year
    male_players.dropna(subset=['club_name', 'league_level'], inplace=True)
    male_players[['value_eur']] = male_players[['value_eur']].fillna(male_players[['value_eur']].mean())

    player_datasets = []
    for i in range(16, 23):
        df = fetch_clean_df('F', i)
        player_datasets.append(df)
    female_players = pd.concat(player_datasets, axis=0)
    female_players.reset_index(inplace=True)
    female_players['year'] = pd.DatetimeIndex(female_players['year']).year
    female_players.drop(['club_name', 'league_level', 'value_eur'], inplace=True, axis=1)
    female_players[['pace', 'dribbling', 'defending', 'physic', 'passing', 'shooting']] = female_players[
        ['pace', 'dribbling', 'defending', 'physic', 'passing', 'shooting']].fillna(female_players[[
        'pace', 'dribbling', 'defending', 'physic', 'passing', 'shooting']].mean())

    return male_players, female_players


# %% Taking a look at the data
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
df = pd.read_csv(url + 'FIFA15_male_players.csv')
get_table(df.head())

# %%
df = pd.read_csv('FIFA16_male_players.csv')

# %%
print(df.info())
print(get_table(df.head()))
df = df[['short_name', 'player_positions', 'overall', 'potential', 'value_eur', 'age', 'club_name',
         'league_level', 'nationality_name', 'preferred_foot', 'pace', 'shooting', 'passing', 'dribbling',
         'defending', 'player_face_url', 'physic', 'goalkeeping_diving', 'goalkeeping_handling',
         'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']]
df['year'] = '01/01/' + '20' + str(15)
df['year'] = pd.DatetimeIndex(df['year']).year
df['sex'] = 'F'
df['player_positions'] = df['player_positions'].str.split(', ')
df = df.explode('player_positions')
df.reset_index(inplace=True, drop=True)

print('\n\n',get_table(df.head()))
# %%
print(df.isna().sum())
df.dropna(subset=['club_name', 'league_level'], inplace=True)
df[['value_eur']] = df[['value_eur']].fillna(df[['value_eur']].mean())
print(df.isna().sum())

# %%
df_male, df_female = form_data()

print(get_table(df_male.sample(10)))
print(get_table(df_female.sample(10)))

# %%
df_male.to_csv('Male_Players.csv')
df_female.to_csv('Female_Players.csv')