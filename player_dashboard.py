# %% """Standard dependancies"""
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import boxcox
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import normaltest
from prettytable import PrettyTable
import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px

url = 'https://raw.githubusercontent.com/karthikiyer365/Football-Scout/Visualizations/'

def ks_test(x):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    return stats, p


def get_table(df):
    # Defining col_headers for all tables
    df = df
    headers = df.columns.tolist()
    headers.insert(0, " Normality tests:")

    # Creating table for passed Dataframe
    x = PrettyTable(headers)
    for index, row in df.iterrows():
        item = row.tolist()
        item.insert(0, index)
        x.add_row(item)

    maxims = df.max().tolist()
    row_normality = []
    for item in maxims:
        if item > 0.05:
            row_normality.append('Might be Normal')
        else:
            row_normality.append('Not Normal')
    row_normality.insert(0, 'Normality Result')
    x.add_row(row_normality)
    return x


# %% Data Fetch Functions


def fetch_clean_df(gender, year):
    # For Male Datasets
    file_str_ending = 'male_players.csv'
    if gender == 'M':
        df_id = f'{url}FIFA{year}_{file_str_ending}'
        df = pd.read_csv(df_id)
        # Dropping NaN values columns, unecessary columns etc.
        df = df[['short_name', 'player_positions', 'overall', 'potential', 'value_eur', 'age', 'club_name',
                 'league_level', 'nationality_name', 'preferred_foot', 'pace', 'shooting', 'passing', 'dribbling',
                 'defending',
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
                 'defending',
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

# %%
df = pd.read_csv('FIFA16_male_players.csv')

# %%
print(df)

# %%
df.info()
# %%
dfm, dff = form_data()

# %%
dfm.isna().sum()

# %%
len(dfm)
# %%
AllPlayers_male = pd.read_csv('Male_Players.csv')
AllPlayers_female = pd.read_csv('Female_Players.csv')
positions = AllPlayers_male['player_positions'].unique()
# %%
AllPlayers_male.head()

# %%
AllPlayers_female.head()