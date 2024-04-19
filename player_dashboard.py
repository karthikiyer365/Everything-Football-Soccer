# %%
"""Standard dependancies"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

# %% '''Dash Dependencies'''
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import math

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

'''Static part of code setup'''
# %% Data Set-up (static)
file_str_ending = 'male_players.csv'
url = 'https://raw.githubusercontent.com/karthikiyer365/Football-Scout/Visualizations/'
'''Understanding and cleaning data'''

# %% Dynamic Data Fetching
df_name = f'{url}FIFA15_male_players.csv'
df = pd.read_csv(df_name)
# print(df.head(5))

# %% Cleaning, Dataset Fetching
def fetch_clean_df(gender, year):
    # For Male Datasets
    if gender == 'M':
        df_id = f'{url}FIFA{year}_{file_str_ending}'
        df = pd.read_csv(df_id)
        # Dropping NaN values columns, unecessary columns etc.
        df.drop(['nation_logo_url', 'club_flag_url', 'club_logo_url', 'goalkeeping_speed',
                 'release_clause_eur', 'mentality_composure', 'player_traits', 'player_tags',
                 'release_clause_eur', 'nation_jersey_number', 'nation_position', 'nation_team_id',
                 'club_contract_valid_until', 'club_joined', 'club_loaned_from', 'club_jersey_number',
                 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
                 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
                 'player_face_url',
                 'nation_flag_url'], inplace=True, axis=1)

        # Adding Year column for identification
        df['year'] = year
        df['sex'] = 'M'
        df['player_positions'] = df['player_positions'].str.split(', ')
        df = df.explode('player_positions')
        df.reset_index(inplace=True, drop=True)

    # For Female Datasets
    else:
        df_id = f'{url}FIFA{year}_fe{file_str_ending}'
        df = pd.read_csv(df_id)
        df.drop(['nation_logo_url', 'club_flag_url', 'club_logo_url', 'goalkeeping_speed',
                 'release_clause_eur', 'mentality_composure', 'player_traits', 'player_tags',
                 'release_clause_eur', 'nation_jersey_number', 'nation_position', 'nation_team_id',
                 'club_contract_valid_until', 'club_joined', 'club_loaned_from', 'club_jersey_number',
                 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
                 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
                 'player_face_url',
                 'nation_flag_url'], inplace=True, axis=1)

        df['year'] = year
        df['sex'] = 'F'
        df['player_positions'] = df['player_positions'].str.split(', ')
        df = df.explode('player_positions')
        df.reset_index(inplace=True, drop=True)

    return df


# %%
player_datasets = []
length = 0

# male player data fetching
for i in range(15, 23):
    df = fetch_clean_df('M', i)
    dataset_name = 'M' + str(i)
    player_datasets.append(df)
    length += len(df)

# female player data fetching
for i in range(16, 23):
    df = fetch_clean_df('F', i)
    dataset_name = 'F' + str(i)
    player_datasets.append(df)
    length += len(df)

# %%
AllPlayers = pd.concat(player_datasets, axis=0)
AllPlayers.reset_index(inplace=True)
print((AllPlayers['year'][0:3]))
