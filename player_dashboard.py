import dash
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import html, dcc
import plotly.subplots as sp
from dash.dependencies import Input, Output, State
from scipy.fft import fft
import math
import base64
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


file_str_ending = 'male_players.csv'
datasets = {}
url = 'https://raw.githubusercontent.com/karthikiyer365/Football-Scout/Visualizations/'

df_name = f'{url}FIFA15_male_players.csv'
# print(f'FIFA{i}_{file_str_ending}')
df = pd.read_csv(df_name)
print(list(df.columns))

df.dropna(axis=1, how='all')
# df.drop(df.iloc[:, 78:], inplace = True, axis = 1)
df.drop(['sofifa_id', 'player_url','long_name', 'dob','nationality', 'league_name', 'preferred_foot', 'international_reputation', 'weak_foot','skill_moves', 'work_rate', 'body_type', 'real_face', 'team_position','team_jersey_number','loaned_from','nation_position','nation_jersey_number','player_traits'], inplace = True, axis = 1)#sofifa_id, player_url,Long Name, DOB, Nationality, League name, 'preferred_foot', 'international_reputation', 'weak_foot','skill_moves', 'work_rate', 'body_type', 'real_face', 'team_position','team_jersey_number','loaned_from','nation_position','nation_jersey_number''player_traits'
df.head().to_string()
#%%
for i in range(15,23):
    df_name = f'{url}FIFA{i}_{file_str_ending}'
    print(f'FIFA{i}_{file_str_ending}')
    df = pd.read_csv(df_name)
    
    print(len(df.columns))

for i in range(16,23):
    df_name = f'{url}FIFA{i}_fe{file_str_ending}'
    print(f'FIFA{i}_fe{file_str_ending}')
    df = pd.read_csv(df_name)
    print(len(df.columns))

    