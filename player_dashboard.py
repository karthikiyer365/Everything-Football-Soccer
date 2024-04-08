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
df_name = ''
url = f'https://raw.githubusercontent.com/karthikiyer365/Football-Scout/Visualizations/'


for i in range(15,23):
    df_name = f'{url}FIFA{i}_{file_str_ending}'
    print(df_name)
    df = pd.read_csv(df_name)
    print(df.head(5))


for i in range(16,23):
    df_name = f'{url}FIFA{i}_fe{file_str_ending}'
    df = pd.read_csv(df_name)
    print(df.head(5))
