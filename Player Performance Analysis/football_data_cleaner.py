import pandas as pd
import plotly.express as px
from statsmodels.graphics.gofplots import qqplot
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from scipy.stats import shapiro, normaltest, kstest
import plotly.graph_objects as go
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('Soccer Player Analytics',
                suppress_callback_exceptions=True,
                external_stylesheets=external_stylesheets)
server = app.server


def shapiro_test(x, title, a):
    stats, p = shapiro(x)
    shap_str = f'Shapiro test : \n{title} Columns : \nStatistic = {stats:.2f} \nP-vlaue of ={p:.2f}'
    alpha = a / 100
    if p > alpha:
        shap_str += f'\n{title} column is Normal'
    else:
        shap_str += f'\n{title} column is Not Normal'

    return shap_str


def ks_test(x, title, a):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)

    ks_str = f'K-S test: \n{title} dataset: \nStatistics= {stats:.2f} \nP-value = {p:.2f}'

    alpha = a / 100
    if p > alpha:
        ks_str += f'\n{title} column is Normal'
    else:
        ks_str += f'\n{title} column is Not Normal'
    return ks_str


def da_k_squared_test(x, title, a):
    stats, p = normaltest(x)
    da_str = f'DA k_squared test: \n{title} column: \nStatistics = {stats:.2f} \nP-value = {p:.2f}'

    alpha = a / 100
    if p > alpha:
        da_str += f'\n{title} column is Normal'
    else:
        da_str += f'\n{title} column is Not Normal'

    return da_str


def removing_outlier(attribute, df):
    q1 = df[attribute].quantile(0.25)
    q3 = df[attribute].quantile(0.75)
    IQR = q3 - q1
    df_short = df[(df[attribute] >= (q1 - 1.5 * IQR)) & (df[attribute] <= (q3 + 1.5 * IQR))]

    return df_short


# %% Data Fetch Functions

AllPlayers_male = pd.read_csv('Male_Players.csv')
AllPlayers_female = pd.read_csv('Female_Players.csv')
positions = AllPlayers_male['player_positions'].unique()

# %% Splitting the data into 4 subsets
GK_male = AllPlayers_male[AllPlayers_male['player_positions'] == 'GK']
df_male = AllPlayers_male[AllPlayers_male['player_positions'] != 'GK']
GK_female = AllPlayers_female[AllPlayers_female['player_positions'] == 'GK']
df_female = AllPlayers_female[AllPlayers_female['player_positions'] != 'GK']

# %% Dash Board

app.layout = html.Div(
    [
        html.Title('Football Dashboard'),
        html.H1('European Football Dashboard', style={'text-align': 'center'}),
        html.Header('Dynamic player evaluation based on player attributes', style={'text-align': 'center'}),
        dcc.Tabs(
            id='evaluation_type',
            children=[
                dcc.Tab(label='Data Download', value='blankslate'),
                dcc.Tab(label='Outlier / Normality view', value='overall'),
                dcc.Tab(label='Player vs Player Evaluation', value='indepth'),

            ],
            value='blankslate'
        ),
        dcc.Loading(
            id="loading-1",
            children=[html.Div(id='upper-layout')],
            type="default"
        )
    ]
)
# %%
static_layout = html.Div(
    [
        html.H3('Data Overview - Understanding the features of the Data', style={'text-align': 'center'}),
        html.H4('Select from following options to evaluate'),
        dcc.RadioItems(
            id='gender',
            options=[
                {'label': 'Male Players', 'value': 'M'},
                {'label': 'Female Players', 'value': 'F'}
            ],
            value='M',
            inline=True,
            style={'text-align': 'center'}
        ),
        html.Br(),
        dcc.RadioItems(
            id='eval-type',
            options=[
                {'label': 'Outlier Check', 'value': 'outlier'},
                {'label': ' Normality Check', 'value': 'normal'}
            ],
            value='outlier',
            inline=True,
            style={'text-align': 'center'}
        ),
        html.Br(),
        dcc.Dropdown(
            id='position',
            options=positions,
            multi=False,
            placeholder='Pick a playing position...',
            value='CAM'
        ),
        html.Br(),
        dcc.Loading(
            id="loading-2",
            children=[
                dcc.Dropdown(
                    id='features',
                    placeholder='Choose the metric to evaluate...',
                    value='overall'
                )
            ]
        ),
        html.Br(),
        html.H4('Based on the normality test:'),
        dcc.RadioItems(
            id='test',
            options=[
                        {'label': 'KS Test', 'value': 'ks'},
                        {'label': 'Shapiro Test', 'value': 'shapiro'},
                        {'label': 'D A Test', 'value': 'da'}
                    ],
            value='normal',
            inline=True,
            style={'text-align': 'center'}
        ),
        dcc.Input(
            id="alpha",
            type='number',
            placeholder="input significance level",
            value=1
        ),
        html.Br(),
        html.Br(),
        dcc.Loading(
            id='loading-graph',
            children=
            [
                html.H5(id='test-value', style={}),
                html.Br(),
                dcc.Graph(id='graph-1'),
                html.Br(),
                dcc.Graph(id='graph-2'),
                html.Br()
            ]
        )
    ]
)


@app.callback(
    [Output('features', 'options')],
    [Input('position', 'value')]
)
def update_features(position):
    if position == 'GK':
        return [[{'label': 'diving', 'value': 'goalkeeping_diving'},
                 {'label': 'handling', 'value': 'goalkeeping_handling'},
                 {'label': 'kicking', 'value': 'goalkeeping_kicking'},
                 {'label': 'positioning', 'value': 'goalkeeping_positioning'},
                 {'label': 'reflexes', 'value': 'goalkeeping_reflexes'},
                 {'label': 'Overall', 'value': 'overall'},
                 {'label': 'Potential', 'value': 'potential'}]]

    elif position != 'GK':
        return [[{'label': 'Pace', 'value': 'pace'},
                 {'label': 'Shooting', 'value': 'shooting'},
                 {'label': 'Passing', 'value': 'passing'},
                 {'label': 'Dribbling', 'value': 'dribbling'},
                 {'label': 'Defending', 'value': 'defending'},
                 {'label': 'Physic', 'value': 'physic'},
                 {'label': 'Overall', 'value': 'overall'},
                 {'label': 'Potential', 'value': 'potential'}]]

@app.callback(
    [Output('graph-1', 'figure'),
     Output('graph-2', 'figure'),
     Output('test-value', 'children')],
    [Input('gender', 'value'),
     Input('position', 'value'),
     Input('features', 'value'),
     Input('eval-type', 'value'),
     Input('test', 'value'),
     Input('alpha', 'value')])
def update_graph(gender, pos, feature, eval_type, test, alpha):
    if alpha is None:
        raise dash.exceptions.PreventUpdate

    if gender == 'M':
        if pos == 'GK':
            df = GK_male
        else:
            df = df_male
    else:
        if pos == 'GK':
            df = GK_female
        else:
            df = df_female

    df_plot = df[df['player_positions'] == pos]

    test_str = ''

    if eval_type == 'normal':
        if test == 'ks':
            test_str = ks_test(df_plot[feature], feature, alpha)
        elif test == 'shapiro':
            test_str = shapiro_test(df_plot[feature], feature, alpha)
        elif test == 'da':
            test_str = da_k_squared_test(df_plot[feature], feature, alpha)

        graph_1 = px.histogram(df_plot[feature],
                               nbins=30,
                               labels={'count': 'Frequency'},
                               title=f'Distribution of {pos} {feature}',
                               marginal='rug'
                               )
        fig = go.Figure()
        qqplot_data = qqplot(df_plot[feature], line='s').gca().lines
        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[0].get_xdata(),
            'y': qqplot_data[0].get_ydata(),
            'mode': 'markers',
            'marker': {
                'color': '#19d3f3'
            }
        })
        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[1].get_xdata(),
            'y': qqplot_data[1].get_ydata(),
            'mode': 'lines',
            'line': {
                'color': '#636efa'
            }

        })

        fig['layout'].update({
            'title': f'Quantile-Quantile Plot for {feature}',
            'xaxis': {
                'title': 'Theoritical Quantities',
                'zeroline': False
            },
            'yaxis': {
                'title': 'Sample Quantities'
            },
            'showlegend': False,
            'width': 800,
            'height': 700,
        })

        return graph_1, fig, test_str
    elif eval_type == 'outlier':
        len1 = len(df_plot[feature])
        graph_1 = px.box(df_plot,
                         x=feature,
                         labels={'count': 'Frequency'},
                         title='Boxplot before outliers')
        df_plot = removing_outlier(feature, df_plot)
        len2 = len(df_plot[feature])
        graph_2 = px.box(df_plot,
                         x=feature,
                         labels={'count': 'Frequency'},
                         title='Boxplot after removing outliers')
        if len1 > len2:
            test_str = 'Outliers found'
        else:
            test_str = 'No outliers found'
        return graph_1, graph_2, test_str


# %%
blank_slate = html.Div(
    [
        dcc.Checklist(
            id='sex-list',
            options=[
                {'label': 'Male Players Data', 'value': 'M'},
                {'label': 'Female Players Data', 'value': 'F'}
            ],
            value=['M']
        ),
        html.Br(),
        html.Header('Choose amount of data to be downloaded ', style={'text-align': 'center'}),
        html.Br(),
        dcc.Slider(
            id='percent-slider',
            min=0,
            max=100,
            step=1,
            marks={i: f'{i}%' for i in range(0, 101, 10)},
            value=20
        ),
        dcc.Download(id="download-button"),
        html.Button("Download Dataset", id="btn", n_clicks=0, style={'text-align': 'center'}),
    ],
    style={'text-align': 'center'}
)


@app.callback(
    Output("download-button", "data"),
    [Input('sex-list', 'value'),
     Input('percent-slider', 'value'),
     Input("btn", "n_clicks")],
    prevent_initial_call=True
)
def generate_csv(sexlist, percent, n_clicks):
    if n_clicks > 0:
        if 'M' in sexlist:
            if 'F' in sexlist:
                df = pd.concat([AllPlayers_male, AllPlayers_female])
            else:
                df = AllPlayers_male
        else:
            if 'F' in sexlist:
                df = pd.concat([AllPlayers_male, AllPlayers_female])
            else:
                df = pd.DataFrame()

        csv_string = df.sample(frac=(percent / 100)).to_csv(index=False, encoding='utf-8-sig')

        file_name = "FootballPlayers.csv"

        return dict(content=csv_string, filename=file_name)
    else:
        return None


dynamic_layout = html.Div(
    [
        html.H2('Player vs Player evaluation'),
        dcc.RadioItems(
            id='gender-d',
            options=[
                {'label': 'Male Players', 'value': 'M'},
                {'label': 'Female Players', 'value': 'F'}
            ],
            value='M',
            inline=True,
            style={'text-align': 'center'}
        ),
        html.Br(),
        html.P('Player Names'),
        dcc.Dropdown(
            id='Player-name',
            multi=True,
            optionHeight=50,
            value=['L. Messi']
        ),
        dcc.Loading(
            [
                html.Br(),
                dcc.Graph(id='line')
            ]
        )
    ]
)


@app.callback(
    Output('Player-name', 'options'),
    [Input('gender-d', 'value')]
)
def update_player_names(sex):
    if sex == 'M':
        return [{'label': name, 'value': name} for name in df_male['short_name'].unique()]
    else:
        return [{'label': name, 'value': name} for name in df_female['short_name'].unique()]


@app.callback(
    Output('line', 'figure'),
    [Input('gender-d', 'value'),
     Input('Player-name', 'value')],
    prevent_intial_callback=True
)
def update_polar_line_graph(sex, names):
    if sex == 'M':
        df = df_male
    else:
        df = df_female
        if not names:
            raise dash.exceptions.PreventUpdate

    if names:
        df_filtered = df[df['short_name'].isin(names)]
        fig = go.Figure()
        for name in names:
            player_data = df_filtered[df_filtered['short_name'] == name]
            fig.add_trace(go.Scatterpolar(
                r=player_data[['pace', 'dribbling', 'defending', 'physic', 'shooting', 'passing']].values[0],
                theta=['Pace', 'Dribbling', 'Defending', 'Physic', 'Shooting', 'Passing'],
                mode='lines+markers',
                name=name
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title='Player Attributes'
        )
    else:
        fig = go.Figure()

    return fig


# %% Main callback
@app.callback(
    Output('upper-layout', 'children'),
    [Input('evaluation_type', 'value')])
def update_layout(evaluation_type):
    if evaluation_type is None:
        raise dash.exceptions.PreventUpdate

    if evaluation_type == 'overall':
        return static_layout
    elif evaluation_type == 'indepth':
        return dynamic_layout
    elif evaluation_type == 'blankslate':
        return blank_slate


# %%

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)
