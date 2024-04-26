import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import boxcox
from prettytable import PrettyTable
import numpy as np

# %% Players data fetching
AllPlayers_male = pd.read_csv('Male_Players.csv')
AllPlayers_female = pd.read_csv('Female_Players.csv')

positions = AllPlayers_male['player_positions'].unique()

# %% Splitting the data into 4 subsets
GK_male = AllPlayers_male[AllPlayers_male['player_positions'] == 'GK']
df_male = AllPlayers_male[AllPlayers_male['player_positions'] != 'GK']
GK_female = AllPlayers_female[AllPlayers_female['player_positions'] == 'GK']
df_female = AllPlayers_female[AllPlayers_female['player_positions'] != 'GK']
attributes = ['pace', 'dribbling', 'defending', 'physic', 'shooting', 'passing']


# %% Box Cox for Attributes
plt.figure(figsize=(20, 21))
for i, column in enumerate(attributes):
    plt.subplot(len(attributes), 2, i + 1)
    sns.histplot(df_male[column], kde=True)
    plt.title(f'{column} Before')

p_vals = []
lambda_values = []
for i, column in enumerate(attributes):
    transformed_data, lam = boxcox(df_male[column])
    lambda_values.append(lam)
    _, p = normaltest(df_male[column])
    p_vals.append(p.round(2))
    plt.subplot(len(attributes), 2, len(attributes) + i + 1)
    sns.histplot(transformed_data, kde=True)
    plt.title(f'{column} After')


plt.suptitle('Box Cox Transformation')
plt.tight_layout()
plt.show()

table = PrettyTable()
table.field_names = ['Column', 'Lambda', 'P-Value after']

for i, column in enumerate(attributes):
    table.add_row([column, lambda_values[i].round(2), p_vals[i]])

print(table)



# %% Heatmap for Player distribution
fig = sns.heatmap(pd.DataFrame(AllPlayers_male['player_positions'].value_counts()))
plt.title('Male Players(Position wise distibution)')
plt.show()


# %% Pir Chart for league level
plt.figure(figsize=(8, 6))
league_level_counts = AllPlayers_male['league_level'].value_counts()
league_level_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('League Level Distribution')
plt.ylabel('')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# %% Pie Chart for nationality
nationality_counts = AllPlayers_male['nationality_name'].value_counts()
threshold = 2500
small_categories = nationality_counts[nationality_counts < threshold].index
nationality_counts['Other'] = nationality_counts[small_categories].sum()
nationality_counts.drop(small_categories, inplace=True)

explode = [0.1 if i % 2 == 0 else 0 for i in range(len(nationality_counts))]

plt.figure(figsize=(10, 8))
nationality_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=explode)
plt.title('Nationality Distribution of Players')
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% Histplot

fig, axes = plt.subplots(3, 2, figsize=(20, 10))

axes = axes.flatten()

for i, attribute in enumerate(attributes):
    sns.histplot(AllPlayers_male[attribute], ax=axes[i], kde=True, label=attribute)
    axes[i].set_title(f'Distribution of {attribute}')
    axes[i].set_xlabel(attribute)
    axes[i].set_ylabel('Frequency')

for ax in axes[len(attributes):]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# %% Overlaid KDE
plt.figure(figsize=(10, 6))

for attribute in attributes:
    sns.kdeplot(AllPlayers_male[attribute],
                label=attribute,
                linewidth=2,
                fill=True,
                alpha=0.6)

plt.title('KDE Plot of Attributes')
plt.xlabel('Attribute Value')
plt.ylabel('Density')

plt.legend()
plt.tight_layout()
plt.show()

# %% Bar Graph Stacked

top_10_clubs = AllPlayers_male['club_name'].value_counts().index[:5]
df_top_10 = AllPlayers_male[AllPlayers_male['club_name'].isin(top_10_clubs)]

club_position_counts = df_top_10.groupby(['club_name', 'player_positions']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
club_position_counts.plot(kind='bar', stacked=False, ax=plt.gca())
plt.title('Player Positions Distribution within Top 10 Clubs')
plt.xlabel('Club')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Player Positions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %% Bar graph grouped
top_5_clubs = AllPlayers_male['club_name'].value_counts().index[:5]
df_top_5 = AllPlayers_male[AllPlayers_male['club_name'].isin(top_5_clubs)]

def age_range(age):
    if age <= 20:
        return '<= 20'
    elif 20 < age <= 25:
        return '21-25'
    elif 25 < age <= 30:
        return '26-30'
    elif 30 < age <= 35:
        return '31-35'
    else:
        return '> 35'

df_top_5['age_range'] = df_top_5['age'].apply(age_range)

club_age_counts = df_top_5.groupby(['club_name', 'age_range']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
club_age_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Player Distribution within Top 5 Clubs by Age Range')
plt.xlabel('Club')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Age Range', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% Joint plot
sns.jointplot(x='pace', y='defending', data=AllPlayers_male, kind='hex', cmap='inferno')
plt.show()

# %% Pairplot
sns.pairplot(df_male[attributes], palette='husl')
plt.show()

# %% Value Eur as per positions
combined_df = pd.concat([df_male, GK_male, df_female, GK_female], ignore_index=True)
mean_value_eur = combined_df.groupby(['year', 'player_positions'])['value_eur'].mean().reset_index()
fig = px.line(mean_value_eur, x='year', y='value_eur', color='player_positions',
              title='Mean Value EUR Over the Years for Both Genders')
fig.update_layout(xaxis_title='Year', yaxis_title='Mean Value EUR')
fig.show(renderer='browser')

# %%
mean_value_eur_league = combined_df.groupby(['year', 'league_level'])['value_eur'].mean().reset_index()

# Create grouped bar plot
fig = px.bar(mean_value_eur_league, x='league_level', y='value_eur', color='year',
             barmode='group', title='Mean Value EUR Across Different Leagues Over the Years')

# Update layout
fig.update_layout(xaxis_title='League Level', yaxis_title='Mean Value EUR')

# Show plot
fig.show(renderer='browser')

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming df_male contains your DataFrame with attributes and 'overall' as the predictive column

# Selecting the columns for PCA (excluding the predictive column)
attributes = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

# Separating the predictive column
y = df_male['overall']

# Separating features
X = df_male[attributes]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing PCA
pca = PCA()  # Assuming you want to reduce to 2 components
principal_components = pca.fit_transform(X_scaled)
cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components_95 = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
plt.axhline(y=0.95, color='red', linestyle='--')
plt.axvline(x=num_components_95, color='black', linestyle='--')
plt.xlabel('Number of Player Attributes')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA for Player Attributes')
plt.grid(True)
plt.show()