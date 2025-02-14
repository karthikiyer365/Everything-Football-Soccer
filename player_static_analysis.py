import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import boxcox
from prettytable import PrettyTable
import numpy as np
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# %% Players data fetching
AllPlayers_male = pd.read_csv('Male_Players.csv')
AllPlayers_female = pd.read_csv('Female_Players.csv')
title_font = {'fontname': 'serif', 'color': 'blue', 'size': 'large'}
label_font = {'fontname': 'serif', 'color': 'darkred', 'size': 'large'}
positions = AllPlayers_male['player_positions'].unique()

# %% Splitting the data into 4 subsets
GK_male = AllPlayers_male[AllPlayers_male['player_positions'] == 'GK']
df_male = AllPlayers_male[AllPlayers_male['player_positions'] != 'GK']
GK_female = AllPlayers_female[AllPlayers_female['player_positions'] == 'GK']
df_female = AllPlayers_female[AllPlayers_female['player_positions'] != 'GK']
attributes_all = ['pace', 'dribbling', 'defending', 'physic', 'shooting', 'passing']
attributes_gk = ['goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking','goalkeeping_positioning','goalkeeping_reflexes']

# %% Ourlier Boxplot
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 24))

axes = axes.flatten()

for i, feature in enumerate(attributes_all):
    axes[2*i].boxplot(df_male[feature])
    axes[2*i].set_title(f'{feature} (With Outliers)', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})

    axes[2*i+1].boxplot(df_male[feature], showfliers=False)
    axes[2*i+1].set_title(f'{feature} (Without Outliers)', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})

    axes[2*i].set_xlabel(feature, fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
    axes[2*i+1].set_xlabel(feature, fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})

plt.suptitle('OUTLIER DETECTION', fontdict=title_font)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

for i, attribute in enumerate(attributes_all):
    stats.probplot(df_male[attribute], dist="norm", plot=axes[i])
    axes[i].set_title(f'QQ Plot for {attribute}', fontdict=title_font)
    axes[i].set_xlabel('Theoretical Quantiles', fontdict=label_font)
    axes[i].set_ylabel('Ordered Values', fontdict=label_font)
    axes[i].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout
plt.tight_layout()
plt.show()

# %% male 4
player_positions = df_male['player_positions'].unique()
counts = df_male['player_positions'].value_counts().values
years = df_male['year'].unique()[:4]  # Considering the first 4 years for line graphs

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].pie(counts, labels=player_positions, autopct='%1.1f%%', startangle=140)
axes[0, 0].set_title('Player Positions Distribution', fontdict=title_font)

axes[0, 1].bar(player_positions, counts, color=['blue', 'green', 'orange', 'red'])
axes[0, 1].set_title('Player Positions Counts', fontdict=title_font)
axes[0, 1].set_xlabel('Player Positions', fontdict=label_font)
axes[0, 1].set_ylabel('Counts', fontdict=label_font)

for i, attribute in enumerate(attributes_all):
    axes[1, 0].plot(years, df_male.groupby('year')[attribute].mean().values[:4], marker='o', label=attribute)

axes[1, 0].set_title('Attributes Over Years', fontdict=title_font)
axes[1, 0].set_xlabel('Year', fontdict=label_font)
axes[1, 0].set_ylabel('Attribute Value', fontdict=label_font)
axes[1, 0].legend(fontsize='medium')

axes[1, 1].plot(years, df_male.groupby('year').size().values[:4], marker='o', linestyle='--', color='purple')
axes[1, 1].set_title('Player Positions Count Over Years', fontdict=title_font)
axes[1, 1].set_xlabel('Year', fontdict=label_font)
axes[1, 1].set_ylabel('Player Positions Count', fontdict=label_font)
plt.tight_layout()
plt.show()

# %% FEmlae 4
player_positions = df_female['player_positions'].unique()
counts = df_female['player_positions'].value_counts().values
years = df_female['year'].unique()[:4]  # Considering the first 4 years for line graphs

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].pie(counts, labels=player_positions, autopct='%1.1f%%', startangle=140)
axes[0, 0].set_title('Player Positions Distribution', fontdict=title_font)

axes[0, 1].bar(player_positions, counts, color=['blue', 'green', 'orange', 'red'])
axes[0, 1].set_title('Player Positions Counts', fontdict=title_font)
axes[0, 1].set_xlabel('Player Positions', fontdict=label_font)
axes[0, 1].set_ylabel('Counts', fontdict=label_font)

for i, attribute in enumerate(attributes_all):
    axes[1, 0].plot(years, df_female.groupby('year')[attribute].mean().values[:4], marker='o', label=attribute)

axes[1, 0].set_title('Attributes Over Years', fontdict=title_font)
axes[1, 0].set_xlabel('Year', fontdict=label_font)
axes[1, 0].set_ylabel('Attribute Value', fontdict=label_font)
axes[1, 0].legend(fontsize='medium')

axes[1, 1].plot(years, df_female.groupby('year').size().values[:4], marker='o', linestyle='--', color='purple')
axes[1, 1].set_title('Player Positions Count Over Years', fontdict=title_font)
axes[1, 1].set_xlabel('Year', fontdict=label_font)
axes[1, 1].set_ylabel('Player Positions Count', fontdict=label_font)
plt.tight_layout()
plt.show()

# %% GK male
player_positions = GK_male['player_positions'].unique()
counts = GK_male['player_positions'].value_counts().values
years = GK_male['year'].unique()[:4]  # Considering the first 4 years for line graphs

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].pie(counts, labels=player_positions, autopct='%1.1f%%', startangle=140)
axes[0, 0].set_title('Player Positions Distribution', fontdict=title_font)

axes[0, 1].bar(player_positions, counts, color=['blue', 'green', 'orange', 'red'])
axes[0, 1].set_title('Player Positions Counts', fontdict=title_font)
axes[0, 1].set_xlabel('Player Positions', fontdict=label_font)
axes[0, 1].set_ylabel('Counts', fontdict=label_font)

for i, attribute in enumerate(attributes_gk):
    axes[1, 0].plot(years, df_female.groupby('year')[attribute].mean().values[:4], marker='o', label=attribute)

axes[1, 0].set_title('Attributes Over Years', fontdict=title_font)
axes[1, 0].set_xlabel('Year', fontdict=label_font)
axes[1, 0].set_ylabel('Attribute Value', fontdict=label_font)
axes[1, 0].legend(fontsize='medium')

axes[1, 1].plot(years, GK_male.groupby('year').size().values[:4], marker='o', linestyle='--', color='purple')
axes[1, 1].set_title('Player Positions Count Over Years', fontdict=title_font)
axes[1, 1].set_xlabel('Year', fontdict=label_font)
axes[1, 1].set_ylabel('Player Positions Count', fontdict=label_font)
plt.tight_layout()
plt.show()

# %%
player_positions = GK_female['player_positions'].unique()
counts = GK_female['player_positions'].value_counts().values
years = GK_female['year'].unique()[:4]  # Considering the first 4 years for line graphs

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].pie(counts, labels=player_positions, autopct='%1.1f%%', startangle=140)
axes[0, 0].set_title('Player Positions Distribution', fontdict=title_font)

axes[0, 1].bar(player_positions, counts, color=['blue', 'green', 'orange', 'red'])
axes[0, 1].set_title('Player Positions Counts', fontdict=title_font)
axes[0, 1].set_xlabel('Player Positions', fontdict=label_font)
axes[0, 1].set_ylabel('Counts', fontdict=label_font)

for i, attribute in enumerate(attributes_gk):
    axes[1, 0].plot(years, GK_female.groupby('year')[attribute].mean().values[:4], marker='o', label=attribute)

axes[1, 0].set_title('Attributes Over Years', fontdict=title_font)
axes[1, 0].set_xlabel('Year', fontdict=label_font)
axes[1, 0].set_ylabel('Attribute Value', fontdict=label_font)
axes[1, 0].legend(fontsize='medium')

axes[1, 1].plot(years, GK_female.groupby('year').size().values[:4], marker='o', linestyle='--', color='purple')
axes[1, 1].set_title('Player Positions Count Over Years', fontdict=title_font)
axes[1, 1].set_xlabel('Year', fontdict=label_font)
axes[1, 1].set_ylabel('Player Positions Count', fontdict=label_font)
plt.tight_layout()
plt.show()

# %% Box Cox for Attributes and dist plot

plt.figure(figsize=(18, 15))
for i, column in enumerate(attributes_all):
    plt.subplot(int(len(attributes_all)/2), 2, i + 1)
    sns.histplot(df_male[column], kde=True)
    plt.title(f'{column} Before', fontdict=title_font)
    plt.xlabel(column, fontdict=label_font)
    plt.ylabel('Frequency', fontdict=label_font)
plt.suptitle('Histograms before Box Cox', fontdict=title_font)
plt.tight_layout()
plt.show()

p_vals = []
lambda_values = []
plt.figure(figsize=(18, 15))

for i, column in enumerate(attributes_all):
    transformed_data, lam = boxcox(df_male[column])
    lambda_values.append(lam)
    _, p = normaltest(df_male[column])
    p_vals.append(p.round(2))
    plt.subplot(int(len(attributes_all)/2), 2, i + 1)
    sns.histplot(transformed_data, kde=True)
    plt.title(f'{column} After', fontdict=title_font)
    plt.xlabel(column, fontdict=label_font)
    plt.ylabel('Frequency', fontdict=label_font)

plt.suptitle('Box Cox Transformation', fontdict=title_font)
plt.tight_layout()
plt.show()


# Create PrettyTable
table = PrettyTable()
table.field_names = ['Column', 'Lambda', 'P-Value after']

for i, column in enumerate(attributes_all):
    table.add_row([column, lambda_values[i].round(2), p_vals[i]])

print(table)



# %%# Performing PCA
y = df_male['overall']

X = df_male[attributes_all]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()  # Assuming you want to reduce to 2 components
principal_components = pca.fit_transform(X_scaled)
cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components_95 = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
plt.axhline(y=0.95, color='red', linestyle='--')
plt.axvline(x=num_components_95, color='black', linestyle='--')
plt.xlabel('Number of Player Attributes', fontdict=label_font)
plt.ylabel('Cumulative Explained Variance Ratio', fontdict=label_font)
plt.title('PCA for Player Attributes', fontdict=title_font)
plt.grid(True)
plt.show()

# %% Pairplot
sns.set(font_scale=1.2)

sns.pairplot(df_male[attributes_all])
plt.suptitle('Pairplot for Player Attributes', fontdict=title_font)
plt.xlabel('Attributes', fontdict=label_font)
plt.ylabel('Attributes', fontdict=label_font)
plt.tight_layout()
plt.show()


# %% Correlation matrix
corr_matrix = df_male[attributes_all].corr()


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Coefficient Matrix', fontdict=title_font)
plt.xlabel('Variables', fontdict=label_font)
plt.ylabel('Variables', fontdict=label_font)
plt.show()

# %% Heatmap for Player distribution
fig = sns.heatmap(pd.DataFrame(AllPlayers_male['player_positions'].value_counts()))
plt.suptitle('Male Players (Position wise distribution)', fontdict=title_font)
plt.xlabel('Count', fontdict=label_font)  # Replace 'X Label' with your desired label
plt.ylabel('Positons', fontdict=label_font)
plt.tight_layout()
plt.show()

# %% Bar Graph Stacked

top_10_clubs = AllPlayers_male['club_name'].value_counts().index[:5]
df_top_10 = AllPlayers_male[AllPlayers_male['club_name'].isin(top_10_clubs)]

club_position_counts = df_top_10.groupby(['club_name', 'player_positions']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
club_position_counts.plot(kind='bar', stacked=False, ax=plt.gca())
plt.title('Player Positions Distribution within Top 5 Clubs')
plt.xlabel('Club')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Player Positions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% bar group
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

# %% nationality pie
nationality_counts = AllPlayers_male['nationality_name'].value_counts()
threshold = 2500
small_categories = nationality_counts[nationality_counts < threshold].index
nationality_counts['Other'] = nationality_counts[small_categories].sum()
nationality_counts.drop(small_categories, inplace=True)

explode = [0.1 if i % 2 == 0 else 0 for i in range(len(nationality_counts))]

plt.figure(figsize=(10, 8))
nationality_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=explode)
plt.title('Nationality Distribution of Players',  fontdict=title_font)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% Pir Chart for league level
plt.figure(figsize=(8, 6))
league_level_counts = AllPlayers_male['league_level'].value_counts()
league_level_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('League Level Distribution', fontdict=title_font)
plt.ylabel('')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# %% group bar
combined_df = pd.concat([df_male, GK_male, df_female, GK_female], ignore_index=True)

mean_value_eur_league = combined_df.groupby(['year', 'league_level'])['value_eur'].mean().reset_index()

fig = px.bar(mean_value_eur_league, x='league_level', y='value_eur', color='year',
             barmode='group', title='Mean Value EUR Across Different Leagues Over the Years')

fig.update_layout(xaxis_title='League Level', yaxis_title='Mean Value EUR')

fig.show(renderer='browser')

# %% Overlaid KDE
plt.figure(figsize=(10, 6))

for attribute in attributes_all:
    sns.kdeplot(AllPlayers_male[attribute],
                label=attribute,
                linewidth=2,
                fill=True,
                alpha=0.6)

plt.title('KDE Plot of Attributes', fontdict=title_font)
plt.xlabel('Attribute Value', fontdict=label_font)
plt.ylabel('Density', fontdict=label_font)

plt.legend()
plt.tight_layout()
plt.show()




# %% Joint plot
sns.jointplot(x='physic', y='defending', data=AllPlayers_male, kind='hex', cmap='inferno')
plt.title('KDE Plot of Attributes', fontdict=title_font)
plt.xlabel('Physic', fontdict=label_font)
plt.ylabel('Defending', fontdict=label_font)
plt.tight_layout()
plt.show()


# %% Value Eur line
combined_df = pd.concat([df_male, GK_male, df_female, GK_female], ignore_index=True)
mean_value_eur = combined_df.groupby(['year', 'player_positions'])['value_eur'].mean().reset_index()
fig = px.line(mean_value_eur, x='year', y='value_eur', color='player_positions',
              title='Mean Value EUR Over the Years for Both Genders')
fig.update_layout(xaxis_title='Year', yaxis_title='Mean Value EUR')
fig.show(renderer='browser')



# %% Reg plot

title_font = {'fontname': 'serif', 'color': 'blue', 'size': 'large'}
label_font = {'fontname': 'serif', 'color': 'darkred', 'size': 'large'}

plt.figure(figsize=(10, 6))
sns.regplot(x='dribbling', y='passing', data=df_male)
plt.title('Scatter Plot with Regression Line', fontdict=title_font)
plt.xlabel('Dribbling', fontdict=label_font)
plt.ylabel('Passing', fontdict=label_font)
plt.show()

# %%Create the strip plot
plt.figure(figsize=(10, 6))
sns.stripplot(x='player_positions', y='age', data=df_male)
plt.title('StripPlot', fontdict=title_font)
plt.xlabel('positions', fontdict=label_font)
plt.ylabel('Age', fontdict=label_font)
plt.show()
