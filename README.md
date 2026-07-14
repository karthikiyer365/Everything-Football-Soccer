# Everything Football / Soccer

A collection of football (soccer) data and analytics projects.

## Projects

### 1. Player Performance Analysis
FIFA player-scouting toolkit spanning 2015–2022 male and female datasets.

- **ETL pipeline** — fetches per-year FIFA datasets, selects 22 core attributes, explodes multi-position players into tidy per-position rows, and imputes missing values.
- **Interactive Dash dashboard** (`localhost:8080`) — three tabs: filtered data download, outlier / normality inspection, and player-vs-player radar comparison.
- **Statistical testing** — Shapiro, Kolmogorov–Smirnov and D'Agostino normality tests, Tukey-IQR outlier removal, and Box-Cox transforms.
- **Exploratory analysis** — PCA, Pearson correlation heatmaps, KDE and pair plots, plus position and nationality distributions via matplotlib / seaborn / plotly.
- **Stack** — pandas, NumPy, SciPy, statsmodels, scikit-learn, Dash and Plotly.

> Note: the scripts read generated `Male_Players.csv` / `Female_Players.csv` from the working directory. Run `player_data_generate.py` from inside the project folder to (re)generate them before launching the dashboard.

---

_Copyright © 2024 Karthik Sivaraman Iyer. All rights reserved._
