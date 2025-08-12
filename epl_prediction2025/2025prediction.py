

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
import xgboost as xgb


# 1. Load Data
csv_files = [
    'epl_prediction2025/epl_data/epl2024.csv',
    'epl_prediction2025/epl_data/epl2023.csv',
    'epl_prediction2025/epl_data/epl2022.csv',
    'epl_prediction2025/epl_data/epl2021.csv',
    'epl_prediction2025/epl_data/epl2020.csv',
    'epl_prediction2025/epl_data/epl2019.csv',
    'epl_prediction2025/epl_data/epl2018.csv',
    'epl_prediction2025/epl_data/epl2017.csv',
]
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)


# 2. Preprocess Data
columns_needed = [
    'HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTHG', 'FTAG',
    'HTHG', 'HTAG', 'HTR'
]
df = df[[col for col in columns_needed if col in df.columns]].dropna()

# --- Feature Engineering: Rolling averages for each team ---

def add_rolling_features(df, window=5):
    df = df.copy()
    # Rolling averages for goals, shots, shots on target, corners, cards
    for stat in ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']:
        df[f'home_avg_{stat}'] = (
            df.groupby('HomeTeam')[stat].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        )
        df[f'away_avg_{stat}'] = (
            df.groupby('AwayTeam')[stat].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        )
        df[f'home_avg_{stat}'] = df[f'home_avg_{stat}'].fillna(df[stat].mean())
        df[f'away_avg_{stat}'] = df[f'away_avg_{stat}'].fillna(df[stat].mean())
    # Rolling form (3=win, 1=draw, 0=loss)
    def rolling_form(results):
        points = results.map({'H': 3, 'D': 1, 'A': 0})
        return points.shift().rolling(window, min_periods=1).sum()
    df['home_form'] = df.groupby('HomeTeam')['FTR'].transform(rolling_form).fillna(0)
    df['away_form'] = df.groupby('AwayTeam')['FTR'].transform(rolling_form).fillna(0)
    return df

df = add_rolling_features(df)

# Encode teams using all unique teams from both HomeTeam and AwayTeam
# Encode teams using all unique teams from both HomeTeam and AwayTeam
all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
le_team = LabelEncoder()
le_team.fit(all_teams)
df['HomeTeam_enc'] = le_team.transform(df['HomeTeam'])
df['AwayTeam_enc'] = le_team.transform(df['AwayTeam'])

# ELO rating system
def calculate_elo(df, base_elo=1500, k=20):
    elos = {team: base_elo for team in all_teams}
    home_elo_list, away_elo_list = [], []
    for idx, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        home_elo, away_elo = elos[home], elos[away]
        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)
        # Expected result
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1 - exp_home
        # Actual result
        if result == 'H':
            score_home, score_away = 1, 0
        elif result == 'A':
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5
        # Update ELOs
        elos[home] += k * (score_home - exp_home)
        elos[away] += k * (score_away - exp_away)
    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    return df

df = calculate_elo(df)

# Target encoding: H=0, D=1, A=2
df['FTR_enc'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Fill missing values (if any) with median
for col in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTHG', 'FTAG']:
    df[col] = df[col].fillna(df[col].median())



# 3. Features and Target (include rolling averages, form, ELO)
feature_cols = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'home_elo', 'away_elo',
    'home_form', 'away_form',
    # rolling averages for goals, shots, shots on target, corners, cards
    'home_avg_FTHG', 'home_avg_FTAG', 'away_avg_FTHG', 'away_avg_FTAG',
    'home_avg_HS', 'home_avg_HST', 'home_avg_HC', 'home_avg_HY', 'home_avg_HR',
    'away_avg_AS', 'away_avg_AST', 'away_avg_AC', 'away_avg_AY', 'away_avg_AR',
    'home_avg_AS', 'home_avg_AST', 'home_avg_AC', 'home_avg_AY', 'home_avg_AR',
    'away_avg_HS', 'away_avg_HST', 'away_avg_HC', 'away_avg_HY', 'away_avg_HR',
]
feature_cols = [col for col in feature_cols if col in df.columns]
X = df[feature_cols]
y = df['FTR_enc']

# For score prediction
score_X = X
score_y_home = df['FTHG']
score_y_away = df['FTAG']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
score_X_train, score_X_test, score_y_home_train, score_y_home_test = train_test_split(score_X, score_y_home, test_size=0.2, random_state=42)
_, _, score_y_away_train, score_y_away_test = train_test_split(score_X, score_y_away, test_size=0.2, random_state=42)


# 5. Model Training (XGBoost)
model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# For score prediction (regression)
home_goal_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
away_goal_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
home_goal_model.fit(score_X_train, score_y_home_train)
away_goal_model.fit(score_X_train, score_y_away_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))

# 8. User Input for Prediction

# Get rolling stats for a team (for prediction)

# Helper to get the latest stat or fallback to mean
def get_latest_or_mean(df, col, mask):
    sub = df[mask]
    return sub[col].iloc[-1] if not sub.empty else df[col].mean()

# Get all features for prediction in the same order as feature_cols
def get_features_for_prediction(home_team, away_team, df, feature_cols, le_team):
    features = []
    # Team encodings
    features.append(le_team.transform([home_team])[0])
    features.append(le_team.transform([away_team])[0])
    # ELO
    features.append(get_latest_or_mean(df, 'home_elo', df['HomeTeam'] == home_team))
    features.append(get_latest_or_mean(df, 'away_elo', df['AwayTeam'] == away_team))
    # Form
    features.append(get_latest_or_mean(df, 'home_form', df['HomeTeam'] == home_team))
    features.append(get_latest_or_mean(df, 'away_form', df['AwayTeam'] == away_team))
    # Rolling averages and stats
    for stat in feature_cols[6:]:
        if stat.startswith('home_avg_'):
            base = stat.replace('home_avg_', '')
            features.append(get_latest_or_mean(df, stat, df['HomeTeam'] == home_team))
        elif stat.startswith('away_avg_'):
            base = stat.replace('away_avg_', '')
            features.append(get_latest_or_mean(df, stat, df['AwayTeam'] == away_team))
        else:
            # fallback for any other stat
            features.append(df[stat].mean())
    return np.array([features])

def predict_match_proba(home_team, away_team):
    features = get_features_for_prediction(home_team, away_team, df, feature_cols, le_team)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        proba = model.predict_proba(features)[0]
        home_goals = home_goal_model.predict(features)[0]
        away_goals = away_goal_model.predict(features)[0]
    # --- Top 3 most likely scorelines for this matchup ---
    matchup = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]
    if not matchup.empty:
        score_counts = matchup.groupby(['FTHG', 'FTAG']).size().reset_index(name='count')
        score_counts['prob'] = score_counts['count'] / score_counts['count'].sum()
        top3 = score_counts.sort_values('prob', ascending=False).head(3)
    else:
        # Fallback to league-wide
        score_counts = df.groupby(['FTHG', 'FTAG']).size().reset_index(name='count')
        score_counts['prob'] = score_counts['count'] / len(df)
        top3 = score_counts.sort_values('prob', ascending=False).head(3)
    # Most likely scoreline is the first of top3
    if not top3.empty:
        most_likely = top3.iloc[0]
        most_likely_score = (int(most_likely['FTHG']), int(most_likely['FTAG']), float(most_likely['prob']))
    else:
        most_likely_score = (0, 0, 0.0)
    return proba, most_likely_score, top3

if __name__ == "__main__":
    print("\n--- EPL Match Outcome Predictor ---")
    valid_teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
    print("Available teams (enter names exactly as shown below):")
    print(", ".join(valid_teams))
    while True:
        home_team = input("\nEnter Home Team (or 'exit' to quit): ")
        if home_team.strip().lower() == 'exit':
            print("Exiting predictor.")
            break
        away_team = input("Enter Away Team (or 'exit' to quit): ")
        if away_team.strip().lower() == 'exit':
            print("Exiting predictor.")
            break
        if home_team not in valid_teams:
            print(f"Home team '{home_team}' not recognized. Please enter the name exactly as shown in the list.")
            continue
        if away_team not in valid_teams:
            print(f"Away team '{away_team}' not recognized. Please enter the name exactly as shown in the list.")
            continue
        proba, (hg, ag, score_p), top3 = predict_match_proba(home_team, away_team)
        # Head-to-head stats
        h2h = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]
        h2h_home_win = (h2h['FTR'] == 'H').sum()
        h2h_away_win = (h2h['FTR'] == 'A').sum()
        h2h_draw = (h2h['FTR'] == 'D').sum()
        print(f"\nHead-to-head (Home/Away): {home_team} vs {away_team}")
        print(f"Matches played: {len(h2h)} | {home_team} wins: {h2h_home_win} | Draws: {h2h_draw} | {away_team} wins: {h2h_away_win}")
        print(f"\nPrediction for {home_team} vs {away_team}:")
        print(f"Home Win: {proba[0]*100:.1f}% | Draw: {proba[1]*100:.1f}% | Away Win: {proba[2]*100:.1f}%")
        print(f"Most likely scoreline: {hg}-{ag} (probability: {score_p*100:.1f}%)")
        print("Top 3 most likely scorelines in EPL history for this matchup:")
        for _, row in top3.iterrows():
            print(f"  {int(row['FTHG'])}-{int(row['FTAG'])}: {row['prob']*100:.1f}%")