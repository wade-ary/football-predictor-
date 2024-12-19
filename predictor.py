import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

matches = pd.read_csv("matches15.csv")

def rolling_averages(group, cols, new_cols):
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["Home Score", "Away Score", "Home xG", "Away xG", "Forecast (W)", "Forecast (D)", "Forecast (L)"]
new_cols = [f"{c}_rolling" for c in cols]

train = matches.head(500)
test = matches.iloc[500:798]

train_rolling = rolling_averages(train.copy(), cols, new_cols)

predictors = ['Home Value', 'Opp Team Value', 'Home xG', 'Away xG', 'OppCode', 'Forecast (W)', 'Forecast (D)', 'Forecast (L)']

rf = RandomForestClassifier(n_estimators=50, min_samples_split=20, random_state=1)

rf.fit(train_rolling[predictors + new_cols], train_rolling["Result Value"])

test_rolling = rolling_averages(test.copy(), cols, new_cols)

preds = rf.predict(test_rolling[predictors + new_cols])

acc = accuracy_score(test_rolling["Result Value"], preds)
precision = precision_score(test_rolling["Result Value"], preds)

combined = pd.DataFrame(dict(actual=test_rolling["Result Value"], prediction=preds))

combined = combined.merge(test_rolling[["Home Team", "Away Team", "Result"]], left_index=True, right_index=True)

# Output the precision
print(f"Precision: {precision}")

