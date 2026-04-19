import pandas as pd
import pickle
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("heart.csv")

# Split features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Encode categorical features
X = pd.get_dummies(X)

# Save column structure
columns = X.columns

# Train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model.fit(X, y)

# Save model and columns
pickle.dump(model, open("xgb_model.pkl", "wb"))
pickle.dump(columns, open("columns.pkl", "wb"))

print("Model trained and saved successfully!")