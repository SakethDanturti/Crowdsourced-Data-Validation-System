import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("Hackrena_dataset.csv")

# Drop null values
df = df.dropna()

# Target variable
TARGET = "total_compensation"

# Features
feature_columns = [
    "company",
    "education",
    "role",
    "level_designation",
    "years_of_experience",
    "location",
    "employment_type"
]

# Encode categorical columns
encoders = {}

categorical_cols = [
    "company",
    "education",
    "role",
    "level_designation",
    "location",
    "employment_type"
]

for col in categorical_cols:
    le = LabelEncoder()

    df[col] = le.fit_transform(df[col].astype(str))

    encoders[col] = le

# Features and target
X = df[feature_columns]

y = df[TARGET]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Validation predictions
predictions = model.predict(X_val)

# Evaluate
mae = mean_absolute_error(y_val, predictions)

print(f"Validation MAE: {mae}")

# Save model
joblib.dump(model, "salary_model.pkl")

# Save encoders
joblib.dump(encoders, "encoder.pkl")

print("Model training completed")