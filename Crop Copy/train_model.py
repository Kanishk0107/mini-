import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Features and Target
X = data.drop(columns=["Label"])  # Ensure this column name matches the dataset
y = data["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl!")
