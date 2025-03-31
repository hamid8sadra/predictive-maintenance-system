# src/main.py (Revised)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('data/predictive_maintenance_dataset.csv')

# Drop unnecessary columns
df = df.drop(columns=['date', 'device'])

# Define features and target
X = df.drop(columns=['failure'])
y = df['failure']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing for numeric features only
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform training data
X_train_processed = pipeline.fit_transform(X_train)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

# Transform test data
X_test_processed = pipeline.transform(X_test)

# Save the preprocessor
joblib.dump(pipeline, 'data/preprocessor.pkl')

print("Preprocessing complete. Shapes:")
print(f"X_train_balanced: {X_train_balanced.shape}, y_train_balanced: {y_train_balanced.shape}")
print(f"X_test_processed: {X_test_processed.shape}, y_test: {y_test.shape}")