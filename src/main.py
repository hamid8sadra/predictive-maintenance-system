# src/main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load the dataset
df = pd.read_csv('data/predictive_maintenance_dataset.csv')
df = df.drop(columns=['date', 'device'])
X = df.drop(columns=['failure'])
y = df['failure']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)])
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Balance training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test_processed)
rf_prob = rf_model.predict_proba(X_test_processed)[:, 1]

print("Random Forest Performance:")
print(classification_report(y_test, rf_pred))
rf_auc = roc_auc_score(y_test, rf_prob)
print(f"ROC-AUC Score: {rf_auc:.4f}")

# Neural Network Model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_balanced, y_train_balanced, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

nn_prob = nn_model.predict(X_test_processed)
nn_pred = (nn_prob > 0.5).astype(int).ravel()

print("\nNeural Network Performance:")
print(classification_report(y_test, nn_pred))
nn_auc = roc_auc_score(y_test, nn_prob)
print(f"ROC-AUC Score: {nn_auc:.4f}")

# Save the better model
best_model = 'rf_model.pkl' if rf_auc > nn_auc else 'nn_model.h5'
if best_model == 'rf_model.pkl':
    joblib.dump(rf_model, 'data/rf_model.pkl')
    print("Saved Random Forest model as 'rf_model.pkl'")
else:
    nn_model.save('data/nn_model.h5')
    print("Saved Neural Network model as 'nn_model.h5'")
joblib.dump(pipeline, 'data/preprocessor.pkl')

# Prediction function for deployment
def predict_maintenance(data, preprocessor_path='data/preprocessor.pkl', model_path='data/rf_model.pkl'):
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path) if model_path.endswith('.pkl') else tf.keras.models.load_model(model_path)
    processed_data = preprocessor.transform(data)
    probs = model.predict_proba(processed_data)[:, 1] if model_path.endswith('.pkl') else model.predict(processed_data).ravel()
    return (probs > 0.5).astype(int), probs

# Example usage
sample_data = X_test.iloc[:5]
pred_labels, pred_probs = predict_maintenance(sample_data)
print("\nSample Predictions:")
print(f"Predicted Labels: {pred_labels}")
print(f"Predicted Probabilities: {pred_probs}")