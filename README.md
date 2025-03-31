# Predictive Maintenance System

## Overview
This project implements a Predictive Maintenance System using Python, leveraging advanced machine learning techniques to predict equipment failures before they occur. Designed with real-world applicability in mind, it demonstrates expertise in data preprocessing, feature engineering, and model development using Pandas, Scikit-learn, and TensorFlow. The system is built to optimize maintenance schedules, reduce downtime, and showcase senior-level proficiency in machine learning and AI.

## Step 1: Project Setup
I began by setting up the project structure and environment:
- Created a virtual environment to isolate dependencies.
- Installed core libraries: Pandas for data manipulation, Scikit-learn for traditional ML models, TensorFlow for deep learning, and Matplotlib/Seaborn for visualization.
- Initialized a Git repository with a `.gitignore` to exclude the virtual environment, ensuring a clean and professional codebase.
- Established a directory structure (`data/`, `src/`, `notebooks/`) to organize raw data, source code, and exploratory analysis.

This setup ensures reproducibility and scalability—key traits of a senior-level project tailored for industry standards.

After inspecting the dataset, I identified 'failure' as the target column (0 for no failure, 1 for failure) instead of 'Target'. I updated the EDA notebook to explicitly use this column and resolved a coding oversight, ensuring the failure distribution and feature correlations were correctly visualized. This iterative refinement underscores my attention to detail and ability to adapt to dataset-specific nuances.

## Step 3: Data Preprocessing and Feature Engineering
To prepare the data for modeling, I implemented a preprocessing pipeline in `src/main.py`:
- Dropped non-predictive columns (`date`, `device`) to focus on sensor metrics.
- Split the data into training (80%) and test (20%) sets, using stratification to preserve the failure class distribution.
- Standardized numeric features (metrics 1-9) with `StandardScaler` to ensure model stability.
- Addressed class imbalance in the training set using SMOTE (Synthetic Minority Oversampling Technique), generating synthetic failure samples.
- Encapsulated preprocessing in a `ColumnTransformer` and `Pipeline` for scalability and reproducibility, saving it as `data/preprocessor.pkl`.

This step highlights my ability to engineer features and build robust ML pipelines—key skills for deploying predictive systems in production.