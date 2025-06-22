"""
Classification Example
=====================

This example demonstrates how to use C60.ai for a classification task.
We'll use the Iris dataset to predict flower species.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from c60 import AutoML

def main():
    # Load the Iris dataset
    print("Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize AutoML
    print("Initializing AutoML...")
    automl = AutoML(
        task='classification',
        time_budget=60,  # 1 minute
        metric='accuracy',
        n_jobs=-1,  # Use all available cores
        random_state=42
    )
    
    # Fit the model
    print("Training model...")
    automl.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = automl.predict(X_test)
    y_proba = automl.predict_proba(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Show feature importances if available
    if hasattr(automl.best_estimator_, 'feature_importances_'):
        importances = automl.best_estimator_.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': data.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importances:")
        print(feature_importance)
    
    # Save the model
    automl.save('iris_classifier.joblib')
    print("\nModel saved as 'iris_classifier.joblib'")

if __name__ == "__main__":
    main()
