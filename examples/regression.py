"""
Regression Example
================

This example demonstrates how to use C60.ai for a regression task.
We'll use the California Housing dataset to predict house prices.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from c60 import AutoML

def main():
    # Load the California Housing dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(X, columns=feature_names)
    df['MedHouseVal'] = y
    
    # Display dataset information
    print(f"Dataset shape: {X.shape}")
    print("\nFeature names:", feature_names)
    print("\nFirst few rows:")
    print(df.head())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize AutoML for regression
    print("\nInitializing AutoML for regression...")
    automl = AutoML(
        task='regression',
        time_budget=120,  # 2 minutes
        metric='neg_mean_squared_error',
        n_jobs=-1,  # Use all available cores
        random_state=42
    )
    
    # Fit the model
    print("Training model...")
    automl.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = automl.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation:")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- MAE: {mae:.4f}")
    print(f"- R²: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(True)
    plt.savefig('regression_plot.png')
    print("\nSaved regression plot as 'regression_plot.png'")
    
    # Show feature importances if available
    if hasattr(automl.best_estimator_, 'feature_importances_'):
        importances = automl.best_estimator_.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importances:")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Saved feature importance plot as 'feature_importance.png'")
    
    # Save the model
    automl.save('california_housing_regressor.joblib')
    print("\nModel saved as 'california_housing_regressor.joblib'")

if __name__ == "__main__":
    main()
