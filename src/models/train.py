# src/models/train.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple
from src.database.connection import get_merged_data, insert_predictions
from src.utils.logger import setup_logger
from src.models.preprocessing import clean_data, prepare_features, split_data
import os

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'LinearRegression': LinearRegression(),
            'XGBoost': XGBRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42),
            'DecisionTree': DecisionTreeRegressor(random_state=42)
        }
        
    def train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a single model"""
        logger.info(f"Training {type(model).__name__}...")
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, np.ndarray]:
        """Evaluate a trained model"""
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Model evaluation - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return rmse, r2, predictions
    
    def train_and_evaluate_all(self, train_data: Dict, test_data: Dict) -> Dict[str, Dict]:
        """Train and evaluate all models"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"\nTraining and evaluating {name}...")
            
            # Train
            trained_model = self.train_model(model, train_data['X'], train_data['y'])
            
            # Evaluate
            rmse, r2, predictions = self.evaluate_model(trained_model, test_data['X'], test_data['y'])
            
            results[name] = {
                'model': trained_model,
                'rmse': rmse,
                'r2': r2,
                'predictions': predictions
            }
        
        return results
    
def save_best_model(self, results: Dict[str, Dict], scaler: Any, feature_names: list) -> str:
        """Save the best performing model"""
        best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
        best_model_info = results[best_model_name]
        
        model_data = {
            'model': best_model_info['model'],
            'scaler': scaler,
            'feature_names': feature_names,
            'rmse': best_model_info['rmse'],
            'r2': best_model_info['r2']
        }
        
        # Create models directory if it doesn't exist
        models_dir = '/opt/airflow/models'
        os.makedirs(models_dir, exist_ok=True)
        
        filename = os.path.join(models_dir, f'best_model_{datetime.now().strftime("%Y%m%d")}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Best model ({best_model_name}) saved to {filename}")
        return best_model_name

def run_training_pipeline():
    """Run the complete training pipeline"""
    logger.info("Starting model training pipeline...")
    
    # Get data from database
    df = get_merged_data()
    logger.info(f"Retrieved {len(df)} rows from database")
    
    # Clean data
    df_clean = clean_data(df)
    
    # Prepare features
    X, y, scaler = prepare_features(df_clean)
    
    # Split data
    split_data_dict = split_data(X, y)
    
    # Train and evaluate models
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate_all(split_data_dict['train'], split_data_dict['test'])
    
    # Save best model
    best_model_name = trainer.save_best_model(results, scaler, X.columns.tolist())
    
    # Store predictions in database
    best_predictions = results[best_model_name]['predictions']
    test_dates = df_clean['date'].iloc[int(len(df_clean) * 0.8):]
    
    insert_predictions(
        dates=test_dates,
        actual_values=split_data_dict['test']['y'],
        predicted_values=best_predictions,
        model_name=best_model_name
    )
    
    logger.info("Training pipeline completed successfully!")
    return best_model_name, results[best_model_name]['rmse']

if __name__ == "__main__":
    run_training_pipeline()