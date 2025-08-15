#!/usr/bin/env python3
"""
Data Preparation Script for Health Consultation API
Use this to prepare your actual dataset for training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_and_explore_data(file_path: str):
    """Load and explore the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.info())
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nRisk Level distribution:")
    print(df['RiskLevel'].value_counts())
    
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    print("\nCleaning data...")
    
    # Handle missing values
    df = df.dropna()  # or use df.fillna() for specific strategies
    
    # Remove outliers (optional)
    numerical_cols = ['Age', 'GestationalWeek', 'SystolicBP', 'DiastolicBP', 
                     'BloodSugar', 'BodyTemp', 'HeartRate', 'BMI', 'WeightGain']
    
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"Outliers in {col}: {len(outliers)}")
            
            # Optionally remove outliers
            # df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def create_visualizations(df):
    """Create visualizations for data exploration"""
    print("\nCreating visualizations...")
    
    plt.figure(figsize=(15, 12))
    
    # Risk level distribution
    plt.subplot(2, 3, 1)
    df['RiskLevel'].value_counts().plot(kind='bar')
    plt.title('Risk Level Distribution')
    plt.xticks(rotation=45)
    
    # Age distribution by risk level
    plt.subplot(2, 3, 2)
    for risk in df['RiskLevel'].unique():
        plt.hist(df[df['RiskLevel'] == risk]['Age'], alpha=0.7, label=risk)
    plt.title('Age Distribution by Risk Level')
    plt.xlabel('Age')
    plt.legend()
    
    # BMI vs Risk Level
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='RiskLevel', y='BMI')
    plt.title('BMI by Risk Level')
    
    # Blood pressure correlation
    plt.subplot(2, 3, 4)
    plt.scatter(df['SystolicBP'], df['DiastolicBP'], 
                c=df['RiskLevel'].map({'Low': 0, 'Medium': 1, 'High': 2}))
    plt.xlabel('Systolic BP')
    plt.ylabel('Diastolic BP')
    plt.title('Blood Pressure Relationship')
    
    # Correlation heatmap
    plt.subplot(2, 3, 5)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    
    # Gestational week distribution
    plt.subplot(2, 3, 6)
    pregnant_data = df[df['GestationalWeek'] > 0]
    if len(pregnant_data) > 0:
        plt.hist(pregnant_data['GestationalWeek'], bins=20)
        plt.title('Gestational Week Distribution')
        plt.xlabel('Gestational Week')
    else:
        plt.text(0.5, 0.5, 'No pregnancy data', ha='center', va='center')
        plt.title('Gestational Week Distribution')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'data_exploration.png'")

def prepare_features(df):
    """Prepare features for model training"""
    print("\nPreparing features...")
    
    # Define feature columns (matching your dataset)
    feature_columns = ['Age', 'GestationalWeek', 'SystolicBP', 'DiastolicBP', 
                      'BloodSugar', 'BodyTemp', 'HeartRate', 'BMI', 
                      'PreviousPregnancies', 'WeightGain']
    
    # Check if all columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    print(f"Using features: {feature_columns}")
    
    X = df[feature_columns]
    y_risk = df['RiskLevel']
    y_recommendation = df['HealthRecommendation'] if 'HealthRecommendation' in df.columns else None
    
    return X, y_risk, y_recommendation, feature_columns

def train_and_evaluate_models(X, y_risk, y_recommendation, feature_columns):
    """Train and evaluate the ML models"""
    print("\nTraining models...")
    
    # Prepare scalers and encoders
    scaler = StandardScaler()
    risk_encoder = LabelEncoder()
    
    # Scale features
    X_scaled = scaler.fit_transform(X)
    y_risk_encoded = risk_encoder.fit_transform(y_risk)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_risk_encoded, test_size=0.2, random_state=42, stratify=y_risk_encoded
    )
    
    # Train risk prediction model
    risk_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    risk_model.fit(X_train, y_train)
    
    # Evaluate risk model
    y_pred = risk_model.predict(X_test)
    
    print("\nRisk Prediction Model Performance:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=risk_encoder.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': risk_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importance for Risk Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Train recommendation model if data is available
    rec_model = None
    rec_encoder = None
    
    if y_recommendation is not None:
        print("\nTraining recommendation model...")
        rec_encoder = LabelEncoder()
        y_rec_encoded = rec_encoder.fit_transform(y_recommendation)
        
        rec_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        rec_model.fit(X_scaled, y_rec_encoded)
        
        print("Recommendation model trained successfully")
    
    return {
        'risk_model': risk_model,
        'recommendation_model': rec_model,
        'scaler': scaler,
        'risk_encoder': risk_encoder,
        'rec_encoder': rec_encoder,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance
    }

def save_models(models_dict, save_dir='models'):
    """Save trained models and preprocessors"""
    print(f"\nSaving models to '{save_dir}' directory...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each component
    joblib.dump(models_dict['risk_model'], f'{save_dir}/risk_model.pkl')
    joblib.dump(models_dict['scaler'], f'{save_dir}/scaler.pkl')
    joblib.dump(models_dict['risk_encoder'], f'{save_dir}/risk_encoder.pkl')
    
    if models_dict['recommendation_model'] is not None:
        joblib.dump(models_dict['recommendation_model'], f'{save_dir}/rec_model.pkl')
        joblib.dump(models_dict['rec_encoder'], f'{save_dir}/rec_encoder.pkl')
    
    # Save feature columns and importance
    pd.DataFrame({'features': models_dict['feature_columns']}).to_csv(
        f'{save_dir}/feature_columns.csv', index=False
    )
    models_dict['feature_importance'].to_csv(
        f'{save_dir}/feature_importance.csv', index=False
    )
    
    print("Models saved successfully!")
    
    # Create a model info file
    model_info = {
        'risk_model_type': 'RandomForestClassifier',
        'n_features': len(models_dict['feature_columns']),
        'risk_classes': models_dict['risk_encoder'].classes_.tolist(),
        'feature_columns': models_dict['feature_columns']
    }
    
    if models_dict['rec_encoder'] is not None:
        model_info['recommendation_classes'] = models_dict['rec_encoder'].classes_.tolist()
    
    import json
    with open(f'{save_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

def load_models(models_dir='models'):
    """Load saved models for use in API"""
    print(f"\nLoading models from '{models_dir}' directory...")
    
    try:
        models = {}
        models['risk_model'] = joblib.load(f'{models_dir}/risk_model.pkl')
        models['scaler'] = joblib.load(f'{models_dir}/scaler.pkl')
        models['risk_encoder'] = joblib.load(f'{models_dir}/risk_encoder.pkl')
        
        # Try to load recommendation model (might not exist)
        try:
            models['recommendation_model'] = joblib.load(f'{models_dir}/rec_model.pkl')
            models['rec_encoder'] = joblib.load(f'{models_dir}/rec_encoder.pkl')
        except FileNotFoundError:
            models['recommendation_model'] = None
            models['rec_encoder'] = None
            print("Recommendation model not found, will use fallback")
        
        # Load feature columns
        feature_df = pd.read_csv(f'{models_dir}/feature_columns.csv')
        models['feature_columns'] = feature_df['features'].tolist()
        
        print("Models loaded successfully!")
        return models
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def validate_data_quality(df):
    """Validate data quality and provide recommendations"""
    print("\nValidating data quality...")
    
    issues = []
    recommendations = []
    
    # Check dataset size
    if len(df) < 1000:
        issues.append(f"Small dataset size: {len(df)} rows")
        recommendations.append("Consider collecting more data for better model performance")
    
    # Check class balance
    risk_counts = df['RiskLevel'].value_counts()
    min_class_ratio = risk_counts.min() / risk_counts.max()
    
    if min_class_ratio < 0.1:
        issues.append("Highly imbalanced classes detected")
        recommendations.append("Consider using techniques like SMOTE or class weights")
    
    # Check for missing values
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > 10]
    
    if len(high_missing) > 0:
        issues.append(f"High missing values in: {high_missing.index.tolist()}")
        recommendations.append("Consider imputation strategies or feature engineering")
    
    # Check for constant features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    constant_features = []
    
    for col in numerical_cols:
        if df[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        issues.append(f"Constant features detected: {constant_features}")
        recommendations.append("Remove constant features as they don't provide information")
    
    # Print results
    if issues:
        print("Data Quality Issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("No major data quality issues detected!")
    
    return issues, recommendations

def main():
    """Main function to run the complete data preparation pipeline"""
    print("Health Data Preparation Pipeline")
    print("=" * 50)
    
    # Replace with your actual data file path
    DATA_FILE = "maternal_health_dataset.csv"  # Change this to your actual file path
    
    # Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found!")
        print("Please update the DATA_FILE variable with the correct path to your dataset.")
        
        # Create a sample dataset for demonstration
        print("\nCreating sample dataset for demonstration...")
        from sklearn.datasets import make_classification
        
        # Generate sample data
        X_sample, y_sample = make_classification(
            n_samples=2000,
            n_features=10,
            n_classes=3,
            n_redundant=2,
            random_state=42
        )
        
        # Create DataFrame with proper column names
        feature_names = ['Age', 'GestationalWeek', 'SystolicBP', 'DiastolicBP', 
                        'BloodSugar', 'BodyTemp', 'HeartRate', 'BMI', 
                        'PreviousPregnancies', 'WeightGain']
        
        df_sample = pd.DataFrame(X_sample, columns=feature_names)
        
        # Scale features to realistic ranges
        df_sample['Age'] = np.clip(df_sample['Age'] * 5 + 28, 18, 45)
        df_sample['GestationalWeek'] = np.clip(df_sample['GestationalWeek'] * 10 + 20, 0, 40)
        df_sample['SystolicBP'] = np.clip(df_sample['SystolicBP'] * 15 + 120, 90, 180)
        df_sample['DiastolicBP'] = np.clip(df_sample['DiastolicBP'] * 10 + 80, 60, 110)
        df_sample['BloodSugar'] = np.clip(df_sample['BloodSugar'] * 20 + 100, 70, 200)
        df_sample['BodyTemp'] = np.clip(df_sample['BodyTemp'] * 2 + 98.6, 97, 102)
        df_sample['HeartRate'] = np.clip(df_sample['HeartRate'] * 10 + 75, 50, 120)
        df_sample['BMI'] = np.clip(df_sample['BMI'] * 4 + 24, 18, 35)
        df_sample['PreviousPregnancies'] = np.clip(df_sample['PreviousPregnancies'], 0, 5).astype(int)
        df_sample['WeightGain'] = np.clip(df_sample['WeightGain'] * 10 + 25, 0, 50)
        
        # Add risk levels and recommendations
        risk_labels = ['Low', 'Medium', 'High']
        df_sample['RiskLevel'] = [risk_labels[i] for i in y_sample]
        
        recommendations = [
            'Maintain healthy lifestyle with regular exercise',
            'Monitor blood pressure and follow balanced diet',
            'Seek immediate medical consultation'
        ]
        df_sample['HealthRecommendation'] = [recommendations[i] for i in y_sample]
        
        # Save sample dataset
        df_sample.to_csv('sample_health_data.csv', index=False)
        print("Sample dataset created as 'sample_health_data.csv'")
        
        df = df_sample
    else:
        # Load actual dataset
        df = load_and_explore_data(DATA_FILE)
    
    # Validate data quality
    validate_data_quality(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Create visualizations
    create_visualizations(df_clean)
    
    # Prepare features
    X, y_risk, y_recommendation, feature_columns = prepare_features(df_clean)
    
    # Train models
    models = train_and_evaluate_models(X, y_risk, y_recommendation, feature_columns)
    
    # Save models
    save_models(models)
    
    # Test loading models
    loaded_models = load_models()
    
    if loaded_models:
        print("\nPipeline completed successfully!")
        print("You can now use these models in your FastAPI application.")
        print("\nNext steps:")
        print("1. Update the FastAPI code to load your trained models")
        print("2. Run the API server: uvicorn main:app --reload")
        print("3. Test the endpoints using the client script")
    
    return df_clean, models

if __name__ == "__main__":
    df, models = main()