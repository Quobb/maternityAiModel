#!/usr/bin/env python3
"""
Enhanced Data Preparation Script for Health Consultation AI
Includes advanced feature engineering, model selection, and validation
FIXED VERSION - Proper handling of categorical features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataPreprocessor:
    """Advanced data preprocessing with health-specific considerations"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def engineer_health_features(self, df):
        """Create health-specific engineered features"""
        print("Engineering health-specific features...")
        
        df_eng = df.copy()
        
        # BMI categories
        if 'BMI' in df_eng.columns:
            df_eng['BMI_Category'] = pd.cut(df_eng['BMI'], 
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            # Convert to string to handle NaN values
            df_eng['BMI_Category'] = df_eng['BMI_Category'].astype(str)
        
        # Blood pressure categories
        if 'SystolicBP' in df_eng.columns and 'DiastolicBP' in df_eng.columns:
            df_eng['BP_Category'] = 'Normal'
            df_eng.loc[(df_eng['SystolicBP'] >= 140) | (df_eng['DiastolicBP'] >= 90), 'BP_Category'] = 'Hypertensive'
            df_eng.loc[(df_eng['SystolicBP'] >= 120) & (df_eng['SystolicBP'] < 140) & 
                      (df_eng['DiastolicBP'] < 90), 'BP_Category'] = 'Prehypertensive'
            
            # Pulse pressure (numerical feature)
            df_eng['PulsePressure'] = df_eng['SystolicBP'] - df_eng['DiastolicBP']
        
        # Age groups
        if 'Age' in df_eng.columns:
            df_eng['AgeGroup'] = pd.cut(df_eng['Age'], 
                                      bins=[0, 25, 35, 45, 100],
                                      labels=['Young', 'Adult', 'Middle', 'Senior'])
            df_eng['AgeGroup'] = df_eng['AgeGroup'].astype(str)
        
        # Pregnancy-specific features
        if 'GestationalWeek' in df_eng.columns:
            df_eng['IsPregnant'] = (df_eng['GestationalWeek'] > 0).astype(int)
            df_eng['Trimester'] = 0
            pregnant_mask = df_eng['GestationalWeek'] > 0
            if pregnant_mask.any():
                trimester_values = pd.cut(
                    df_eng.loc[pregnant_mask, 'GestationalWeek'],
                    bins=[0, 12, 24, 42],
                    labels=[1, 2, 3]
                )
                df_eng.loc[pregnant_mask, 'Trimester'] = trimester_values
        
        # Blood sugar categories
        if 'BloodSugar' in df_eng.columns:
            df_eng['BloodSugar_Category'] = 'Normal'
            df_eng.loc[df_eng['BloodSugar'] >= 126, 'BloodSugar_Category'] = 'Diabetic'
            df_eng.loc[(df_eng['BloodSugar'] >= 100) & (df_eng['BloodSugar'] < 126), 'BloodSugar_Category'] = 'Prediabetic'
        
        # Heart rate zones
        if 'HeartRate' in df_eng.columns and 'Age' in df_eng.columns:
            max_hr = 220 - df_eng['Age']
            df_eng['HeartRate_Pct_Max'] = df_eng['HeartRate'] / max_hr * 100
            
            df_eng['HR_Zone'] = 'Normal'
            df_eng.loc[df_eng['HeartRate'] < 60, 'HR_Zone'] = 'Bradycardia'
            df_eng.loc[df_eng['HeartRate'] > 100, 'HR_Zone'] = 'Tachycardia'
        
        # Risk factor combinations (numerical features)
        risk_factors = []
        if 'BMI' in df_eng.columns:
            risk_factors.append((df_eng['BMI'] > 30).astype(int))
        if 'SystolicBP' in df_eng.columns:
            risk_factors.append((df_eng['SystolicBP'] > 140).astype(int))
        if 'BloodSugar' in df_eng.columns:
            risk_factors.append((df_eng['BloodSugar'] > 126).astype(int))
        
        if risk_factors:
            df_eng['TotalRiskFactors'] = sum(risk_factors)
        
        print(f"Created {len(df_eng.columns) - len(df.columns)} new features")
        return df_eng
    
    def handle_missing_values(self, df, strategy='advanced'):
        """Advanced missing value handling"""
        print("Handling missing values...")
        
        df_clean = df.copy()
        missing_summary = df_clean.isnull().sum()
        
        if missing_summary.sum() == 0:
            print("No missing values found")
            return df_clean
        
        print(f"Missing values found:\n{missing_summary[missing_summary > 0]}")
        
        if strategy == 'advanced':
            # Use different strategies for different types of columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            
            # For numeric columns, use median
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"Filled {col} with median: {median_val:.2f}")
            
            # For categorical columns, use mode
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"Filled {col} with mode: {mode_val}")
        
        else:
            # Simple strategy - drop rows with missing values
            df_clean = df_clean.dropna()
            print(f"Dropped rows with missing values. New shape: {df_clean.shape}")
        
        return df_clean
    
    def detect_and_handle_outliers(self, df, method='iqr', contamination=0.1):
        """Detect and handle outliers"""
        print(f"Detecting outliers using {method} method...")
        
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        outlier_summary = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                outlier_summary[col] = len(outliers)
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_clean[col]))
                outliers = df_clean[z_scores > 3]
                outlier_summary[col] = len(outliers)
                
                # Cap outliers
                threshold = df_clean[col].mean() + 3 * df_clean[col].std()
                df_clean[col] = df_clean[col].clip(upper=threshold)
        
        total_outliers = sum(outlier_summary.values())
        print(f"Handled {total_outliers} outliers across all columns")
        
        return df_clean, outlier_summary


class ModelTrainer:
    """Advanced model training with multiple algorithms and hyperparameter tuning"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_performance = {}
        
    def prepare_models(self):
        """Prepare multiple models for comparison"""
        
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Hyperparameter grids
        self.param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def train_and_compare_models(self, X_train, y_train, X_test, y_test, 
                                class_weights=None, use_smote=False):
        """Train and compare multiple models"""
        print("Training and comparing models...")
        
        # Handle class imbalance if needed
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        self.prepare_models()
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Set class weights if provided
            if class_weights is not None and hasattr(model, 'class_weight'):
                model.set_params(class_weight=class_weights)
            
            # Perform grid search for hyperparameter tuning
            if model_name in self.param_grids:
                print(f"Performing grid search for {model_name}...")
                grid_search = GridSearchCV(
                    model, 
                    self.param_grids[model_name],
                    cv=5,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train_balanced, y_train_balanced)
                best_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate model
            train_score = best_model.score(X_train_balanced, y_train_balanced)
            test_score = best_model.score(X_test, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=5)
            
            # Predictions for detailed metrics
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
            
            # Calculate AUC if possible
            auc_score = None
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            self.model_performance[model_name] = {
                'model': best_model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'auc_score': auc_score,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba
            }
            
            print(f"Train Score: {train_score:.4f}")
            print(f"Test Score: {test_score:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            if auc_score:
                print(f"AUC Score: {auc_score:.4f}")
        
        # Select best model based on test score
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['test_score'])
        self.best_model = self.model_performance[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        return best_model_name
    
    def create_detailed_evaluation_report(self, y_test, class_names, feature_names=None):
        """Create detailed evaluation report with visualizations"""
        print("\nCreating detailed evaluation report...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # 1. Model comparison bar plot
        models = list(self.model_performance.keys())
        test_scores = [self.model_performance[m]['test_score'] for m in models]
        cv_scores = [self.model_performance[m]['cv_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, test_scores, width, label='Test Score', alpha=0.8)
        axes[0, 0].bar(x + width/2, cv_scores, width, label='CV Score', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Best model confusion matrix
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['test_score'])
        best_predictions = self.model_performance[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. ROC curves (if binary classification)
        if len(class_names) == 2:
            for model_name, performance in self.model_performance.items():
                if performance['prediction_probabilities'] is not None:
                    fpr, tpr, _ = roc_curve(y_test, performance['prediction_probabilities'][:, 1])
                    auc = performance['auc_score']
                    axes[0, 2].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
            
            axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 2].set_xlabel('False Positive Rate')
            axes[0, 2].set_ylabel('True Positive Rate')
            axes[0, 2].set_title('ROC Curves')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'ROC curves only\navailable for\nbinary classification', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('ROC Curves')
        
        # 4. Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_') and feature_names is not None:
            feature_importance = self.best_model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(feature_importance)[::-1][:10]  # Top 10
            
            axes[1, 0].bar(range(len(indices)), feature_importance[indices])
            axes[1, 0].set_xlabel('Features')
            axes[1, 0].set_ylabel('Importance')
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xticks(range(len(indices)))
            axes[1, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')
        
        # 5. Prediction confidence distribution
        best_proba = self.model_performance[best_model_name]['prediction_probabilities']
        if best_proba is not None:
            max_proba = np.max(best_proba, axis=1)
            axes[1, 1].hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Prediction Confidence')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Prediction Confidence Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Prediction confidence\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Prediction Confidence')
        
        # 6. Model performance summary table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        # Create performance summary data
        summary_data = []
        for model_name, perf in self.model_performance.items():
            summary_data.append([
                model_name,
                f"{perf['test_score']:.4f}",
                f"{perf['cv_mean']:.4f}",
                f"{perf['auc_score']:.4f}" if perf['auc_score'] else "N/A"
            ])
        
        table = axes[1, 2].table(cellText=summary_data,
                                colLabels=['Model', 'Test Score', 'CV Score', 'AUC Score'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(y_test, best_predictions, target_names=class_names))


class HealthDataPipeline:
    """Complete pipeline for health data processing and model training"""
    
    def __init__(self):
        self.preprocessor = AdvancedDataPreprocessor()
        self.trainer = ModelTrainer()
        self.models = {}
        self.metadata = {}
        
    def run_complete_pipeline(self, data_file, target_column='RiskLevel', 
                             test_size=0.2, use_smote=False, save_models=True):
        """Run the complete data processing and training pipeline"""
        
        print("=" * 60)
        print("HEALTH DATA PROCESSING PIPELINE")
        print("=" * 60)
        
        # 1. Load data
        print("\n1. LOADING DATA")
        print("-" * 30)
        
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            print(f"Loaded dataset: {df.shape}")
        else:
            print(f"File {data_file} not found. Creating sample data...")
            df = self._create_sample_health_data()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{df[target_column].value_counts()}")
        
        # 2. Data preprocessing
        print("\n2. DATA PREPROCESSING")
        print("-" * 30)
        
        # Handle missing values
        df_clean = self.preprocessor.handle_missing_values(df)
        
        # Engineer features
        df_engineered = self.preprocessor.engineer_health_features(df_clean)
        
        # Handle outliers
        df_final, outlier_summary = self.preprocessor.detect_and_handle_outliers(df_engineered)
        
        print(f"Final dataset shape: {df_final.shape}")
        
        # 3. Feature preparation
        print("\n3. FEATURE PREPARATION")
        print("-" * 30)
        
        # Separate features and target
        feature_columns = [col for col in df_final.columns if col != target_column]
        
        # Handle categorical features BEFORE trying to separate by dtype
        df_processed = df_final.copy()
        
        # Encode categorical features first
        categorical_features = []
        numerical_features = []
        label_encoders = {}
        
        for col in feature_columns:
            if df_processed[col].dtype == 'object' or col in ['BMI_Category', 'BP_Category', 'AgeGroup', 'BloodSugar_Category', 'HR_Zone']:
                categorical_features.append(col)
                le = LabelEncoder()
                # Handle any NaN values by converting to string first
                df_processed[col] = df_processed[col].astype(str)
                df_processed[col] = le.fit_transform(df_processed[col])
                label_encoders[col] = le
            else:
                numerical_features.append(col)
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Now all features should be numerical
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        # Encode target if categorical
        target_encoder = None
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        # 4. Model training
        print("\n4. MODEL TRAINING")
        print("-" * 30)
        
        # Calculate class weights for imbalanced data
        class_weights = None
        if len(np.unique(y_train)) > 1:
            class_weight_values = compute_class_weight('balanced', 
                                                      classes=np.unique(y_train), 
                                                      y=y_train)
            class_weights = dict(zip(np.unique(y_train), class_weight_values))
            print(f"Class weights: {class_weights}")
        
        # Train models
        best_model_name = self.trainer.train_and_compare_models(
            X_train_scaled, y_train, X_test_scaled, y_test,
            class_weights=class_weights, use_smote=use_smote
        )
        
        # 5. Model evaluation
        print("\n5. MODEL EVALUATION")
        print("-" * 30)
        
        # Get class names for evaluation
        if target_encoder:
            class_names = target_encoder.classes_
        else:
            class_names = [str(i) for i in np.unique(y)]
        
        # Create evaluation report
        self.trainer.create_detailed_evaluation_report(y_test, class_names, feature_columns)
        
        # 6. Save models and metadata
        if save_models:
            print("\n6. SAVING MODELS")
            print("-" * 30)
            
            self.models = {
                'best_model': self.trainer.best_model,
                'scaler': scaler,
                'target_encoder': target_encoder,
                'label_encoders': label_encoders,
                'feature_columns': feature_columns
            }
            
            self.metadata = {
                'model_type': best_model_name,
                'training_date': datetime.now().isoformat(),
                'dataset_shape': df_final.shape,
                'feature_count': len(feature_columns),
                'class_names': class_names.tolist() if target_encoder else class_names,
                'performance': {
                    name: {
                        'test_score': perf['test_score'],
                        'cv_score': perf['cv_mean'],
                        'auc_score': perf['auc_score']
                    }
                    for name, perf in self.trainer.model_performance.items()
                }
            }
            
            self._save_pipeline('models')
            print("Pipeline saved successfully!")
        
        return df_final, self.models, self.metadata
    
    def _create_sample_health_data(self, n_samples=2000):
        """Create sample health data for demonstration"""
        print("Creating sample health dataset...")
        
        np.random.seed(42)
        
        # Generate base features
        age = np.random.normal(30, 8, n_samples).clip(18, 60)
        gestational_week = np.where(np.random.random(n_samples) < 0.3, 
                                   np.random.normal(25, 10, n_samples).clip(0, 40), 0)
        
        # Generate correlated health metrics
        systolic_bp = np.random.normal(120, 15, n_samples) + age * 0.3
        diastolic_bp = systolic_bp * 0.6 + np.random.normal(0, 5, n_samples)
        
        blood_sugar = np.random.normal(100, 20, n_samples) + age * 0.2
        body_temp = np.random.normal(98.6, 0.8, n_samples)
        heart_rate = np.random.normal(72, 12, n_samples) + np.random.normal(0, 5, n_samples)
        
        # BMI with age correlation
        bmi = np.random.normal(24, 4, n_samples) + (age - 30) * 0.1
        
        previous_pregnancies = np.random.poisson(1.5, n_samples).clip(0, 6)
        weight_gain = np.where(gestational_week > 0, 
                              np.random.normal(25, 10, n_samples).clip(0, 50), 0)
        
        # Create risk levels based on multiple factors
        risk_score = (
            (systolic_bp > 140) * 2 +
            (diastolic_bp > 90) * 2 +
            (blood_sugar > 126) * 2 +
            (bmi > 30) * 1 +
            (age > 40) * 1 +
            (heart_rate > 100) * 1
        )
        
        # Assign risk levels
        risk_level = ['Low Risk'] * n_samples
        for i in range(n_samples):
            if risk_score[i] >= 4:
                risk_level[i] = 'High Risk'
            elif risk_score[i] >= 2:
                risk_level[i] = 'Medium Risk'
        
        # Create DataFrame
        df = pd.DataFrame({
            'Age': age.round(0).astype(int),
            'GestationalWeek': gestational_week.round(0).astype(int),
            'SystolicBP': systolic_bp.round(0).astype(int),
            'DiastolicBP': diastolic_bp.round(0).astype(int),
            'BloodSugar': blood_sugar.round(1),
            'BodyTemp': body_temp.round(1),
            'HeartRate': heart_rate.round(0).astype(int),
            'BMI': bmi.round(1),
            'PreviousPregnancies': previous_pregnancies,
            'WeightGain': weight_gain.round(1),
            'RiskLevel': risk_level
        })
        
        # Add some realistic constraints
        df['SystolicBP'] = df['SystolicBP'].clip(80, 200)
        df['DiastolicBP'] = df['DiastolicBP'].clip(50, 120)
        df['BloodSugar'] = df['BloodSugar'].clip(60, 300)
        df['BodyTemp'] = df['BodyTemp'].clip(96, 104)
        df['HeartRate'] = df['HeartRate'].clip(40, 150)
        df['BMI'] = df['BMI'].clip(15, 50)
        
        print(f"Sample data created: {df.shape}")
        print(f"Risk level distribution:\n{df['RiskLevel'].value_counts()}")
        
        # Save sample data
        df.to_csv('sample_health_data.csv', index=False)
        print("Sample data saved as 'sample_health_data.csv'")
        
        return df
    
    def _save_pipeline(self, save_dir):
        """Save the complete pipeline"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.models['best_model'], f'{save_dir}/best_model.pkl')
        joblib.dump(self.models['scaler'], f'{save_dir}/scaler.pkl')
        
        if self.models['target_encoder']:
            joblib.dump(self.models['target_encoder'], f'{save_dir}/target_encoder.pkl')
        
        if self.models['label_encoders']:
            joblib.dump(self.models['label_encoders'], f'{save_dir}/label_encoders.pkl')
        
        # Save feature columns
        pd.DataFrame({'features': self.models['feature_columns']}).to_csv(
            f'{save_dir}/feature_columns.csv', index=False
        )
        
        # Save metadata
        import json
        with open(f'{save_dir}/pipeline_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Pipeline saved to {save_dir}/")
    
    def load_pipeline(self, model_dir):
        """Load a saved pipeline"""
        print(f"Loading pipeline from {model_dir}/...")
        
        try:
            # Load models
            models = {
                'best_model': joblib.load(f'{model_dir}/best_model.pkl'),
                'scaler': joblib.load(f'{model_dir}/scaler.pkl')
            }
            
            # Load encoders if they exist
            try:
                models['target_encoder'] = joblib.load(f'{model_dir}/target_encoder.pkl')
            except FileNotFoundError:
                models['target_encoder'] = None
            
            try:
                models['label_encoders'] = joblib.load(f'{model_dir}/label_encoders.pkl')
            except FileNotFoundError:
                models['label_encoders'] = {}
            
            # Load feature columns
            feature_df = pd.read_csv(f'{model_dir}/feature_columns.csv')
            models['feature_columns'] = feature_df['features'].tolist()
            
            # Load metadata
            import json
            with open(f'{model_dir}/pipeline_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.models = models
            self.metadata = metadata
            
            print("Pipeline loaded successfully!")
            print(f"Model type: {metadata['model_type']}")
            print(f"Features: {len(models['feature_columns'])}")
            print(f"Classes: {metadata['class_names']}")
            
            return models, metadata
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return None, None
    
    def predict(self, input_data):
        """Make predictions using the trained pipeline"""
        if not self.models:
            raise ValueError("No trained model available. Train a model first or load a saved pipeline.")
        
        # Convert to DataFrame if necessary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data.copy()
        
        # Engineer features
        input_engineered = self.preprocessor.engineer_health_features(input_df)
        
        # Select and order features
        feature_columns = self.models['feature_columns']
        
        # Handle missing features
        for col in feature_columns:
            if col not in input_engineered.columns:
                input_engineered[col] = 0  # Default value
        
        X_input = input_engineered[feature_columns].copy()
        
        # Encode categorical features
        if self.models['label_encoders']:
            for col, encoder in self.models['label_encoders'].items():
                if col in X_input.columns:
                    # Handle unseen categories
                    try:
                        X_input[col] = encoder.transform(X_input[col].astype(str))
                    except ValueError:
                        # If unseen category, use the most frequent class
                        X_input[col] = encoder.transform([encoder.classes_[0]] * len(X_input))
        
        # Scale features
        X_scaled = self.models['scaler'].transform(X_input)
        
        # Make predictions
        predictions = self.models['best_model'].predict(X_scaled)
        prediction_probabilities = self.models['best_model'].predict_proba(X_scaled)
        
        # Decode predictions if necessary
        if self.models['target_encoder']:
            predictions = self.models['target_encoder'].inverse_transform(predictions)
        
        return {
            'predictions': predictions,
            'probabilities': prediction_probabilities,
            'class_names': self.metadata['class_names']
        }


class HealthDataValidator:
    """Validator for health data quality and medical reasonableness"""
    
    def __init__(self):
        self.health_ranges = {
            'Age': (0, 120),
            'SystolicBP': (70, 250),
            'DiastolicBP': (40, 150),
            'BloodSugar': (30, 600),
            'BodyTemp': (95, 110),
            'HeartRate': (30, 220),
            'BMI': (10, 70),
            'GestationalWeek': (0, 42),
            'WeightGain': (0, 100)
        }
        
        self.critical_combinations = [
            ('SystolicBP', '>', 180, 'DiastolicBP', '>', 110),  # Hypertensive crisis
            ('BloodSugar', '>', 400, None, None, None),         # Severe hyperglycemia
            ('BodyTemp', '>', 104, None, None, None),           # Hyperthermia
            ('HeartRate', '>', 150, 'Age', '>', 65),            # Elderly tachycardia
        ]
    
    def validate_health_data(self, df):
        """Comprehensive health data validation"""
        print("Validating health data...")
        
        validation_results = {
            'total_records': len(df),
            'valid_records': 0,
            'warnings': [],
            'errors': [],
            'critical_cases': [],
            'recommendations': []
        }
        
        # Range validation
        for column, (min_val, max_val) in self.health_ranges.items():
            if column in df.columns:
                out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                if len(out_of_range) > 0:
                    validation_results['warnings'].append(
                        f"{column}: {len(out_of_range)} values out of normal range ({min_val}-{max_val})"
                    )
        
        # Medical logic validation
        if 'SystolicBP' in df.columns and 'DiastolicBP' in df.columns:
            invalid_bp = df[df['SystolicBP'] <= df['DiastolicBP']]
            if len(invalid_bp) > 0:
                validation_results['errors'].append(
                    f"Invalid blood pressure: {len(invalid_bp)} cases where systolic <= diastolic"
                )
        
        # Pregnancy logic validation
        if 'GestationalWeek' in df.columns and 'Age' in df.columns:
            invalid_pregnancy = df[(df['GestationalWeek'] > 0) & (df['Age'] < 12)]
            if len(invalid_pregnancy) > 0:
                validation_results['errors'].append(
                    f"Invalid pregnancy data: {len(invalid_pregnancy)} cases of pregnancy in children"
                )
        
        # Critical case detection
        for condition in self.critical_combinations:
            col1, op1, val1, col2, op2, val2 = condition
            
            if col1 in df.columns:
                if op1 == '>':
                    mask1 = df[col1] > val1
                elif op1 == '<':
                    mask1 = df[col1] < val1
                else:
                    mask1 = df[col1] == val1
                
                if col2 and col2 in df.columns:
                    if op2 == '>':
                        mask2 = df[col2] > val2
                    elif op2 == '<':
                        mask2 = df[col2] < val2
                    else:
                        mask2 = df[col2] == val2
                    
                    critical_cases = df[mask1 & mask2]
                else:
                    critical_cases = df[mask1]
                
                if len(critical_cases) > 0:
                    validation_results['critical_cases'].append(
                        f"Critical condition detected: {len(critical_cases)} cases of {condition}"
                    )
        
        # Calculate valid records
        validation_results['valid_records'] = len(df) - len(validation_results['errors'])
        validation_results['data_quality_score'] = validation_results['valid_records'] / len(df)
        
        # Generate recommendations
        if validation_results['data_quality_score'] < 0.9:
            validation_results['recommendations'].append("Consider data cleaning and validation")
        
        if len(validation_results['critical_cases']) > 0:
            validation_results['recommendations'].append("Review critical cases for data entry errors")
        
        if len(validation_results['warnings']) > 5:
            validation_results['recommendations'].append("High number of warnings - review data collection process")
        
        return validation_results


def main():
    """Main function to run the enhanced pipeline"""
    print("Enhanced Health Data Processing Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = HealthDataPipeline()
    validator = HealthDataValidator()
    
    # Configuration
    DATA_FILE = "maternal_health_dataset.csv"  # Change to your data file
    TARGET_COLUMN = "RiskLevel"
    USE_SMOTE = True  # Enable SMOTE for class balancing
    
    try:
        # Run the complete pipeline
        df, models, metadata = pipeline.run_complete_pipeline(
            data_file=DATA_FILE,
            target_column=TARGET_COLUMN,
            use_smote=USE_SMOTE,
            save_models=True
        )
        
        # Validate the data
        print("\n7. DATA VALIDATION")
        print("-" * 30)
        validation_results = validator.validate_health_data(df)
        
        print(f"Data quality score: {validation_results['data_quality_score']:.3f}")
        print(f"Valid records: {validation_results['valid_records']}/{validation_results['total_records']}")
        
        if validation_results['warnings']:
            print("\nWarnings:")
            for warning in validation_results['warnings'][:5]:  # Show first 5
                print(f"  - {warning}")
        
        if validation_results['critical_cases']:
            print("\nCritical cases detected:")
            for case in validation_results['critical_cases']:
                print(f"  - {case}")
        
        # Test prediction functionality
        print("\n8. TESTING PREDICTIONS")
        print("-" * 30)
        
        # Create sample test data
        test_input = {
            'Age': 32,
            'GestationalWeek': 28,
            'SystolicBP': 145,
            'DiastolicBP': 95,
            'BloodSugar': 110,
            'BodyTemp': 98.6,
            'HeartRate': 88,
            'BMI': 27.5,
            'PreviousPregnancies': 1,
            'WeightGain': 22
        }
        
        predictions = pipeline.predict(test_input)
        print(f"Test prediction: {predictions['predictions'][0]}")
        print(f"Confidence: {predictions['probabilities'][0].max():.3f}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Review the model evaluation report")
        print("2. Check data validation results")
        print("3. Test the saved models in your application")
        print("4. Consider collecting more data if needed")
        
        return df, models, metadata, validation_results
        
    except Exception as e:
        print(f"\nError in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    results = main()