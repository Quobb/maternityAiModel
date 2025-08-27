import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic maternal health dataset
def generate_synthetic_maternal_data(n_samples=2000):
    """
    Generate synthetic maternal health data for pregnant mothers
    """
    np.random.seed(42)
    
    # Age distribution (realistic for pregnant mothers)
    age = np.random.normal(28, 6, n_samples)
    age = np.clip(age, 16, 45).astype(int)
    
    # Gestational week (pregnancy week)
    gestational_week = np.random.uniform(12, 40, n_samples)
    
    # Systolic Blood Pressure (adjusted for pregnancy)
    systolic_bp = np.random.normal(115, 15, n_samples)
    # Higher BP for older mothers and later in pregnancy
    systolic_bp += (age - 25) * 0.5 + (gestational_week - 20) * 0.3
    systolic_bp = np.clip(systolic_bp, 90, 200)
    
    # Diastolic Blood Pressure
    diastolic_bp = systolic_bp * 0.65 + np.random.normal(0, 5, n_samples)
    diastolic_bp = np.clip(diastolic_bp, 60, 120)
    
    # Blood Sugar (mg/dL) - pregnancy affects glucose levels
    blood_sugar = np.random.normal(95, 20, n_samples)
    # Gestational diabetes risk increases with age and week
    blood_sugar += (age - 25) * 0.8 + (gestational_week - 20) * 0.2
    blood_sugar = np.clip(blood_sugar, 70, 200)
    
    # Body Temperature (Fahrenheit)
    body_temp = np.random.normal(98.6, 0.8, n_samples)
    body_temp = np.clip(body_temp, 97.0, 102.0)
    
    # Heart Rate (pregnancy increases resting heart rate)
    heart_rate = np.random.normal(85, 12, n_samples)
    heart_rate += (gestational_week - 12) * 0.3  # Increases during pregnancy
    heart_rate = np.clip(heart_rate, 60, 120)
    
    # BMI (Body Mass Index) - important for pregnancy
    bmi = np.random.normal(24, 5, n_samples)
    bmi = np.clip(bmi, 16, 40)
    
    # Previous pregnancies
    previous_pregnancies = np.random.poisson(1, n_samples)
    previous_pregnancies = np.clip(previous_pregnancies, 0, 8)
    
    # Weight gain during pregnancy (pounds)
    target_weight_gain = np.where(bmi < 18.5, 35, 
                         np.where(bmi < 25, 30,
                         np.where(bmi < 30, 20, 15)))
    weight_gain = target_weight_gain + np.random.normal(0, 8, n_samples)
    weight_gain = np.clip(weight_gain, -10, 60)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'GestationalWeek': gestational_week.round(1),
        'SystolicBP': systolic_bp.round(0),
        'DiastolicBP': diastolic_bp.round(0),
        'BloodSugar': blood_sugar.round(1),
        'BodyTemp': body_temp.round(1),
        'HeartRate': heart_rate.round(0),
        'BMI': bmi.round(1),
        'PreviousPregnancies': previous_pregnancies,
        'WeightGain': weight_gain.round(1)
    })
    
    # Generate risk levels based on clinical criteria
    risk_level = []
    health_recommendations = []
    
    for _, row in data.iterrows():
        risk_score = 0
        
        # Age risk factors
        if row['Age'] < 18 or row['Age'] > 35:
            risk_score += 2
        
        # Blood pressure risk
        if row['SystolicBP'] > 140 or row['DiastolicBP'] > 90:
            risk_score += 3  # Hypertension
        elif row['SystolicBP'] > 130 or row['DiastolicBP'] > 85:
            risk_score += 1  # Pre-hypertension
        
        # Blood sugar risk (gestational diabetes)
        if row['BloodSugar'] > 140:
            risk_score += 3
        elif row['BloodSugar'] > 125:
            risk_score += 2
        
        # BMI risk
        if row['BMI'] < 18.5 or row['BMI'] > 30:
            risk_score += 2
        
        # Heart rate risk
        if row['HeartRate'] > 100 or row['HeartRate'] < 60:
            risk_score += 1
        
        # Body temperature risk
        if row['BodyTemp'] > 100.4:
            risk_score += 2
        
        # Weight gain risk
        expected_gain = 30 if row['BMI'] < 25 else 20
        if abs(row['WeightGain'] - expected_gain) > 15:
            risk_score += 1
        
        # Multiple pregnancies
        if row['PreviousPregnancies'] > 4:
            risk_score += 1
        
        # Assign risk level
        if risk_score >= 6:
            risk_level.append('High Risk')
        elif risk_score >= 3:
            risk_level.append('Medium Risk')
        else:
            risk_level.append('Low Risk')
        
        # Generate health recommendations
        if row['BloodSugar'] > 125 or row['BMI'] > 30:
            health_recommendations.append('Nutrition Focus')
        elif row['BMI'] < 20 or abs(row['WeightGain'] - expected_gain) > 10:
            health_recommendations.append('Exercise Focus')
        else:
            health_recommendations.append('Wellness Focus')
    
    data['RiskLevel'] = risk_level
    data['HealthRecommendation'] = health_recommendations
    
    return data

# Create enhanced models
def create_risk_model(input_shape=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(3, activation='softmax')  # Low, Medium, High Risk
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

def create_health_model(input_shape=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(3, activation='softmax')  # Nutrition, Exercise, Wellness
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

# Training function
def train_maternal_health_models():
    print("Generating synthetic maternal health dataset...")
    
    # Generate data
    df = generate_synthetic_maternal_data(2000)
    
    # Save the generated dataset
    df.to_csv('maternal_health_dataset.csv', index=False)
    print(f"Generated dataset with {len(df)} samples")
    
    # Display dataset statistics
    print("\nDataset Overview:")
    print(df.describe())
    print(f"\nRisk Level Distribution:")
    print(df['RiskLevel'].value_counts())
    print(f"\nHealth Recommendation Distribution:")
    print(df['HealthRecommendation'].value_counts())
    
    # Prepare features
    feature_columns = ['Age', 'GestationalWeek', 'SystolicBP', 'DiastolicBP', 
                      'BloodSugar', 'BodyTemp', 'HeartRate', 'BMI', 
                      'PreviousPregnancies', 'WeightGain']
    
    X = df[feature_columns].values
    
    # Prepare risk labels
    risk_le = LabelEncoder()
    risk_labels_encoded = risk_le.fit_transform(df['RiskLevel'])
    risk_labels = tf.keras.utils.to_categorical(risk_labels_encoded)
    
    # Prepare health recommendation labels
    health_le = LabelEncoder()
    health_labels_encoded = health_le.fit_transform(df['HealthRecommendation'])
    health_labels = tf.keras.utils.to_categorical(health_labels_encoded)
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_risk_train, y_risk_test, y_health_train, y_health_test = train_test_split(
        X_scaled, risk_labels, health_labels, test_size=0.2, random_state=42, stratify=risk_labels_encoded
    )
    
    # Create and train risk model
    print("\nTraining Risk Prediction Model...")
    risk_model = create_risk_model(input_shape=X_train.shape[1])
    
    risk_history = risk_model.fit(
        X_train, y_risk_train,
        validation_data=(X_test, y_risk_test),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
        ],
        verbose=1
    )
    
    # Create and train health model
    print("\nTraining Health Recommendation Model...")
    health_model = create_health_model(input_shape=X_train.shape[1])
    
    health_history = health_model.fit(
        X_train, y_health_train,
        validation_data=(X_test, y_health_test),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
        ],
        verbose=1
    )
    
    # Evaluate models
    print("\nEvaluating Models...")
    risk_loss, risk_acc, risk_prec, risk_recall = risk_model.evaluate(X_test, y_risk_test, verbose=0)
    health_loss, health_acc, health_prec, health_recall = health_model.evaluate(X_test, y_health_test, verbose=0)
    
    print(f"\nRisk Model Performance:")
    print(f"Accuracy: {risk_acc:.4f}, Precision: {risk_prec:.4f}, Recall: {risk_recall:.4f}")
    
    print(f"\nHealth Model Performance:")
    print(f"Accuracy: {health_acc:.4f}, Precision: {health_prec:.4f}, Recall: {health_recall:.4f}")
    
    # Save models and preprocessors
    print("\nSaving models and preprocessors...")
    os.makedirs('maternal_models', exist_ok=True)
    
    risk_model.save('maternal_models/risk_model.keras')
    health_model.save('maternal_models/health_model.keras')
    
    with open('maternal_models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('maternal_models/risk_label_encoder.pkl', 'wb') as f:
        pickle.dump(risk_le, f)
    with open('maternal_models/health_label_encoder.pkl', 'wb') as f:
        pickle.dump(health_le, f)
    with open('maternal_models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("All models and preprocessors saved successfully!")
    
    return risk_model, health_model, scaler, risk_le, health_le, feature_columns

# Prediction function
def predict_maternal_health(age, gestational_week, systolic_bp, diastolic_bp, 
                           blood_sugar, body_temp, heart_rate, bmi, 
                           previous_pregnancies, weight_gain):
    """
    Predict maternal health risk and recommendations
    
    Parameters:
    - age: Mother's age
    - gestational_week: Current week of pregnancy (12-40)
    - systolic_bp: Systolic blood pressure
    - diastolic_bp: Diastolic blood pressure  
    - blood_sugar: Blood sugar level (mg/dL)
    - body_temp: Body temperature (°F)
    - heart_rate: Heart rate (bpm)
    - bmi: Body Mass Index
    - previous_pregnancies: Number of previous pregnancies
    - weight_gain: Weight gain during current pregnancy (lbs)
    """
    try:
        # Load models and preprocessors
        risk_model = tf.keras.models.load_model('maternal_models/risk_model.keras')
        health_model = tf.keras.models.load_model('maternal_models/health_model.keras')
        
        with open('maternal_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('maternal_models/risk_label_encoder.pkl', 'rb') as f:
            risk_le = pickle.load(f)
        with open('maternal_models/health_label_encoder.pkl', 'rb') as f:
            health_le = pickle.load(f)
        
        # Prepare input
        input_data = np.array([[age, gestational_week, systolic_bp, diastolic_bp,
                               blood_sugar, body_temp, heart_rate, bmi,
                               previous_pregnancies, weight_gain]])
        
        input_scaled = scaler.transform(input_data)
        
        # Make predictions
        risk_pred = risk_model.predict(input_scaled, verbose=0)
        health_pred = health_model.predict(input_scaled, verbose=0)
        
        # Get results
        risk_idx = np.argmax(risk_pred[0])
        risk_level = risk_le.inverse_transform([risk_idx])[0]
        risk_confidence = risk_pred[0][risk_idx]
        
        health_idx = np.argmax(health_pred[0])
        health_recommendation = health_le.inverse_transform([health_idx])[0]
        health_confidence = health_pred[0][health_idx]
        
        # Generate clinical insights
        insights = []
        if systolic_bp > 140 or diastolic_bp > 90:
            insights.append("⚠️ High blood pressure detected - monitor closely")
        if blood_sugar > 140:
            insights.append("⚠️ Elevated blood sugar - check for gestational diabetes")
        if bmi > 30:
            insights.append("⚠️ High BMI - nutritional guidance recommended")
        if age > 35:
            insights.append("ℹ️ Advanced maternal age - additional monitoring may be needed")
        if gestational_week > 37 and risk_level == "High Risk":
            insights.append("⚠️ Near term with high risk - consider closer monitoring")
        
        return {
            'risk_level': risk_level,
            'risk_confidence': float(risk_confidence),
            'health_recommendation': health_recommendation,
            'health_confidence': float(health_confidence),
            'clinical_insights': insights,
            'all_risk_probabilities': {
                label: float(prob) for label, prob in zip(risk_le.classes_, risk_pred[0])
            },
            'all_health_probabilities': {
                label: float(prob) for label, prob in zip(health_le.classes_, health_pred[0])
            }
        }
        
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

# Demo function with realistic pregnancy scenarios
def demo_maternal_health_predictions():
    """
    Demo with realistic pregnant mother scenarios
    """
    print("\n" + "="*60)
    print("MATERNAL HEALTH RISK PREDICTION DEMO")
    print("="*60)
    
    test_cases = [
        {
            'name': 'Healthy Young Mother (Low Risk)',
            'params': [25, 28, 115, 75, 95, 98.6, 85, 22, 0, 25],
            'description': '25yo, 28 weeks pregnant, first pregnancy, normal vitals'
        },
        {
            'name': 'High-Risk Pregnancy',
            'params': [38, 34, 165, 95, 155, 99.2, 105, 32, 3, 45],
            'description': '38yo, 34 weeks, hypertension, diabetes, high BMI'
        },
        {
            'name': 'Moderate Risk Case',
            'params': [32, 26, 135, 88, 125, 98.8, 92, 27, 1, 35],
            'description': '32yo, 26 weeks, slightly elevated BP and glucose'
        },
        {
            'name': 'Underweight Concern',
            'params': [22, 24, 105, 68, 85, 98.4, 78, 17, 0, 15],
            'description': '22yo, 24 weeks, underweight, low weight gain'
        }
    ]
    
    for case in test_cases:
        print(f"\n{'='*20}")
        print(f"Case: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"{'='*20}")
        
        params = case['params']
        print(f"Age: {params[0]}, Gestational Week: {params[1]}")
        print(f"BP: {params[2]}/{params[3]}, Blood Sugar: {params[4]}")
        print(f"Temp: {params[5]}°F, Heart Rate: {params[6]} bpm")
        print(f"BMI: {params[7]}, Previous Pregnancies: {params[8]}")
        print(f"Weight Gain: {params[9]} lbs")
        
        result = predict_maternal_health(*params)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n📊 RESULTS:")
            print(f"Risk Level: {result['risk_level']} ({result['risk_confidence']:.1%} confidence)")
            print(f"Recommendation: {result['health_recommendation']} ({result['health_confidence']:.1%} confidence)")
            
            if result['clinical_insights']:
                print(f"\n🏥 Clinical Insights:")
                for insight in result['clinical_insights']:
                    print(f"  {insight}")
            else:
                print(f"\n✅ No immediate clinical concerns identified")

if __name__ == "__main__":
    print("Maternal Health Risk Prediction System for Pregnant Mothers")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists('maternal_models/risk_model.keras'):
        print("Training new models...")
        train_maternal_health_models()
    else:
        print("Using existing trained models...")
    
    # Run demo
    demo_maternal_health_predictions()
    
    print(f"\n{'='*60}")
    print("System ready for predictions!")
    print("Use predict_maternal_health() function with patient data.")