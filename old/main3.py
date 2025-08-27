# Health Consultation API with ML Models
# Requirements: pip install fastapi uvicorn pandas scikit-learn numpy pydantic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personalized Health Consultation API",
    description="AI-powered health tips and consultation system",
    version="1.0.0"
)

# Pydantic models for request/response
class HealthData(BaseModel):
    age: float = Field(..., ge=15, le=50, description="Age in years")
    gestational_week: float = Field(..., ge=0, le=42, description="Gestational week (0 if not pregnant)")
    systolic_bp: float = Field(..., ge=70, le=200, description="Systolic blood pressure")
    diastolic_bp: float = Field(..., ge=40, le=120, description="Diastolic blood pressure")
    blood_sugar: float = Field(..., ge=50, le=400, description="Blood sugar level")
    body_temp: float = Field(..., ge=95, le=110, description="Body temperature in Fahrenheit")
    heart_rate: float = Field(..., ge=40, le=150, description="Heart rate in BPM")
    bmi: float = Field(..., ge=15, le=50, description="Body Mass Index")
    previous_pregnancies: int = Field(..., ge=0, le=10, description="Number of previous pregnancies")
    weight_gain: float = Field(..., ge=-20, le=100, description="Weight gain in pounds")

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    user_id: Optional[str] = None

class PersonalizedTipRequest(BaseModel):
    health_data: HealthData
    focus_area: Optional[str] = Field(None, description="Specific area to focus on (nutrition, exercise, mental_health)")

class ConsultationRequest(BaseModel):
    health_data: HealthData
    symptoms: List[str] = Field(default=[], description="List of symptoms")
    concerns: str = Field("", description="Specific health concerns")

# Response models
class HealthTipResponse(BaseModel):
    risk_level: str
    personalized_tips: List[str]
    health_score: float
    recommendations: List[str]
    focus_areas: List[str]

class ChatResponse(BaseModel):
    response: str
    confidence: float
    suggested_actions: List[str]

class ConsultationResponse(BaseModel):
    risk_assessment: str
    severity_level: str
    recommendations: List[str]
    next_steps: List[str]
    when_to_seek_help: List[str]

class PredictionResponse(BaseModel):
    predicted_risk_level: str
    risk_probability: Dict[str, float]
    health_recommendation: str
    confidence_score: float

# Global variables for models
risk_model = None
recommendation_model = None
scaler = None
label_encoder = None

# Sample data for model training (you'll replace this with your actual dataset)
def create_sample_data():
    """Create sample data for model training - replace with your actual dataset"""
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'Age': np.random.normal(28, 6, n_samples),
        'GestationalWeek': np.random.uniform(0, 40, n_samples),
        'SystolicBP': np.random.normal(120, 15, n_samples),
        'DiastolicBP': np.random.normal(80, 10, n_samples),
        'BloodSugar': np.random.normal(100, 20, n_samples),
        'BodyTemp': np.random.normal(98.6, 1, n_samples),
        'HeartRate': np.random.normal(75, 12, n_samples),
        'BMI': np.random.normal(24, 4, n_samples),
        'PreviousPregnancies': np.random.poisson(1, n_samples),
        'WeightGain': np.random.normal(25, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create risk levels based on health indicators
    risk_conditions = [
        (df['SystolicBP'] > 140) | (df['DiastolicBP'] > 90) | (df['BMI'] > 30) | (df['BloodSugar'] > 140),
        (df['SystolicBP'] > 130) | (df['DiastolicBP'] > 85) | (df['BMI'] > 27) | (df['BloodSugar'] > 120),
    ]
    risk_choices = ['High', 'Medium', 'Low']
    df['RiskLevel'] = np.select(risk_conditions, risk_choices[:2], default=risk_choices[2])
    
    # Generate health recommendations
    recommendations = {
        'High': [
            'Consult with healthcare provider immediately',
            'Monitor blood pressure daily',
            'Follow strict dietary guidelines',
            'Limit physical activity until cleared by doctor'
        ],
        'Medium': [
            'Schedule regular check-ups',
            'Moderate exercise 3-4 times per week',
            'Follow balanced diet with reduced sodium',
            'Monitor symptoms closely'
        ],
        'Low': [
            'Maintain healthy lifestyle',
            'Regular exercise routine',
            'Balanced nutrition',
            'Annual health screenings'
        ]
    }
    
    df['HealthRecommendation'] = df['RiskLevel'].map(
        lambda x: np.random.choice(recommendations[x])
    )
    
    return df

def train_models():
    """Train the ML models"""
    global risk_model, recommendation_model, scaler, label_encoder
    
    try:
        # Create or load your dataset
        df = create_sample_data()  # Replace with: df = pd.read_csv('your_dataset.csv')
        
        # Prepare features
        feature_columns = ['Age', 'GestationalWeek', 'SystolicBP', 'DiastolicBP', 
                          'BloodSugar', 'BodyTemp', 'HeartRate', 'BMI', 
                          'PreviousPregnancies', 'WeightGain']
        
        X = df[feature_columns]
        y_risk = df['RiskLevel']
        y_recommendation = df['HealthRecommendation']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_risk_encoded = label_encoder.fit_transform(y_risk)
        
        # Train risk prediction model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_risk_encoded, test_size=0.2, random_state=42
        )
        
        risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        risk_model.fit(X_train, y_train)
        
        # Train recommendation model (simplified)
        recommendation_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rec_encoder = LabelEncoder()
        y_rec_encoded = rec_encoder.fit_transform(y_recommendation)
        recommendation_model.fit(X_scaled, y_rec_encoded)
        
        logger.info("Models trained successfully")
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise

# Health tip generation functions
def generate_personalized_tips(health_data: HealthData, risk_level: str) -> List[str]:
    """Generate personalized health tips based on health data and risk level"""
    tips = []
    
    # Blood pressure tips
    if health_data.systolic_bp > 130 or health_data.diastolic_bp > 85:
        tips.extend([
            "Reduce sodium intake to less than 2300mg per day",
            "Practice deep breathing exercises for 10 minutes daily",
            "Consider the DASH diet for blood pressure management"
        ])
    
    # BMI tips
    if health_data.bmi > 25:
        tips.extend([
            "Aim for 150 minutes of moderate exercise per week",
            "Focus on portion control and mindful eating",
            "Include more fiber-rich foods in your diet"
        ])
    elif health_data.bmi < 18.5:
        tips.append("Consider consulting a nutritionist for healthy weight gain strategies")
    
    # Blood sugar tips
    if health_data.blood_sugar > 100:
        tips.extend([
            "Limit refined carbohydrates and sugary foods",
            "Include protein with each meal to stabilize blood sugar",
            "Consider regular blood sugar monitoring"
        ])
    
    # Heart rate tips
    if health_data.heart_rate > 100:
        tips.extend([
            "Practice stress reduction techniques",
            "Limit caffeine intake",
            "Ensure adequate sleep (7-9 hours per night)"
        ])
    
    # Pregnancy-specific tips
    if health_data.gestational_week > 0:
        tips.extend([
            "Take prenatal vitamins with folic acid",
            "Stay hydrated with 8-10 glasses of water daily",
            "Attend all scheduled prenatal appointments"
        ])
    
    # General wellness tips
    if risk_level == "Low":
        tips.extend([
            "Maintain your current healthy habits",
            "Schedule annual preventive health screenings",
            "Consider adding meditation to your routine"
        ])
    
    return tips[:6]  # Return top 6 tips

def generate_chatbot_response(message: str) -> ChatResponse:
    """Generate chatbot response based on message content"""
    message_lower = message.lower()
    
    # Simple keyword-based responses (in production, use more sophisticated NLP)
    responses = {
        'blood pressure': {
            'response': "Blood pressure management involves lifestyle changes like reducing sodium, exercising regularly, and managing stress. Normal BP is typically below 120/80 mmHg.",
            'actions': ["Check your blood pressure regularly", "Reduce sodium intake", "Exercise 30 minutes daily"]
        },
        'diabetes': {
            'response': "Managing blood sugar involves a balanced diet, regular exercise, and monitoring. Focus on complex carbohydrates and avoid sugary foods.",
            'actions': ["Monitor blood sugar levels", "Follow a balanced diet", "Consult with a healthcare provider"]
        },
        'pregnancy': {
            'response': "During pregnancy, focus on prenatal care, proper nutrition, and staying active with doctor approval. Take prenatal vitamins and attend regular check-ups.",
            'actions': ["Take prenatal vitamins", "Attend regular appointments", "Follow a healthy diet"]
        },
        'weight': {
            'response': "Healthy weight management involves balanced nutrition and regular physical activity. Aim for gradual, sustainable changes rather than quick fixes.",
            'actions': ["Create a balanced meal plan", "Include regular exercise", "Track your progress"]
        }
    }
    
    # Find matching keywords
    for keyword, response_data in responses.items():
        if keyword in message_lower:
            return ChatResponse(
                response=response_data['response'],
                confidence=0.8,
                suggested_actions=response_data['actions']
            )
    
    # Default response
    return ChatResponse(
        response="I understand you have health-related questions. Could you be more specific about your concerns? I can help with topics like blood pressure, diabetes, pregnancy care, or weight management.",
        confidence=0.6,
        suggested_actions=["Be more specific about your health concerns", "Consider consulting with a healthcare provider"]
    )

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        train_models()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

@app.get("/")
async def root():
    return {"message": "Personalized Health Consultation API", "version": "1.0.0"}

@app.post("/personalized-tips", response_model=HealthTipResponse)
async def get_personalized_tips(request: PersonalizedTipRequest):
    """Endpoint 1: Get personalized health tips based on health data"""
    try:
        # Prepare data for prediction
        health_data = request.health_data
        features = np.array([[
            health_data.age, health_data.gestational_week, health_data.systolic_bp,
            health_data.diastolic_bp, health_data.blood_sugar, health_data.body_temp,
            health_data.heart_rate, health_data.bmi, health_data.previous_pregnancies,
            health_data.weight_gain
        ]])
        
        # Scale features and predict
        features_scaled = scaler.transform(features)
        risk_pred = risk_model.predict(features_scaled)[0]
        risk_level = label_encoder.inverse_transform([risk_pred])[0]
        
        # Calculate health score (0-100)
        risk_proba = risk_model.predict_proba(features_scaled)[0]
        health_score = (1 - risk_proba[risk_pred]) * 100
        
        # Generate personalized tips
        tips = generate_personalized_tips(health_data, risk_level)
        
        # Focus areas based on health data
        focus_areas = []
        if health_data.systolic_bp > 130: focus_areas.append("Blood Pressure")
        if health_data.bmi > 25 or health_data.bmi < 18.5: focus_areas.append("Weight Management")
        if health_data.blood_sugar > 100: focus_areas.append("Blood Sugar")
        if health_data.gestational_week > 0: focus_areas.append("Prenatal Care")
        
        return HealthTipResponse(
            risk_level=risk_level,
            personalized_tips=tips,
            health_score=round(health_score, 1),
            recommendations=[f"Focus on {area.lower()} management" for area in focus_areas],
            focus_areas=focus_areas
        )
        
    except Exception as e:
        logger.error(f"Error in personalized tips: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatbot", response_model=ChatResponse)
async def chatbot_endpoint(message: ChatMessage):
    """Endpoint 2: Health chatbot for general questions"""
    try:
        response = generate_chatbot_response(message.message)
        return response
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consultation", response_model=ConsultationResponse)
async def health_consultation(request: ConsultationRequest):
    """Endpoint 3: Comprehensive health consultation"""
    try:
        health_data = request.health_data
        symptoms = request.symptoms
        concerns = request.concerns
        
        # Get risk assessment
        features = np.array([[
            health_data.age, health_data.gestational_week, health_data.systolic_bp,
            health_data.diastolic_bp, health_data.blood_sugar, health_data.body_temp,
            health_data.heart_rate, health_data.bmi, health_data.previous_pregnancies,
            health_data.weight_gain
        ]])
        
        features_scaled = scaler.transform(features)
        risk_pred = risk_model.predict(features_scaled)[0]
        risk_level = label_encoder.inverse_transform([risk_pred])[0]
        
        # Assess severity based on symptoms and vital signs
        severity_factors = 0
        if health_data.systolic_bp > 140 or health_data.diastolic_bp > 90: severity_factors += 1
        if health_data.body_temp > 100.4: severity_factors += 1
        if health_data.heart_rate > 100 or health_data.heart_rate < 60: severity_factors += 1
        if len(symptoms) > 3: severity_factors += 1
        
        severity_levels = ["Low", "Moderate", "High", "Critical"]
        severity_level = severity_levels[min(severity_factors, 3)]
        
        # Generate recommendations
        recommendations = []
        if severity_level == "Critical":
            recommendations.extend([
                "Seek immediate medical attention",
                "Go to emergency room if experiencing severe symptoms",
                "Do not delay medical care"
            ])
        elif severity_level == "High":
            recommendations.extend([
                "Schedule urgent appointment with healthcare provider",
                "Monitor symptoms closely",
                "Consider telehealth consultation if unavailable"
            ])
        else:
            recommendations.extend([
                "Schedule routine appointment with healthcare provider",
                "Continue monitoring symptoms",
                "Maintain healthy lifestyle habits"
            ])
        
        # Next steps
        next_steps = [
            "Document all symptoms and their duration",
            "Keep a health diary for the next few days",
            "Follow up as recommended by your healthcare provider"
        ]
        
        # When to seek help
        seek_help = [
            "If symptoms worsen or new symptoms appear",
            "If you experience chest pain or difficulty breathing",
            "If you have concerns about your condition"
        ]
        
        return ConsultationResponse(
            risk_assessment=f"{risk_level} risk based on current health indicators",
            severity_level=severity_level,
            recommendations=recommendations,
            next_steps=next_steps,
            when_to_seek_help=seek_help
        )
        
    except Exception as e:
        logger.error(f"Error in consultation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_health_risk(health_data: HealthData):
    """Endpoint 4: ML-based health risk prediction"""
    try:
        # Prepare features
        features = np.array([[
            health_data.age, health_data.gestational_week, health_data.systolic_bp,
            health_data.diastolic_bp, health_data.blood_sugar, health_data.body_temp,
            health_data.heart_rate, health_data.bmi, health_data.previous_pregnancies,
            health_data.weight_gain
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        
        # Risk prediction
        risk_pred = risk_model.predict(features_scaled)[0]
        risk_proba = risk_model.predict_proba(features_scaled)[0]
        predicted_risk = label_encoder.inverse_transform([risk_pred])[0]
        
        # Create probability dictionary
        risk_classes = label_encoder.classes_
        risk_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(risk_proba)
        }
        
        # Generate recommendation
        rec_pred = recommendation_model.predict(features_scaled)[0]
        # Simplified recommendation mapping
        recommendations_map = {
            0: "Maintain current healthy lifestyle with regular check-ups",
            1: "Focus on diet and exercise improvements with medical monitoring",
            2: "Seek immediate medical consultation and follow treatment plan"
        }
        health_recommendation = recommendations_map.get(rec_pred, "Consult with healthcare provider")
        
        # Confidence score
        confidence_score = float(np.max(risk_proba))
        
        return PredictionResponse(
            predicted_risk_level=predicted_risk,
            risk_probability=risk_probabilities,
            health_recommendation=health_recommendation,
            confidence_score=round(confidence_score, 3)
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Model info endpoint
@app.get("/model-info")
async def model_info():
    """Get information about the trained models"""
    try:
        if risk_model is None:
            return {"status": "Models not loaded"}
        
        return {
            "risk_model": {
                "type": "RandomForestClassifier",
                "n_estimators": risk_model.n_estimators,
                "feature_count": len(risk_model.feature_importances_)
            },
            "classes": label_encoder.classes_.tolist() if label_encoder else [],
            "features": [
                "Age", "GestationalWeek", "SystolicBP", "DiastolicBP",
                "BloodSugar", "BodyTemp", "HeartRate", "BMI",
                "PreviousPregnancies", "WeightGain"
            ]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)