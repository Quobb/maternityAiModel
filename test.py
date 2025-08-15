from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Maternal Health Risk Prediction API",
    description="AI-powered maternal health risk assessment and recommendations for pregnant mothers",
    version="1.0.0"
)

# Global variables for models and preprocessors
risk_model = None
health_model = None
scaler = None
risk_le = None
health_le = None
feature_columns = None

def load_models():
    """Load all models and preprocessors at startup"""
    global risk_model, health_model, scaler, risk_le, health_le, feature_columns
    
    try:
        # Load models
        risk_model = tf.keras.models.load_model('maternal_models/risk_model.keras')
        health_model = tf.keras.models.load_model('maternal_models/health_model.keras')
        
        # Load preprocessors
        with open('maternal_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('maternal_models/risk_label_encoder.pkl', 'rb') as f:
            risk_le = pickle.load(f)
        with open('maternal_models/health_label_encoder.pkl', 'rb') as f:
            health_le = pickle.load(f)
        with open('maternal_models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
            
        logger.info("All models and preprocessors loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Load models at startup
@app.on_event("startup")
async def startup_event():
    success = load_models()
    if not success:
        logger.error("Failed to load models. Please ensure models are trained and saved properly.")

# Input schemas
class MaternalHealthInput(BaseModel):
    """Input schema for maternal health prediction"""
    age: int = Field(..., ge=16, le=50, description="Mother's age in years")
    gestational_week: float = Field(..., ge=12, le=42, description="Current week of pregnancy")
    systolic_bp: int = Field(..., ge=80, le=200, description="Systolic blood pressure (mmHg)")
    diastolic_bp: int = Field(..., ge=50, le=120, description="Diastolic blood pressure (mmHg)")
    blood_sugar: float = Field(..., ge=60, le=250, description="Blood sugar level (mg/dL)")
    body_temp: float = Field(..., ge=96, le=104, description="Body temperature (°F)")
    heart_rate: int = Field(..., ge=50, le=150, description="Heart rate (beats per minute)")
    bmi: float = Field(..., ge=15, le=50, description="Body Mass Index")
    previous_pregnancies: int = Field(..., ge=0, le=10, description="Number of previous pregnancies")
    weight_gain: float = Field(..., ge=-20, le=80, description="Weight gain during pregnancy (lbs)")

    class Config:
        schema_extra = {
            "example": {
                "age": 28,
                "gestational_week": 32.0,
                "systolic_bp": 125,
                "diastolic_bp": 80,
                "blood_sugar": 95.0,
                "body_temp": 98.6,
                "heart_rate": 85,
                "bmi": 24.5,
                "previous_pregnancies": 1,
                "weight_gain": 28.0
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    risk_level: str
    risk_confidence: float
    health_recommendation: str
    health_confidence: float
    clinical_insights: List[str]
    all_risk_probabilities: Dict[str, float]
    all_health_probabilities: Dict[str, float]
    timestamp: str

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool
    version: str

# Helper function to generate clinical insights
def generate_clinical_insights(input_data: MaternalHealthInput) -> List[str]:
    """Generate clinical insights based on input parameters"""
    insights = []
    
    # Age-related insights
    if input_data.age < 18:
        insights.append("⚠️ Teen pregnancy - requires specialized care and monitoring")
    elif input_data.age > 35:
        insights.append("ℹ️ Advanced maternal age - increased monitoring recommended")
    
    # Blood pressure insights
    if input_data.systolic_bp > 140 or input_data.diastolic_bp > 90:
        insights.append("⚠️ Hypertension detected - immediate medical evaluation needed")
    elif input_data.systolic_bp > 130 or input_data.diastolic_bp > 85:
        insights.append("⚠️ Pre-hypertension - monitor blood pressure closely")
    
    # Blood sugar insights
    if input_data.blood_sugar > 140:
        insights.append("⚠️ High blood glucose - gestational diabetes screening recommended")
    elif input_data.blood_sugar > 125:
        insights.append("⚠️ Elevated blood glucose - dietary consultation advised")
    
    # BMI insights
    if input_data.bmi < 18.5:
        insights.append("⚠️ Underweight - nutritional support and weight gain monitoring needed")
    elif input_data.bmi > 30:
        insights.append("⚠️ Obesity - increased risk of complications, dietary guidance recommended")
    elif input_data.bmi > 25:
        insights.append("ℹ️ Overweight - monitor weight gain and consider nutritional counseling")
    
    # Heart rate insights
    if input_data.heart_rate > 100:
        insights.append("⚠️ Elevated heart rate - assess for fever, anxiety, or cardiac issues")
    elif input_data.heart_rate < 60:
        insights.append("ℹ️ Low heart rate - monitor if symptomatic")
    
    # Temperature insights
    if input_data.body_temp > 100.4:
        insights.append("⚠️ Fever detected - evaluate for infection")
    elif input_data.body_temp < 97.0:
        insights.append("ℹ️ Low body temperature - monitor for hypothermia")
    
    # Gestational week insights
    if input_data.gestational_week < 20 and len([i for i in insights if "⚠️" in i]) > 2:
        insights.append("⚠️ Multiple risk factors in early pregnancy - close monitoring essential")
    elif input_data.gestational_week > 37:
        insights.append("ℹ️ Full term - monitor for signs of labor")
    
    # Weight gain insights
    expected_gain = 25 if input_data.bmi < 25 else 15 if input_data.bmi < 30 else 10
    if abs(input_data.weight_gain - expected_gain) > 15:
        insights.append("⚠️ Abnormal weight gain pattern - nutritional assessment recommended")
    
    return insights

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_maternal_health(input_data: MaternalHealthInput):
    """
    Predict maternal health risk and provide recommendations
    """
    try:
        if risk_model is None or health_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Prepare input data in the correct order
        input_array = np.array([[
            input_data.age,
            input_data.gestational_week,
            input_data.systolic_bp,
            input_data.diastolic_bp,
            input_data.blood_sugar,
            input_data.body_temp,
            input_data.heart_rate,
            input_data.bmi,
            input_data.previous_pregnancies,
            input_data.weight_gain
        ]])
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Make predictions
        risk_pred = risk_model.predict(input_scaled, verbose=0)
        health_pred = health_model.predict(input_scaled, verbose=0)
        
        # Get risk level
        risk_idx = np.argmax(risk_pred[0])
        risk_level = risk_le.inverse_transform([risk_idx])[0]
        risk_confidence = float(risk_pred[0][risk_idx])
        
        # Get health recommendation
        health_idx = np.argmax(health_pred[0])
        health_recommendation = health_le.inverse_transform([health_idx])[0]
        health_confidence = float(health_pred[0][health_idx])
        
        # Generate clinical insights
        clinical_insights = generate_clinical_insights(input_data)
        
        # Prepare probability dictionaries
        all_risk_probs = {
            label: float(prob) 
            for label, prob in zip(risk_le.classes_, risk_pred[0])
        }
        
        all_health_probs = {
            label: float(prob) 
            for label, prob in zip(health_le.classes_, health_pred[0])
        }
        
        return PredictionResponse(
            risk_level=risk_level,
            risk_confidence=risk_confidence,
            health_recommendation=health_recommendation,
            health_confidence=health_confidence,
            clinical_insights=clinical_insights,
            all_risk_probabilities=all_risk_probs,
            all_health_probabilities=all_health_probs,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Simplified health tips endpoint
@app.post("/health-tips")
async def get_health_tips(input_data: MaternalHealthInput):
    """
    Get health tips based on maternal health parameters
    """
    try:
        if health_model is None:
            raise HTTPException(status_code=503, detail="Health model not loaded")
        
        # Prepare and scale input
        input_array = np.array([[
            input_data.age, input_data.gestational_week, input_data.systolic_bp,
            input_data.diastolic_bp, input_data.blood_sugar, input_data.body_temp,
            input_data.heart_rate, input_data.bmi, input_data.previous_pregnancies,
            input_data.weight_gain
        ]])
        
        input_scaled = scaler.transform(input_array)
        prediction = health_model.predict(input_scaled, verbose=0)
        
        health_idx = np.argmax(prediction[0])
        health_category = health_le.inverse_transform([health_idx])[0]
        confidence = float(prediction[0][health_idx])
        
        # Generate specific tips based on category
        tips = {
            "Nutrition Focus": [
                "Focus on balanced meals with adequate protein",
                "Monitor blood sugar levels regularly",
                "Include folate-rich foods and prenatal vitamins",
                "Stay hydrated with plenty of water"
            ],
            "Exercise Focus": [
                "Engage in safe prenatal exercises like walking",
                "Practice prenatal yoga for flexibility",
                "Maintain appropriate weight gain",
                "Consult with healthcare provider about exercise routine"
            ],
            "Wellness Focus": [
                "Get adequate rest and sleep",
                "Practice stress management techniques",
                "Attend all prenatal appointments",
                "Monitor baby's movements regularly"
            ]
        }
        
        return {
            "category": health_category,
            "confidence": confidence,
            "tips": tips.get(health_category, ["Consult with your healthcare provider"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health tips error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate health tips: {str(e)}")

# Risk assessment endpoint
@app.post("/risk-assessment")
async def assess_risk(input_data: MaternalHealthInput):
    """
    Assess maternal health risk level
    """
    try:
        if risk_model is None:
            raise HTTPException(status_code=503, detail="Risk model not loaded")
        
        # Prepare and scale input
        input_array = np.array([[
            input_data.age, input_data.gestational_week, input_data.systolic_bp,
            input_data.diastolic_bp, input_data.blood_sugar, input_data.body_temp,
            input_data.heart_rate, input_data.bmi, input_data.previous_pregnancies,
            input_data.weight_gain
        ]])
        
        input_scaled = scaler.transform(input_array)
        prediction = risk_model.predict(input_scaled, verbose=0)
        
        risk_idx = np.argmax(prediction[0])
        risk_level = risk_le.inverse_transform([risk_idx])[0]
        confidence = float(prediction[0][risk_idx])
        
        # Risk-specific recommendations
        recommendations = {
            "Low Risk": "Continue regular prenatal care and healthy lifestyle",
            "Medium Risk": "Increased monitoring recommended - consult healthcare provider",
            "High Risk": "Immediate medical evaluation and close monitoring required"
        }
        
        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "recommendation": recommendations.get(risk_level, "Consult healthcare provider"),
            "all_probabilities": {
                label: float(prob) 
                for label, prob in zip(risk_le.classes_, prediction[0])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    API health check
    """
    return HealthCheckResponse(
        status="OK",
        timestamp=datetime.now().isoformat(),
        models_loaded=risk_model is not None and health_model is not None,
        version="1.0.0"
    )

# Root endpoint
@app.get("/")
async def root():
    """
    API root endpoint
    """
    return {
        "message": "Maternal Health Risk Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )