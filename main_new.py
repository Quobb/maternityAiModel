from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import uvicorn
import os
import sys

# Import the advanced maternal health AI system
try:
    # Assuming the AdvancedMaternalHealthAI class is in the same directory
    from paste import AdvancedMaternalHealthAI
except ImportError:
    try:
        # Alternative import path
        from maternal_health_ai import AdvancedMaternalHealthAI
    except ImportError:
        # Create a placeholder if not available
        class AdvancedMaternalHealthAI:
            def __init__(self):
                pass
            def load_models(self):
                return False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

app = FastAPI(
    title="Advanced Maternal Health AI API",
    description="Comprehensive AI-powered maternal health system with risk prediction, chat support, and personalized recommendations",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI system instance
ai_system: Optional[AdvancedMaternalHealthAI] = None

# Enhanced Input Schemas
class MaternalHealthInput(BaseModel):
    """Comprehensive maternal health input matching the AI system requirements"""
    # Demographics
    age: float = Field(..., ge=15, le=50, description="Mother's age in years")
    education_level: int = Field(default=3, ge=1, le=5, description="Education level (1-5)")
    income_level: int = Field(default=2, ge=1, le=4, description="Income level (1-4)")
    marital_status: int = Field(default=1, ge=0, le=1, description="Marital status (0: single, 1: married)")
    employment: int = Field(default=1, ge=0, le=1, description="Employment status (0: unemployed, 1: employed)")
    
    # Medical History
    previous_pregnancies: int = Field(default=0, ge=0, le=10, description="Number of previous pregnancies")
    previous_miscarriages: int = Field(default=0, ge=0, le=5, description="Number of previous miscarriages")
    diabetes_history: int = Field(default=0, ge=0, le=1, description="Diabetes history (0: no, 1: yes)")
    hypertension_history: int = Field(default=0, ge=0, le=1, description="Hypertension history (0: no, 1: yes)")
    heart_disease: int = Field(default=0, ge=0, le=1, description="Heart disease (0: no, 1: yes)")
    kidney_disease: int = Field(default=0, ge=0, le=1, description="Kidney disease (0: no, 1: yes)")
    autoimmune_disorders: int = Field(default=0, ge=0, le=1, description="Autoimmune disorders (0: no, 1: yes)")
    mental_health_history: int = Field(default=0, ge=0, le=1, description="Mental health history (0: no, 1: yes)")
    
    # Current Pregnancy Data
    gestational_age: float = Field(..., ge=4, le=42, description="Gestational age in weeks")
    weight_pre_pregnancy: float = Field(..., ge=40, le=120, description="Pre-pregnancy weight in kg")
    height: float = Field(..., ge=145, le=185, description="Height in cm")
    weight_gain: float = Field(default=0, ge=-5, le=35, description="Weight gain during pregnancy in kg")
    
    # Vital Signs & Lab Values
    systolic_bp: float = Field(..., ge=90, le=180, description="Systolic blood pressure")
    diastolic_bp: float = Field(..., ge=60, le=120, description="Diastolic blood pressure")
    heart_rate: float = Field(default=80, ge=50, le=120, description="Heart rate")
    hemoglobin: float = Field(default=11.5, ge=7, le=16, description="Hemoglobin level")
    glucose_fasting: float = Field(default=90, ge=60, le=180, description="Fasting glucose level")
    protein_urine: int = Field(default=0, ge=0, le=3, description="Protein in urine (0-3)")
    white_blood_cells: float = Field(default=8000, ge=3000, le=15000, description="White blood cell count")
    platelets: float = Field(default=250000, ge=100000, le=450000, description="Platelet count")
    
    # Lifestyle Factors
    smoking: int = Field(default=0, ge=0, le=1, description="Smoking status (0: no, 1: yes)")
    alcohol: int = Field(default=0, ge=0, le=1, description="Alcohol consumption (0: no, 1: yes)")
    drug_use: int = Field(default=0, ge=0, le=1, description="Drug use (0: no, 1: yes)")
    exercise_level: int = Field(default=2, ge=1, le=4, description="Exercise level (1-4)")
    stress_level: int = Field(default=2, ge=1, le=5, description="Stress level (1-5)")
    sleep_hours: float = Field(default=7, ge=4, le=12, description="Hours of sleep per night")
    
    # Nutritional Status
    vitamin_d: float = Field(default=30, ge=10, le=80, description="Vitamin D level")
    iron_levels: float = Field(default=15, ge=5, le=30, description="Iron levels")
    folic_acid_intake: int = Field(default=1, ge=0, le=1, description="Folic acid intake (0: no, 1: yes)")
    prenatal_vitamins: int = Field(default=1, ge=0, le=1, description="Prenatal vitamins (0: no, 1: yes)")
    
    # Social Determinants
    access_to_healthcare: int = Field(default=3, ge=1, le=4, description="Healthcare access (1-4)")
    social_support: int = Field(default=3, ge=1, le=4, description="Social support level (1-4)")
    transportation_access: int = Field(default=1, ge=0, le=1, description="Transportation access (0: no, 1: yes)")
    insurance_coverage: int = Field(default=1, ge=0, le=1, description="Insurance coverage (0: no, 1: yes)")
    
    # Environmental Factors
    air_quality_index: float = Field(default=50, ge=10, le=200, description="Air quality index")
    water_quality: int = Field(default=2, ge=1, le=3, description="Water quality (1-3)")
    housing_quality: int = Field(default=3, ge=1, le=4, description="Housing quality (1-4)")
    
    # Fetal Measurements
    fetal_weight_percentile: float = Field(default=50, ge=5, le=95, description="Fetal weight percentile")
    amniotic_fluid_level: int = Field(default=2, ge=1, le=3, description="Amniotic fluid level (1: low, 2: normal, 3: high)")
    placental_position: int = Field(default=2, ge=1, le=3, description="Placental position (1: previa, 2: normal, 3: abruption)")
    
    @property
    def bmi_pre_pregnancy(self) -> float:
        """Calculate BMI from weight and height"""
        return self.weight_pre_pregnancy / ((self.height / 100) ** 2)

    class Config:
        schema_extra = {
            "example": {
                "age": 28,
                "gestational_age": 32.0,
                "weight_pre_pregnancy": 65.0,
                "height": 165.0,
                "systolic_bp": 125,
                "diastolic_bp": 80,
                "glucose_fasting": 95.0,
                "hemoglobin": 11.5,
                "previous_pregnancies": 1,
                "diabetes_history": 0,
                "smoking": 0,
                "exercise_level": 2,
                "stress_level": 2
            }
        }

class SimpleMaternalInput(BaseModel):
    """Simplified input for basic predictions"""
    age: float = Field(..., ge=15, le=50)
    gestational_age: float = Field(..., ge=4, le=42)
    systolic_bp: float = Field(..., ge=90, le=180)
    diastolic_bp: float = Field(..., ge=60, le=120)
    weight_pre_pregnancy: float = Field(..., ge=40, le=120)
    height: float = Field(..., ge=145, le=185)
    
    def to_comprehensive(self) -> MaternalHealthInput:
        """Convert to comprehensive input with defaults"""
        return MaternalHealthInput(
            age=self.age,
            gestational_age=self.gestational_age,
            weight_pre_pregnancy=self.weight_pre_pregnancy,
            height=self.height,
            systolic_bp=self.systolic_bp,
            diastolic_bp=self.diastolic_bp
        )

class ChatInput(BaseModel):
    """Chat input schema"""
    message: str = Field(..., min_length=1, max_length=1000, description="User's message")
    user_id: Optional[str] = Field(default="anonymous", description="User identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Conversation context")
    health_data: Optional[MaternalHealthInput] = Field(default=None, description="Health data for context")

# Response Schemas with proper typing
class PredictionResponse(BaseModel):
    """Comprehensive prediction response"""
    predictions: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    monitoring_schedule: Dict[str, Any]
    educational_resources: List[str]
    timestamp: str
    success: bool

class ChatResponse(BaseModel):
    """Chat response schema"""
    response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    emergency: bool = False
    suggestions: List[str] = []
    timestamp: str
    user_id: str

class HealthReportResponse(BaseModel):
    """Health report response"""
    assessment_date: str
    patient_summary: Dict[str, Any]
    risk_predictions: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    monitoring_schedule: Dict[str, Any]
    educational_resources: List[str]
    emergency_contacts: List[str]

# Dependency for AI system
async def get_ai_system() -> AdvancedMaternalHealthAI:
    """Get the AI system instance"""
    if ai_system is None:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    return ai_system

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the AI system on startup"""
    global ai_system
    
    try:
        logger.info("Initializing Advanced Maternal Health AI System...")
        ai_system = AdvancedMaternalHealthAI()
        
        # Try to load existing models
        models_loaded = ai_system.load_models()
        
        if not models_loaded:
            logger.warning("No existing models found. You may need to train the models first.")
            logger.info("To train models, run the main() function in the AdvancedMaternalHealthAI module.")
        else:
            logger.info("✅ AI system initialized successfully with pre-trained models!")
            
    except Exception as e:
        logger.error(f"Failed to initialize AI system: {str(e)}")
        # Don't fail startup, but log the error
        ai_system = None

# Core Prediction Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def comprehensive_prediction(
    health_data: MaternalHealthInput,
    ai: AdvancedMaternalHealthAI = Depends(get_ai_system)
):
    """
    Get comprehensive maternal health predictions and recommendations
    """
    try:
        # Convert to dictionary
        health_dict = health_data.dict()
        
        # Add calculated BMI
        health_dict['bmi_pre_pregnancy'] = health_data.bmi_pre_pregnancy
        
        # Get predictions and convert NumPy types
        predictions = ai.predict_comprehensive_health_risk(health_dict)
        predictions = convert_numpy_types(predictions)
        
        # Get personalized recommendations
        recommendations = ai.generate_personalized_recommendations(health_dict)
        recommendations = convert_numpy_types(recommendations)
        
        # Get monitoring schedule
        monitoring_schedule = ai.generate_monitoring_schedule(health_dict)
        monitoring_schedule = convert_numpy_types(monitoring_schedule)
        
        # Get educational resources
        educational_resources = ai.get_educational_resources(health_dict)
        educational_resources = convert_numpy_types(educational_resources)
        
        return PredictionResponse(
            predictions=predictions,
            recommendations=recommendations,
            monitoring_schedule=monitoring_schedule,
            educational_resources=educational_resources,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-simple")
async def simple_prediction(
    health_data: SimpleMaternalInput,
    ai: AdvancedMaternalHealthAI = Depends(get_ai_system)
):
    """
    Simple prediction endpoint with basic health data
    """
    try:
        # Convert to comprehensive format
        comprehensive_data = health_data.to_comprehensive()
        
        # Use the comprehensive prediction
        return await comprehensive_prediction(comprehensive_data, ai)
        
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Chat Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    chat_input: ChatInput,
    ai: AdvancedMaternalHealthAI = Depends(get_ai_system)
):
    """
    Chat with the advanced maternal health AI
    """
    try:
        # Prepare user context
        user_context = chat_input.context or {}
        
        # Add health data context if provided
        if chat_input.health_data:
            health_dict = chat_input.health_data.dict()
            health_dict['bmi_pre_pregnancy'] = chat_input.health_data.bmi_pre_pregnancy
            
            # Add some health indicators to context
            if health_dict['gestational_age'] < 13:
                user_context['trimester_1'] = True
            elif health_dict['gestational_age'] < 27:
                user_context['trimester_2'] = True
            else:
                user_context['trimester_3'] = True
                
            if health_dict['diabetes_history'] == 1:
                user_context['gestational_diabetes'] = True
            if health_dict['hemoglobin'] < 11:
                user_context['anemia'] = True
            if health_dict['previous_pregnancies'] == 0:
                user_context['first_pregnancy'] = True
        
        # Get AI response
        response = ai.get_advanced_chat_response(chat_input.message, user_context)
        
        # Convert NumPy types
        response = convert_numpy_types(response)
        
        return ChatResponse(
            response=response['response'],
            intent=response.get('intent'),
            confidence=response.get('confidence'),
            emergency=response.get('emergency', False),
            suggestions=response.get('suggestions', []),
            timestamp=datetime.now().isoformat(),
            user_id=chat_input.user_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/integrated-consultation")
async def integrated_consultation(
    chat_input: ChatInput,
    ai: AdvancedMaternalHealthAI = Depends(get_ai_system)
):
    """
    Integrated consultation combining chat and health predictions
    """
    try:
        response_data = {
            "chat_response": "",
            "prediction_results": None,
            "health_recommendations": [],
            "clinical_insights": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Get chat response
        user_context = chat_input.context or {}
        if chat_input.health_data:
            health_dict = chat_input.health_data.dict()
            health_dict['bmi_pre_pregnancy'] = chat_input.health_data.bmi_pre_pregnancy
        else:
            health_dict = None
        
        chat_response = ai.get_advanced_chat_response(chat_input.message, user_context)
        response_data["chat_response"] = chat_response['response']
        
        # Add predictions if health data is available
        if health_dict:
            predictions = ai.predict_comprehensive_health_risk(health_dict)
            recommendations = ai.generate_personalized_recommendations(health_dict)
            
            # Convert NumPy types
            predictions = convert_numpy_types(predictions)
            recommendations = convert_numpy_types(recommendations)
            
            response_data["prediction_results"] = predictions
            response_data["health_recommendations"] = [rec['recommendation'] for rec in recommendations[:5]]
            
            # Generate clinical insights summary
            clinical_insights = []
            for target, pred_info in predictions.items():
                if 'prediction' in pred_info:
                    if pred_info['prediction'] in ['High', 'Critical', 1]:
                        clinical_insights.append(f"⚠️ {target.replace('_', ' ').title()}: Requires attention")
                    elif pred_info['prediction'] in ['Medium', 'Moderate']:
                        clinical_insights.append(f"ℹ️ {target.replace('_', ' ').title()}: Monitor closely")
            
            response_data["clinical_insights"] = clinical_insights[:3]
        
        # Convert all NumPy types in response_data
        response_data = convert_numpy_types(response_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Integrated consultation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Consultation failed: {str(e)}")

# Health Report Endpoint
@app.post("/health-report", response_model=HealthReportResponse)
async def generate_health_report(
    health_data: MaternalHealthInput,
    ai: AdvancedMaternalHealthAI = Depends(get_ai_system)
):
    """
    Generate comprehensive health report
    """
    try:
        # Convert to dictionary
        health_dict = health_data.dict()
        health_dict['bmi_pre_pregnancy'] = health_data.bmi_pre_pregnancy
        
        # Generate comprehensive report
        report = ai.generate_comprehensive_health_report(health_dict)
        
        # Convert NumPy types
        report = convert_numpy_types(report)
        
        return HealthReportResponse(**report)
        
    except Exception as e:
        logger.error(f"Health report error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# Risk Assessment Endpoints
@app.post("/risk-assessment")
async def risk_assessment(
    health_data: MaternalHealthInput,
    ai: AdvancedMaternalHealthAI = Depends(get_ai_system)
):
    """
    Detailed risk assessment with specific focus areas
    """
    try:
        health_dict = health_data.dict()
        health_dict['bmi_pre_pregnancy'] = health_data.bmi_pre_pregnancy
        
        predictions = ai.predict_comprehensive_health_risk(health_dict)
        recommendations = ai.generate_personalized_recommendations(health_dict)
        
        # Focus on risk-related predictions
        risk_summary = {
            "overall_risk": predictions.get('risk_level', {}).get('prediction', 'Unknown'),
            "specific_risks": {},
            "recommendations": [rec for rec in recommendations if rec['priority'] in ['High', 'Critical']],
            "monitoring_needs": ai.generate_monitoring_schedule(health_dict),
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract specific risk predictions
        risk_conditions = ['gestational_diabetes', 'preeclampsia', 'preterm_birth_risk', 
                          'postpartum_depression_risk', 'cesarean_risk']
        
        for condition in risk_conditions:
            if condition in predictions and 'prediction' in predictions[condition]:
                risk_summary["specific_risks"][condition] = {
                    "risk": predictions[condition]['prediction'],
                    "confidence": predictions[condition]['confidence'],
                    "algorithm": predictions[condition]['algorithm_used']
                }
        
        # Convert NumPy types
        risk_summary = convert_numpy_types(risk_summary)
        
        return risk_summary
        
    except Exception as e:
        logger.error(f"Risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# Health Tips and Recommendations
@app.post("/health-tips")
async def get_health_tips(
    health_data: MaternalHealthInput,
    focus_area: Optional[str] = None,
    ai: AdvancedMaternalHealthAI = Depends(get_ai_system)
):
    """
    Get personalized health tips and recommendations
    """
    try:
        health_dict = health_data.dict()
        health_dict['bmi_pre_pregnancy'] = health_data.bmi_pre_pregnancy
        
        recommendations = ai.generate_personalized_recommendations(health_dict)
        educational_resources = ai.get_educational_resources(health_dict)
        
        # Filter by focus area if specified
        if focus_area:
            recommendations = [rec for rec in recommendations 
                             if focus_area.lower() in rec['category'].lower()]
        
        response_data = {
            "gestational_age": health_data.gestational_age,
            "recommendations": recommendations,
            "educational_resources": educational_resources,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to get additional methods if available
        try:
            response_data["lifestyle_tips"] = ai.get_lifestyle_recommendations(health_dict)
            response_data["nutritional_guidance"] = ai.get_nutritional_guidance(health_dict)
        except AttributeError:
            # These methods might not exist in all versions
            pass
        
        # Convert NumPy types
        response_data = convert_numpy_types(response_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Health tips error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health tips generation failed: {str(e)}")

# Training Endpoint (for admin use)
@app.post("/train-models")
async def train_models(admin_key: str = "admin_secret_key"):
    """
    Train or retrain the AI models (admin endpoint)
    """
    if admin_key != "admin_secret_key":  # In production, use proper authentication
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        global ai_system
        
        logger.info("Starting model training...")
        
        if ai_system is None:
            ai_system = AdvancedMaternalHealthAI()
        
        # Generate dataset and train models
        dataset = ai_system.generate_comprehensive_maternal_dataset()
        ai_system.train_comprehensive_models(dataset)
        
        response_data = {
            "message": "Models trained successfully",
            "dataset_size": len(dataset),
            "features": list(dataset.columns),
            "models_trained": list(ai_system.models.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert NumPy types
        response_data = convert_numpy_types(response_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# System Status and Health Check
@app.get("/health")
async def health_check():
    """
    API health check
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_system_available": ai_system is not None,
        "models_loaded": bool(ai_system and hasattr(ai_system, 'models') and ai_system.models),
        "chat_available": bool(ai_system and hasattr(ai_system, 'chat_model') and ai_system.chat_model),
        "version": "3.0.0",
        "features": [
            "Comprehensive Risk Prediction",
            "Advanced Chat System",
            "Personalized Recommendations",
            "Health Report Generation",
            "Emergency Detection",
            "Clinical Insights"
        ]
    }

@app.get("/model-status")
async def model_status():
    """
    Get detailed model status
    """
    if ai_system is None:
        return {"error": "AI system not initialized"}
    
    try:
        model_info = {}
        if hasattr(ai_system, 'models') and ai_system.models:
            for model_name, model_data in ai_system.models.items():
                model_info[model_name] = {
                    "algorithm": model_data.get('algorithm', 'Unknown'),
                    "accuracy": model_data.get('accuracy', 'Unknown'),
                    "features": len(model_data.get('features', [])),
                    "available": True
                }
        
        response_data = {
            "models_available": len(model_info),
            "chat_model_available": hasattr(ai_system, 'chat_model') and ai_system.chat_model is not None,
            "intents_loaded": hasattr(ai_system, 'intents') and ai_system.intents is not None,
            "scalers_available": len(ai_system.scalers) if hasattr(ai_system, 'scalers') and ai_system.scalers else 0,
            "encoders_available": len(ai_system.encoders) if hasattr(ai_system, 'encoders') and ai_system.encoders else 0,
            "model_details": model_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert NumPy types
        response_data = convert_numpy_types(response_data)
        
        return response_data
        
    except Exception as e:
        return {"error": f"Status check failed: {str(e)}"}

# Root endpoint
@app.get("/")
async def root():
    """
    API root endpoint with comprehensive information
    """
    return {
        "title": "Advanced Maternal Health AI API",
        "version": "3.0.0",
        "description": "Comprehensive AI-powered maternal health system",
        "features": [
            "🤖 Advanced Conversational AI",
            "📊 Multi-Algorithm Risk Prediction",
            "💡 Personalized Health Recommendations",
            "📋 Comprehensive Health Reports",
            "🚨 Emergency Situation Detection",
            "🎯 Clinical Insights Generation",
            "📈 Progress Monitoring",
            "📚 Educational Resources"
        ],
        "endpoints": {
            "predictions": {
                "/predict": "Comprehensive health prediction",
                "/predict-simple": "Simple prediction with basic data",
                "/risk-assessment": "Detailed risk assessment",
                "/health-tips": "Personalized health tips"
            },
            "chat": {
                "/chat": "AI chat support",
                "/integrated-consultation": "Combined chat and predictions"
            },
            "reports": {
                "/health-report": "Comprehensive health report"
            },
            "system": {
                "/health": "API health check",
                "/model-status": "Model status information",
                "/docs": "API documentation"
            }
        },
        "ai_system_status": "initialized" if ai_system else "not available",
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_excludes=["maternal_models/*"]  # Don't reload when model files change
    )