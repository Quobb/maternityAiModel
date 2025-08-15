from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import uvicorn
import os
import sys

# Import the chatbot class (assuming it's in the same directory)
try:
    from maternal_chatbot import MaternalHealthChatBot, IntegratedMaternalAssistant
except ImportError:
    # If the chatbot module isn't available, create a simple placeholder
    class MaternalHealthChatBot:
        def __init__(self):
            self.model = None
            self.intents = None
            
        def load_chat_model(self):
            return False
            
        def get_response(self, user_input, user_id=None):
            """Get response with optional user_id parameter for compatibility"""
            return "Chat functionality is not available. Please ensure the chatbot model is trained."
            
        def train_chat_model(self):
            """Placeholder for training method"""
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Integrated Maternal Health API",
    description="AI-powered maternal health risk assessment, recommendations, and conversational support for pregnant mothers",
    version="2.0.0"
)

# Global variables for models and chatbot
risk_model = None
health_model = None
scaler = None
risk_le = None
health_le = None
feature_columns = None
chatbot = None

def load_models():
    """Load all models and preprocessors at startup"""
    global risk_model, health_model, scaler, risk_le, health_le, feature_columns, chatbot
    
    try:
        # Load prediction models
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
        
        # Initialize chatbot
        chatbot = MaternalHealthChatBot()
        if hasattr(chatbot, 'load_chat_model'):
            if not chatbot.load_chat_model():
                logger.warning("Chatbot model not found.")
                if hasattr(chatbot, 'chat_model'):
                    logger.warning("Training new model...")
                    chatbot.train_chat_model()
        else:
            logger.warning("Chatbot does not have load_chat_model method")
            
        logger.info("All models and chatbot loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Load models at startup
@app.on_event("startup")
async def startup_event():
    success = load_models()
    if not success:
        logger.error("Failed to load some models. Some functionality may be limited.")

# Enhanced input schemas
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

class ChatInput(BaseModel):
    """Input schema for chat messages"""
    message: str = Field(..., min_length=1, max_length=500, description="User's message")
    user_id: Optional[str] = Field(default="anonymous", description="Optional user identifier")
    health_data: Optional[MaternalHealthInput] = Field(default=None, description="Optional health data for contextualized responses")

class ChatResponse(BaseModel):
    """Response schema for chat messages"""
    response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: str
    user_id: str
    prediction_available: bool = False
    emergency_detected: bool = False

class IntegratedResponse(BaseModel):
    """Combined prediction and chat response"""
    chat_response: str
    prediction_results: Optional[Dict] = None
    health_recommendations: List[str] = []
    clinical_insights: List[str] = []
    timestamp: str
    
class ConversationContext(BaseModel):
    """Conversation context for personalized responses"""
    user_id: str
    gestational_week: Optional[float] = None
    risk_level: Optional[str] = None
    last_prediction_date: Optional[str] = None
    conversation_history: List[Dict] = []

# Helper functions
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
    
    return insights

def make_prediction(input_data: MaternalHealthInput) -> Dict:
    """Make risk and health predictions"""
    try:
        if risk_model is None or health_model is None:
            return {"error": "Prediction models not loaded"}
        
        # Prepare input
        input_array = np.array([[
            input_data.age, input_data.gestational_week, input_data.systolic_bp,
            input_data.diastolic_bp, input_data.blood_sugar, input_data.body_temp,
            input_data.heart_rate, input_data.bmi, input_data.previous_pregnancies,
            input_data.weight_gain
        ]])
        
        input_scaled = scaler.transform(input_array)
        
        # Make predictions
        risk_pred = risk_model.predict(input_scaled, verbose=0)
        health_pred = health_model.predict(input_scaled, verbose=0)
        
        # Get results
        risk_idx = np.argmax(risk_pred[0])
        risk_level = risk_le.inverse_transform([risk_idx])[0]
        risk_confidence = float(risk_pred[0][risk_idx])
        
        health_idx = np.argmax(health_pred[0])
        health_recommendation = health_le.inverse_transform([health_idx])[0]
        health_confidence = float(health_pred[0][health_idx])
        
        # Generate insights
        clinical_insights = generate_clinical_insights(input_data)
        
        return {
            'risk_level': risk_level,
            'risk_confidence': risk_confidence,
            'health_recommendation': health_recommendation,
            'health_confidence': health_confidence,
            'clinical_insights': clinical_insights,
            'all_risk_probabilities': {
                label: float(prob) for label, prob in zip(risk_le.classes_, risk_pred[0])
            },
            'all_health_probabilities': {
                label: float(prob) for label, prob in zip(health_le.classes_, health_pred[0])
            }
        }
        
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(chat_input: ChatInput):
    """
    Chat with the maternal health AI assistant
    """
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not loaded")
        
        # Check for emergency keywords
        emergency_keywords = [
            "severe bleeding", "heavy bleeding", "severe pain", "can't breathe",
            "chest pain", "severe headache", "vision problems", "emergency",
            "call 911", "hospital now"
        ]
        
        emergency_detected = any(keyword in chat_input.message.lower() for keyword in emergency_keywords)
        
        # FIXED: Only pass the message, not user_id
        response = chatbot.get_response(chat_input.message)
        
        # Determine if prediction is available
        prediction_available = chat_input.health_data is not None
        
        # If health data is provided and user is asking about health assessment
        if prediction_available and any(word in chat_input.message.lower() for word in 
                                      ['risk', 'assess', 'predict', 'health check', 'analyze']):
            prediction_prompt = "\n\n🔍 I notice you have health data available. Would you like me to provide a detailed risk assessment based on your current parameters?"
            response += prediction_prompt
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat(),
            user_id=chat_input.user_id,
            prediction_available=prediction_available,
            emergency_detected=emergency_detected
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/integrated-consultation")
async def integrated_consultation(chat_input: ChatInput):
    """
    Provide integrated consultation combining chat and predictions
    """
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not loaded")
        
        # FIXED: Only pass the message
        chat_response = chatbot.get_response(chat_input.message)
        
        # Initialize response
        response = {
            'chat_response': chat_response,
            'prediction_results': None,
            'health_recommendations': [],
            'clinical_insights': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # If health data is provided, add predictions
        if chat_input.health_data:
            prediction_result = make_prediction(chat_input.health_data)
            
            if 'error' not in prediction_result:
                response['prediction_results'] = prediction_result
                response['clinical_insights'] = prediction_result['clinical_insights']
                
                # Generate personalized health recommendations
                recommendations = []
                
                if prediction_result['risk_level'] == 'High Risk':
                    recommendations.append("🚨 Schedule an immediate consultation with your healthcare provider")
                    recommendations.append("📊 Monitor your vital signs closely")
                elif prediction_result['risk_level'] == 'Medium Risk':
                    recommendations.append("⚠️ Increase frequency of prenatal visits")
                    recommendations.append("📋 Keep a daily log of symptoms")
                else:
                    recommendations.append("✅ Continue with regular prenatal care")
                    recommendations.append("🌟 Maintain your healthy lifestyle")
                
                # Add specific recommendations based on health focus
                if prediction_result['health_recommendation'] == 'Nutrition Focus':
                    recommendations.extend([
                        "🥗 Focus on balanced nutrition with adequate protein",
                        "💊 Ensure you're taking prenatal vitamins",
                        "🩺 Consider consultation with a nutritionist"
                    ])
                elif prediction_result['health_recommendation'] == 'Exercise Focus':
                    recommendations.extend([
                        "🚶‍♀️ Incorporate safe prenatal exercises like walking",
                        "🧘‍♀️ Try prenatal yoga for flexibility and relaxation",
                        "⚖️ Monitor weight gain appropriately"
                    ])
                else:  # Wellness Focus
                    recommendations.extend([
                        "😴 Prioritize adequate rest and sleep",
                        "🧘‍♀️ Practice stress management techniques",
                        "👥 Consider joining a prenatal support group"
                    ])
                
                response['health_recommendations'] = recommendations
        
        return IntegratedResponse(**response)
        
    except Exception as e:
        logger.error(f"Integrated consultation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Consultation failed: {str(e)}")

@app.post("/smart-chat")
async def smart_chat(chat_input: ChatInput):
    """
    Smart chat that automatically provides predictions when relevant
    """
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not loaded")
        
        # FIXED: Only pass the message
        base_response = chatbot.get_response(chat_input.message)
        
        # Check if user is asking about specific health concerns that warrant prediction
        prediction_triggers = [
            'how am i doing', 'am i healthy', 'should i worry', 'my risk',
            'everything okay', 'normal', 'check my health', 'assess me',
            'worried about', 'concerned about', 'is this normal'
        ]
        
        should_predict = any(trigger in chat_input.message.lower() for trigger in prediction_triggers)
        
        response_data = {
            'chat_response': base_response,
            'timestamp': datetime.now().isoformat(),
            'prediction_triggered': False
        }
        
        # If health data available and prediction is relevant, provide automatic assessment
        if chat_input.health_data and should_predict:
            prediction_result = make_prediction(chat_input.health_data)
            
            if 'error' not in prediction_result:
                response_data['prediction_triggered'] = True
                response_data['automatic_assessment'] = {
                    'risk_level': prediction_result['risk_level'],
                    'confidence': prediction_result['risk_confidence'],
                    'key_insights': prediction_result['clinical_insights'][:3],  # Top 3 insights
                    'recommendation': prediction_result['health_recommendation']
                }
                
                # Add prediction summary to chat response
                risk_emoji = "🟢" if prediction_result['risk_level'] == "Low Risk" else "🟡" if prediction_result['risk_level'] == "Medium Risk" else "🔴"
                
                prediction_summary = f"\n\n📊 **Quick Health Assessment:**\n{risk_emoji} Risk Level: {prediction_result['risk_level']} ({prediction_result['risk_confidence']:.1%} confidence)\n💡 Focus Area: {prediction_result['health_recommendation']}"
                
                if prediction_result['clinical_insights']:
                    prediction_summary += f"\n⚠️ Key Point: {prediction_result['clinical_insights'][0]}"
                
                response_data['chat_response'] += prediction_summary
        
        return response_data
        
    except Exception as e:
        logger.error(f"Smart chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Smart chat failed: {str(e)}")

# Enhanced prediction endpoints with chat integration
@app.post("/predict-with-chat")
async def predict_with_chat_support(input_data: MaternalHealthInput, 
                                  follow_up_question: Optional[str] = None):
    """
    Make predictions and provide chat-based follow-up support
    """
    try:
        # Make prediction
        prediction_result = make_prediction(input_data)
        
        if 'error' in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result['error'])
        
        response = {
            'prediction_results': prediction_result,
            'chat_support': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # If user has a follow-up question, provide chat support
        if follow_up_question and chatbot:
            # Create context-aware response
            context_info = f"Based on your assessment showing {prediction_result['risk_level']} risk level and {prediction_result['health_recommendation']} focus, here's my response to your question: "
            
            # FIXED: Only pass the question (this was already correct)
            chat_response = chatbot.get_response(follow_up_question)
            response['chat_support'] = {
                'question': follow_up_question,
                'response': context_info + chat_response,
                'context_provided': True
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction with chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversation management endpoints
@app.post("/start-conversation")
async def start_conversation(user_id: str, initial_health_data: Optional[MaternalHealthInput] = None):
    """
    Start a new conversation session with optional initial health assessment
    """
    try:
        session_data = {
            'user_id': user_id,
            'session_start': datetime.now().isoformat(),
            'initial_assessment': None,
            'welcome_message': None
        }
        
        # If initial health data provided, do assessment
        if initial_health_data:
            prediction_result = make_prediction(initial_health_data)
            if 'error' not in prediction_result:
                session_data['initial_assessment'] = {
                    'risk_level': prediction_result['risk_level'],
                    'gestational_week': initial_health_data.gestational_week,
                    'key_recommendations': prediction_result['clinical_insights'][:2]
                }
        
        # Generate personalized welcome message
        if chatbot:
            if session_data['initial_assessment']:
                welcome_context = f"Hello! I see you're {initial_health_data.gestational_week} weeks pregnant with {prediction_result['risk_level']} risk level."
                # FIXED: Only pass the message
                welcome_message = chatbot.get_response("Hello, I'm pregnant and would like support")
                session_data['welcome_message'] = welcome_context + " " + welcome_message
            else:
                # FIXED: Only pass the message
                session_data['welcome_message'] = chatbot.get_response("Hello, I'm pregnant")
        else:
            session_data['welcome_message'] = "Welcome to your maternal health assistant! I'm here to support you throughout your pregnancy journey."
        
        return session_data
        
    except Exception as e:
        logger.error(f"Session start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoints
@app.get("/health")
async def health_check():
    """
    API health check including chatbot status
    """
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "risk_model": risk_model is not None,
            "health_model": health_model is not None,
            "chatbot": chatbot is not None,
        },
        "version": "2.0.0",
        "features": [
            "Risk Prediction",
            "Health Recommendations", 
            "Conversational AI",
            "Integrated Consultation",
            "Smart Chat",
            "Emergency Detection"
        ]
    }

@app.get("/chat-capabilities")
async def get_chat_capabilities():
    """
    Get information about chatbot capabilities
    """
    if chatbot is None:
        return {"error": "Chatbot not available"}
    
    return {
        "available": True,
        "capabilities": [
            "Pregnancy symptom support",
            "Nutrition guidance",
            "Exercise recommendations",
            "Labor preparation",
            "Mental health support",
            "Emergency situation recognition",
            "Personalized tips based on gestational week"
        ],
        "supported_intents": [
            "greeting", "pregnancy_symptoms", "nutrition", "exercise",
            "baby_development", "prenatal_care", "concerns_warnings",
            "mental_health", "labor_delivery", "general_support"
        ],
        "integration_features": [
            "Context-aware responses",
            "Prediction-based recommendations",
            "Emergency detection",
            "Personalized guidance"
        ]
    }

# Root endpoint
@app.get("/")
async def root():
    """
    API root endpoint
    """
    return {
        "message": "Integrated Maternal Health AI Assistant",
        "version": "2.0.0",
        "features": [
            "🤖 Conversational AI Support",
            "📊 Risk Prediction & Assessment", 
            "💡 Personalized Health Recommendations",
            "🚨 Emergency Situation Detection",
            "🤝 Integrated Consultation Experience"
        ],
        "endpoints": {
            "chat": "/chat",
            "integrated_consultation": "/integrated-consultation",
            "smart_chat": "/smart-chat",
            "predict_with_chat": "/predict-with-chat",
            "documentation": "/docs"
        }
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "integrated_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )