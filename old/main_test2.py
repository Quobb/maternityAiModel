from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import uvicorn
import asyncio
from enum import Enum
import uuid
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Maternal Health AI Assistant",
    description="AI-powered maternal health risk assessment with chat consultation and verification reasoning",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and processors
risk_model = None
health_model = None
scaler = None
risk_le = None
health_le = None
feature_columns = None

# Chat and consultation system
class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    VERIFICATION = "verification"

class ConsultationType(str, Enum):
    GENERAL = "general"
    EMERGENCY = "emergency"
    FOLLOWUP = "followup"
    SYMPTOM_CHECK = "symptom_check"
    MEDICATION = "medication"
    LIFESTYLE = "lifestyle"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConsultationSession:
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    consultation_type: ConsultationType
    patient_data: Optional[Dict] = None
    chat_history: List[Dict] = None
    risk_assessments: List[Dict] = None
    verified_concerns: List[Dict] = None
    recommendations: List[Dict] = None
    is_active: bool = True

# In-memory storage for demo (use database in production)
active_sessions: Dict[str, ConsultationSession] = {}
verification_rules = {}

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

# Initialize verification rules
def initialize_verification_rules():
    """Initialize medical verification and reasoning rules"""
    global verification_rules
    
    verification_rules = {
        "blood_pressure": {
            "critical_high": {"systolic": 160, "diastolic": 100},
            "high": {"systolic": 140, "diastolic": 90},
            "elevated": {"systolic": 130, "diastolic": 85},
            "verification_questions": [
                "Have you been monitoring your blood pressure regularly?",
                "Are you experiencing headaches or vision changes?",
                "Do you have any swelling in your hands or face?"
            ]
        },
        "blood_sugar": {
            "critical_high": 180,
            "high": 140,
            "elevated": 125,
            "verification_questions": [
                "Have you been following your dietary recommendations?",
                "Are you experiencing increased thirst or frequent urination?",
                "Have you had a recent glucose tolerance test?"
            ]
        },
        "maternal_age": {
            "high_risk_young": 18,
            "high_risk_old": 35,
            "verification_questions": [
                "Are you receiving appropriate prenatal care?",
                "Have you discussed age-related risks with your healthcare provider?",
                "Are you taking prenatal vitamins as recommended?"
            ]
        },
        "emergency_symptoms": [
            "severe headache",
            "vision changes",
            "severe abdominal pain",
            "heavy bleeding",
            "decreased fetal movement",
            "difficulty breathing",
            "chest pain",
            "severe vomiting"
        ]
    }

# Pydantic models
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

class ChatMessage(BaseModel):
    """Chat message schema"""
    session_id: str
    message: str
    message_type: MessageType = MessageType.USER
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict] = None

class ConsultationRequest(BaseModel):
    """Consultation request schema"""
    user_id: str
    consultation_type: ConsultationType
    initial_message: str
    patient_data: Optional[MaternalHealthInput] = None
    symptoms: Optional[List[str]] = None
    urgency_level: Optional[SeverityLevel] = None

class VerificationQuery(BaseModel):
    """Verification query schema"""
    session_id: str
    concern_type: str
    user_responses: Dict[str, str]
    additional_symptoms: Optional[List[str]] = None

class EnhancedPredictionResponse(BaseModel):
    """Enhanced prediction response with verification"""
    risk_level: str
    risk_confidence: float
    health_recommendation: str
    health_confidence: float
    clinical_insights: List[str]
    verification_needed: bool
    verification_questions: List[str]
    severity_assessment: SeverityLevel
    immediate_actions: List[str]
    follow_up_recommendations: List[str]
    all_risk_probabilities: Dict[str, float]
    all_health_probabilities: Dict[str, float]
    reasoning_chain: List[str]
    timestamp: str

# AI Reasoning and Verification Engine
class MaternalHealthReasoner:
    """Advanced reasoning engine for maternal health assessment"""
    
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize medical knowledge base"""
        return {
            "pregnancy_stages": {
                "first_trimester": {"start": 12, "end": 13, "key_concerns": ["nausea", "bleeding", "cramping"]},
                "second_trimester": {"start": 14, "end": 27, "key_concerns": ["glucose_screening", "anatomy_scan"]},
                "third_trimester": {"start": 28, "end": 42, "key_concerns": ["preeclampsia", "preterm_labor", "fetal_monitoring"]}
            },
            "risk_factors": {
                "hypertension": {"threshold": {"systolic": 140, "diastolic": 90}, "severity": "high"},
                "diabetes": {"threshold": 140, "severity": "high"},
                "advanced_age": {"threshold": 35, "severity": "medium"},
                "young_age": {"threshold": 18, "severity": "medium"},
                "obesity": {"threshold": 30, "severity": "medium"}
            },
            "emergency_indicators": [
                "systolic_bp > 160",
                "diastolic_bp > 100",
                "blood_sugar > 200",
                "body_temp > 101.5",
                "severe_headache",
                "vision_changes",
                "severe_abdominal_pain"
            ]
        }
    
    def analyze_with_reasoning(self, patient_data: MaternalHealthInput) -> Dict:
        """Perform comprehensive analysis with reasoning chain"""
        reasoning_chain = []
        concerns = []
        severity = SeverityLevel.LOW
        verification_needed = False
        verification_questions = []
        
        # Analyze pregnancy stage
        stage = self._get_pregnancy_stage(patient_data.gestational_week)
        reasoning_chain.append(f"Patient is in {stage} at {patient_data.gestational_week} weeks gestation")
        
        # Blood pressure analysis
        bp_analysis = self._analyze_blood_pressure(patient_data.systolic_bp, patient_data.diastolic_bp)
        reasoning_chain.extend(bp_analysis["reasoning"])
        if bp_analysis["severity"] != SeverityLevel.LOW:
            concerns.append(bp_analysis["concern"])
            severity = max(severity, bp_analysis["severity"], key=lambda x: ["low", "medium", "high", "critical"].index(x.value))
            if bp_analysis["verify"]:
                verification_needed = True
                verification_questions.extend(verification_rules["blood_pressure"]["verification_questions"])
        
        # Blood sugar analysis
        glucose_analysis = self._analyze_blood_glucose(patient_data.blood_sugar, patient_data.gestational_week)
        reasoning_chain.extend(glucose_analysis["reasoning"])
        if glucose_analysis["severity"] != SeverityLevel.LOW:
            concerns.append(glucose_analysis["concern"])
            severity = max(severity, glucose_analysis["severity"], key=lambda x: ["low", "medium", "high", "critical"].index(x.value))
            if glucose_analysis["verify"]:
                verification_needed = True
                verification_questions.extend(verification_rules["blood_sugar"]["verification_questions"])
        
        # Age analysis
        age_analysis = self._analyze_maternal_age(patient_data.age)
        reasoning_chain.extend(age_analysis["reasoning"])
        if age_analysis["severity"] != SeverityLevel.LOW:
            concerns.append(age_analysis["concern"])
            if age_analysis["verify"]:
                verification_needed = True
                verification_questions.extend(verification_rules["maternal_age"]["verification_questions"])
        
        # BMI analysis
        bmi_analysis = self._analyze_bmi(patient_data.bmi, patient_data.weight_gain)
        reasoning_chain.extend(bmi_analysis["reasoning"])
        if bmi_analysis["severity"] != SeverityLevel.LOW:
            concerns.append(bmi_analysis["concern"])
        
        # Generate immediate actions
        immediate_actions = self._generate_immediate_actions(concerns, severity)
        
        # Generate follow-up recommendations
        follow_up = self._generate_follow_up_recommendations(concerns, patient_data)
        
        return {
            "reasoning_chain": reasoning_chain,
            "concerns": concerns,
            "severity": severity,
            "verification_needed": verification_needed,
            "verification_questions": list(set(verification_questions)),
            "immediate_actions": immediate_actions,
            "follow_up_recommendations": follow_up
        }
    
    def _get_pregnancy_stage(self, gestational_week: float) -> str:
        """Determine pregnancy stage"""
        if gestational_week < 14:
            return "first_trimester"
        elif gestational_week < 28:
            return "second_trimester"
        else:
            return "third_trimester"
    
    def _analyze_blood_pressure(self, systolic: int, diastolic: int) -> Dict:
        """Analyze blood pressure with reasoning"""
        reasoning = []
        concern = None
        severity = SeverityLevel.LOW
        verify = False
        
        if systolic >= 160 or diastolic >= 100:
            reasoning.append(f"CRITICAL: Blood pressure {systolic}/{diastolic} indicates severe hypertension")
            concern = "Severe hypertension requiring immediate medical attention"
            severity = SeverityLevel.CRITICAL
            verify = True
        elif systolic >= 140 or diastolic >= 90:
            reasoning.append(f"HIGH: Blood pressure {systolic}/{diastolic} indicates hypertension")
            concern = "Hypertension - risk for preeclampsia"
            severity = SeverityLevel.HIGH
            verify = True
        elif systolic >= 130 or diastolic >= 85:
            reasoning.append(f"ELEVATED: Blood pressure {systolic}/{diastolic} is elevated")
            concern = "Elevated blood pressure requiring monitoring"
            severity = SeverityLevel.MEDIUM
            verify = True
        else:
            reasoning.append(f"Blood pressure {systolic}/{diastolic} is within normal range")
        
        return {
            "reasoning": reasoning,
            "concern": concern,
            "severity": severity,
            "verify": verify
        }
    
    def _analyze_blood_glucose(self, glucose: float, gestational_week: float) -> Dict:
        """Analyze blood glucose with reasoning"""
        reasoning = []
        concern = None
        severity = SeverityLevel.LOW
        verify = False
        
        if glucose >= 200:
            reasoning.append(f"CRITICAL: Blood glucose {glucose} mg/dL is critically high")
            concern = "Severe hyperglycemia requiring immediate intervention"
            severity = SeverityLevel.CRITICAL
            verify = True
        elif glucose >= 140:
            reasoning.append(f"HIGH: Blood glucose {glucose} mg/dL suggests gestational diabetes")
            concern = "Probable gestational diabetes"
            severity = SeverityLevel.HIGH
            verify = True
        elif glucose >= 125:
            reasoning.append(f"ELEVATED: Blood glucose {glucose} mg/dL is elevated")
            concern = "Elevated glucose requiring glucose tolerance test"
            severity = SeverityLevel.MEDIUM
            verify = True
        else:
            reasoning.append(f"Blood glucose {glucose} mg/dL is within normal range")
        
        return {
            "reasoning": reasoning,
            "concern": concern,
            "severity": severity,
            "verify": verify
        }
    
    def _analyze_maternal_age(self, age: int) -> Dict:
        """Analyze maternal age with reasoning"""
        reasoning = []
        concern = None
        severity = SeverityLevel.LOW
        verify = False
        
        if age < 18:
            reasoning.append(f"Young maternal age ({age}) increases risk of complications")
            concern = "Teen pregnancy requiring specialized care"
            severity = SeverityLevel.MEDIUM
            verify = True
        elif age >= 35:
            reasoning.append(f"Advanced maternal age ({age}) increases genetic and pregnancy risks")
            concern = "Advanced maternal age requiring additional screening"
            severity = SeverityLevel.MEDIUM
            verify = True
        else:
            reasoning.append(f"Maternal age ({age}) is within optimal range")
        
        return {
            "reasoning": reasoning,
            "concern": concern,
            "severity": severity,
            "verify": verify
        }
    
    def _analyze_bmi(self, bmi: float, weight_gain: float) -> Dict:
        """Analyze BMI and weight gain with reasoning"""
        reasoning = []
        concern = None
        severity = SeverityLevel.LOW
        
        if bmi < 18.5:
            reasoning.append(f"BMI {bmi} indicates underweight status")
            concern = "Underweight - risk of inadequate fetal growth"
            severity = SeverityLevel.MEDIUM
        elif bmi >= 30:
            reasoning.append(f"BMI {bmi} indicates obesity")
            concern = "Obesity - increased risk of complications"
            severity = SeverityLevel.MEDIUM
        elif bmi >= 25:
            reasoning.append(f"BMI {bmi} indicates overweight status")
            concern = "Overweight - monitor weight gain closely"
            severity = SeverityLevel.LOW
        
        # Weight gain analysis
        expected_gain = 25 if bmi < 25 else 15 if bmi < 30 else 11
        if abs(weight_gain - expected_gain) > 15:
            reasoning.append(f"Weight gain {weight_gain} lbs deviates significantly from expected {expected_gain} lbs")
            if concern:
                concern += " and abnormal weight gain pattern"
            else:
                concern = "Abnormal weight gain pattern"
                severity = SeverityLevel.MEDIUM
        
        return {
            "reasoning": reasoning,
            "concern": concern,
            "severity": severity
        }
    
    def _generate_immediate_actions(self, concerns: List[str], severity: SeverityLevel) -> List[str]:
        """Generate immediate actions based on concerns and severity"""
        actions = []
        
        if severity == SeverityLevel.CRITICAL:
            actions.append("🚨 URGENT: Contact healthcare provider immediately or go to emergency room")
            actions.append("📞 Call emergency services if experiencing severe symptoms")
        elif severity == SeverityLevel.HIGH:
            actions.append("⚠️ Contact healthcare provider within 24 hours")
            actions.append("📊 Monitor symptoms and vital signs closely")
        elif severity == SeverityLevel.MEDIUM:
            actions.append("📞 Schedule appointment with healthcare provider")
            actions.append("📝 Keep detailed symptom diary")
        else:
            actions.append("✅ Continue regular prenatal care schedule")
            actions.append("📖 Follow general pregnancy health guidelines")
        
        return actions
    
    def _generate_follow_up_recommendations(self, concerns: List[str], patient_data: MaternalHealthInput) -> List[str]:
        """Generate follow-up recommendations"""
        recommendations = []
        
        # General recommendations
        recommendations.append("Regular prenatal check-ups as scheduled")
        recommendations.append("Monitor fetal movements daily")
        recommendations.append("Maintain healthy diet and appropriate exercise")
        
        # Specific recommendations based on concerns
        for concern in concerns:
            if "hypertension" in concern.lower() or "blood pressure" in concern.lower():
                recommendations.append("Daily blood pressure monitoring at home")
                recommendations.append("Reduce sodium intake and manage stress")
            
            if "diabetes" in concern.lower() or "glucose" in concern.lower():
                recommendations.append("Blood glucose monitoring as directed")
                recommendations.append("Follow diabetic diet guidelines")
            
            if "weight" in concern.lower():
                recommendations.append("Nutritionist consultation for weight management")
            
            if "age" in concern.lower():
                recommendations.append("Additional genetic screening tests")
        
        return list(set(recommendations))

# Initialize reasoning engine
reasoner = MaternalHealthReasoner()

# Chat and consultation functions
class ChatAssistant:
    """Intelligent chat assistant for maternal health consultation"""
    
    def __init__(self):
        self.context_memory = {}
    
    def process_message(self, message: str, session: ConsultationSession) -> Dict:
        """Process incoming chat message and generate response"""
        
        # Check for emergency keywords
        emergency_detected = self._detect_emergency_keywords(message)
        
        if emergency_detected:
            return self._handle_emergency_response(message, session)
        
        # Classify message intent
        intent = self._classify_intent(message)
        
        # Generate contextual response
        response = self._generate_contextual_response(message, intent, session)
        
        return response
    
    def _detect_emergency_keywords(self, message: str) -> bool:
        """Detect emergency symptoms in message"""
        emergency_keywords = verification_rules["emergency_symptoms"]
        message_lower = message.lower()
        
        for keyword in emergency_keywords:
            if keyword in message_lower:
                return True
        return False
    
    def _handle_emergency_response(self, message: str, session: ConsultationSession) -> Dict:
        """Handle emergency situation response"""
        return {
            "response": "🚨 **EMERGENCY ALERT** 🚨\n\nI've detected mention of potentially serious symptoms. "
                       "Please contact your healthcare provider immediately or call emergency services "
                       "if you are experiencing:\n\n"
                       "• Severe headache with vision changes\n"
                       "• Severe abdominal pain\n"
                       "• Heavy bleeding\n"
                       "• Difficulty breathing\n"
                       "• Decreased fetal movement\n\n"
                       "**Do not wait - seek immediate medical attention.**\n\n"
                       "If this is not an emergency, please describe your symptoms in more detail.",
            "message_type": MessageType.SYSTEM,
            "requires_verification": True,
            "severity": SeverityLevel.CRITICAL
        }
    
    def _classify_intent(self, message: str) -> str:
        """Classify the intent of user message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["pain", "hurt", "ache"]):
            return "symptom_report"
        elif any(word in message_lower for word in ["medication", "medicine", "drug", "pill"]):
            return "medication_query"
        elif any(word in message_lower for word in ["diet", "food", "eat", "nutrition"]):
            return "nutrition_query"
        elif any(word in message_lower for word in ["exercise", "activity", "workout"]):
            return "activity_query"
        elif any(word in message_lower for word in ["normal", "safe", "worry", "concerned"]):
            return "reassurance_seeking"
        else:
            return "general_query"
    
    def _generate_contextual_response(self, message: str, intent: str, session: ConsultationSession) -> Dict:
        """Generate contextual response based on intent and session history"""
        
        # Get patient context if available
        context = ""
        if session.patient_data:
            context = f"Based on your current pregnancy status ({session.patient_data.gestational_week} weeks, age {session.patient_data.age}), "
        
        responses = {
            "symptom_report": f"{context}I understand you're experiencing some symptoms. Can you describe them in more detail? "
                             "Please tell me about the location, severity (1-10), duration, and any associated symptoms.",
            
            "medication_query": f"{context}For any medication questions during pregnancy, it's important to consult with your healthcare provider. "
                               "Can you tell me which specific medication you're asking about? "
                               "I can provide general safety information, but your doctor should make the final decision.",
            
            "nutrition_query": f"{context}Nutrition during pregnancy is very important. "
                              "Are you asking about specific foods, dietary restrictions, or general nutrition guidance? "
                              "I can help with pregnancy-safe food recommendations and nutritional needs.",
            
            "activity_query": f"{context}Staying active during pregnancy is generally beneficial. "
                             "Are you asking about specific exercises, or do you have concerns about physical activity? "
                             "I can provide guidance on safe pregnancy exercises.",
            
            "reassurance_seeking": f"{context}I understand your concerns. Many pregnancy symptoms and experiences are normal, "
                                  "but it's always good to stay informed. Can you share more details about what's worrying you?",
            
            "general_query": f"{context}I'm here to help with your pregnancy-related questions. "
                            "Could you provide more specific details about what you'd like to know?"
        }
        
        return {
            "response": responses.get(intent, responses["general_query"]),
            "message_type": MessageType.ASSISTANT,
            "requires_verification": False,
            "suggestions": self._get_follow_up_suggestions(intent),
            "severity": SeverityLevel.LOW
        }
    
    def _get_follow_up_suggestions(self, intent: str) -> List[str]:
        """Get follow-up suggestions based on intent"""
        suggestions = {
            "symptom_report": [
                "Rate pain severity 1-10",
                "Describe symptom duration",
                "Mention associated symptoms",
                "Note when symptoms started"
            ],
            "medication_query": [
                "Specify medication name",
                "Mention dosage if known",
                "Describe reason for taking",
                "Ask about alternatives"
            ],
            "nutrition_query": [
                "Ask about specific foods",
                "Inquire about supplements",
                "Discuss dietary restrictions",
                "Get meal planning help"
            ],
            "activity_query": [
                "Ask about safe exercises",
                "Discuss activity restrictions",
                "Get workout modifications",
                "Learn about warning signs"
            ]
        }
        
        return suggestions.get(intent, [])

# Initialize chat assistant
chat_assistant = ChatAssistant()

# Load models at startup
@app.on_event("startup")
async def startup_event():
    success = load_models()
    initialize_verification_rules()
    if not success:
        logger.error("Failed to load models. Please ensure models are trained and saved properly.")

# Enhanced prediction endpoint with reasoning
@app.post("/predict-enhanced", response_model=EnhancedPredictionResponse)
async def predict_with_reasoning(input_data: MaternalHealthInput):
    """Enhanced prediction with reasoning chain and verification"""
    try:
        if risk_model is None or health_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Prepare input data
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
        
        # Get basic predictions
        risk_idx = np.argmax(risk_pred[0])
        risk_level = risk_le.inverse_transform([risk_idx])[0]
        risk_confidence = float(risk_pred[0][risk_idx])
        
        health_idx = np.argmax(health_pred[0])
        health_recommendation = health_le.inverse_transform([health_idx])[0]
        health_confidence = float(health_pred[0][health_idx])
        
        # Advanced reasoning analysis
        reasoning_analysis = reasoner.analyze_with_reasoning(input_data)
        
        # Generate enhanced clinical insights
        clinical_insights = self._generate_enhanced_insights(input_data, reasoning_analysis)
        
        # Prepare probability dictionaries
        all_risk_probs = {
            label: float(prob) 
            for label, prob in zip(risk_le.classes_, risk_pred[0])
        }
        
        all_health_probs = {
            label: float(prob) 
            for label, prob in zip(health_le.classes_, health_pred[0])
        }
        
        return EnhancedPredictionResponse(
            risk_level=risk_level,
            risk_confidence=risk_confidence,
            health_recommendation=health_recommendation,
            health_confidence=health_confidence,
            clinical_insights=clinical_insights,
            verification_needed=reasoning_analysis["verification_needed"],
            verification_questions=reasoning_analysis["verification_questions"],
            severity_assessment=reasoning_analysis["severity"],
            immediate_actions=reasoning_analysis["immediate_actions"],
            follow_up_recommendations=reasoning_analysis["follow_up_recommendations"],
            all_risk_probabilities=all_risk_probs,
            all_health_probabilities=all_health_probs,
            reasoning_chain=reasoning_analysis["reasoning_chain"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Enhanced prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced prediction failed: {str(e)}")

def _generate_enhanced_insights(input_data: MaternalHealthInput, reasoning_analysis: Dict) -> List[str]:
    """Generate enhanced clinical insights"""
    insights = []
    
    # Add reasoning-based insights
    for concern in reasoning_analysis["concerns"]:
        if concern:
            if reasoning_analysis["severity"] == SeverityLevel.CRITICAL:
                insights.append(f"🚨 CRITICAL: {concern}")
            elif reasoning_analysis["severity"] == SeverityLevel.HIGH:
                insights.append(f"⚠️ HIGH RISK: {concern}")
            elif reasoning_analysis["severity"] == SeverityLevel.MEDIUM:
                insights.append(f"⚠️ MODERATE: {concern}")
            else:
                insights.append(f"ℹ️ NOTE: {concern}")
    
    # Add contextual insights based on pregnancy stage
    stage = "first trimester" if input_data.gestational_week < 14 else \
            "second trimester" if input_data.gestational_week < 28 else "third trimester"
    
    insights.append(f"📅 Currently in {stage} at {input_data.gestational_week} weeks gestation")
    
    # Add positive insights if low risk
    if reasoning_analysis["severity"] == SeverityLevel.LOW:
        insights.append("✅ Overall assessment shows low risk profile")
        insights.append("🎯 Continue current prenatal care routine")
    
    return insights

# Consultation management endpoints
@app.post("/consultation/start")
async def start_consultation(request: ConsultationRequest):
    """Start a new consultation session"""
    try:
        session_id = str(uuid.uuid4())
        
        session = ConsultationSession(
            session_id=session_id,
            user_id=request.user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            consultation_type=request.consultation_type,
            patient_data=request.patient_data.dict() if request.patient_data else None,
            chat_history=[],
            risk_assessments=[],
            verified_concerns=[],
            recommendations=[]
        )
        
        active_sessions[session_id] = session
        
        # Generate initial response
        initial_response = chat_assistant.process_message(request.initial_message, session)
        
        # Add initial messages to history
        session.chat_history.append({
            "message": request.initial_message,
            "message_type": MessageType.USER,
            "timestamp": datetime.now().isoformat()
        })
        
        session.chat_history.append({
            "message": initial_response["response"],
            "message_type": initial_response["message_type"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "severity": initial_response.get("severity", SeverityLevel.LOW).value,
                "requires_verification": initial_response.get("requires_verification", False)
            }
        })
        
        return {
            "session_id": session_id,
            "status": "consultation_started",
            "initial_response": initial_response,
            "consultation_type": request.consultation_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start consultation: {str(e)}")

@app.post("/consultation/chat")
async def chat_message(message: ChatMessage):
    """Send a chat message in consultation session"""
    try:
        if message.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Consultation session not found")
        
        session = active_sessions[message.session_id]
        session.last_activity = datetime.now()
        
        # Add user message to history
        session.chat_history.append({
            "message": message.message,
            "message_type": MessageType.USER,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process message and generate response
        response = chat_assistant.process_message(message.message, session)
        
        # Add assistant response to history
        session.chat_history.append({
            "message": response["response"],
            "message_type": response["message_type"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "severity": response.get("severity", SeverityLevel.LOW).value,
                "requires_verification": response.get("requires_verification", False),
                "suggestions": response.get("suggestions", [])
            }
        })
        
        # If verification needed, generate verification questions
        verification_questions = []
        if response.get("requires_verification"):
            verification_questions = self._generate_verification_questions(message.message, session)
        
        return {
            "response": response["response"],
            "message_type": response["message_type"],
            "severity": response.get("severity", SeverityLevel.LOW),
            "suggestions": response.get("suggestions", []),
            "verification_questions": verification_questions,
            "requires_verification": response.get("requires_verification", False),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat message processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

def _generate_verification_questions(message: str, session: ConsultationSession) -> List[str]:
    """Generate verification questions based on message content"""
    questions = []
    message_lower = message.lower()
    
    # Check for specific symptoms and generate relevant questions
    if any(word in message_lower for word in ["headache", "head"]):
        questions.extend([
            "On a scale of 1-10, how severe is your headache?",
            "Are you experiencing any vision changes or blurred vision?",
            "Do you have any nausea or vomiting with the headache?",
            "When did the headache start?"
        ])
    
    if any(word in message_lower for word in ["bleeding", "blood"]):
        questions.extend([
            "How heavy is the bleeding compared to a normal period?",
            "What color is the blood (bright red, dark red, brown)?",
            "Are you experiencing any cramping or pain?",
            "When did the bleeding start?"
        ])
    
    if any(word in message_lower for word in ["pain", "cramp"]):
        questions.extend([
            "Where exactly is the pain located?",
            "Rate the pain intensity from 1-10",
            "Is the pain constant or does it come and go?",
            "What makes the pain better or worse?"
        ])
    
    if any(word in message_lower for word in ["movement", "kick", "baby"]):
        questions.extend([
            "When did you last feel the baby move?",
            "How many movements do you typically feel per hour?",
            "Have you tried drinking cold water or lying on your side?",
            "Are you past 28 weeks of pregnancy?"
        ])
    
    return questions[:4]  # Limit to 4 most relevant questions

@app.post("/consultation/verify")
async def verify_concerns(verification: VerificationQuery):
    """Process verification responses and provide clinical reasoning"""
    try:
        if verification.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Consultation session not found")
        
        session = active_sessions[verification.session_id]
        session.last_activity = datetime.now()
        
        # Analyze verification responses
        verification_result = self._analyze_verification_responses(
            verification.concern_type, 
            verification.user_responses,
            verification.additional_symptoms or []
        )
        
        # Add to verified concerns
        session.verified_concerns.append({
            "concern_type": verification.concern_type,
            "responses": verification.user_responses,
            "analysis": verification_result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate clinical reasoning
        clinical_reasoning = self._generate_clinical_reasoning(verification_result, session)
        
        # Add reasoning to chat history
        session.chat_history.append({
            "message": clinical_reasoning["message"],
            "message_type": MessageType.VERIFICATION,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "severity": verification_result["severity"].value,
                "recommendations": clinical_reasoning["recommendations"]
            }
        })
        
        return {
            "verification_result": verification_result,
            "clinical_reasoning": clinical_reasoning,
            "session_updated": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Verification processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

def _analyze_verification_responses(self, concern_type: str, responses: Dict[str, str], additional_symptoms: List[str]) -> Dict:
    """Analyze verification responses and determine severity"""
    
    severity = SeverityLevel.LOW
    risk_factors = []
    recommendations = []
    
    # Analyze based on concern type
    if concern_type == "headache":
        intensity = int(responses.get("pain_intensity", "1"))
        vision_changes = responses.get("vision_changes", "").lower()
        nausea = responses.get("nausea", "").lower()
        
        if intensity >= 8 or "yes" in vision_changes:
            severity = SeverityLevel.CRITICAL
            risk_factors.append("Severe headache with possible preeclampsia signs")
            recommendations.append("Seek immediate medical attention")
        elif intensity >= 6 or "yes" in nausea:
            severity = SeverityLevel.HIGH
            risk_factors.append("Moderate to severe headache requiring evaluation")
            recommendations.append("Contact healthcare provider within 24 hours")
        else:
            severity = SeverityLevel.MEDIUM
            recommendations.append("Monitor symptoms and stay hydrated")
    
    elif concern_type == "bleeding":
        heaviness = responses.get("bleeding_heaviness", "").lower()
        color = responses.get("blood_color", "").lower()
        pain = responses.get("cramping", "").lower()
        
        if "heavy" in heaviness or "bright red" in color:
            severity = SeverityLevel.CRITICAL
            risk_factors.append("Heavy bleeding requiring immediate evaluation")
            recommendations.append("Go to emergency room immediately")
        elif "moderate" in heaviness or "yes" in pain:
            severity = SeverityLevel.HIGH
            risk_factors.append("Moderate bleeding with concerning features")
            recommendations.append("Contact healthcare provider immediately")
        else:
            severity = SeverityLevel.MEDIUM
            recommendations.append("Monitor closely and contact provider if worsens")
    
    elif concern_type == "fetal_movement":
        last_movement = responses.get("last_movement", "").lower()
        typical_movements = responses.get("typical_movements", "").lower()
        
        if "day" in last_movement or "yesterday" in last_movement:
            severity = SeverityLevel.CRITICAL
            risk_factors.append("Significantly decreased fetal movement")
            recommendations.append("Go to labor and delivery immediately")
        elif "hours" in last_movement and any(num in last_movement for num in ["6", "8", "10"]):
            severity = SeverityLevel.HIGH
            risk_factors.append("Decreased fetal movement requiring evaluation")
            recommendations.append("Contact healthcare provider immediately")
    
    return {
        "severity": severity,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "additional_symptoms": additional_symptoms,
        "analysis_timestamp": datetime.now().isoformat()
    }

def _generate_clinical_reasoning(self, verification_result: Dict, session: ConsultationSession) -> Dict:
    """Generate clinical reasoning based on verification results"""
    
    severity = verification_result["severity"]
    risk_factors = verification_result["risk_factors"]
    
    # Create reasoning message
    if severity == SeverityLevel.CRITICAL:
        message = ("🚨 **URGENT CLINICAL ASSESSMENT** 🚨\n\n"
                  "Based on your responses, I've identified serious concerns that require immediate medical attention:\n\n"
                  f"**Risk Factors Identified:**\n")
        for factor in risk_factors:
            message += f"• {factor}\n"
        
        message += ("\n**IMMEDIATE ACTION REQUIRED:**\n"
                   "Please contact your healthcare provider immediately or go to the nearest emergency room. "
                   "Do not wait or try to manage these symptoms at home.\n\n"
                   "**What to tell medical staff:**\n"
                   "• Your current gestational week\n"
                   "• All symptoms you're experiencing\n"
                   "• When symptoms started\n"
                   "• Any medications you're taking")
    
    elif severity == SeverityLevel.HIGH:
        message = ("⚠️ **CLINICAL CONCERN IDENTIFIED** ⚠️\n\n"
                  "Your symptoms require prompt medical evaluation:\n\n"
                  f"**Concerns:**\n")
        for factor in risk_factors:
            message += f"• {factor}\n"
        
        message += ("\n**Recommended Actions:**\n"
                   "• Contact your healthcare provider within 24 hours\n"
                   "• Monitor symptoms closely\n"
                   "• Seek immediate care if symptoms worsen\n"
                   "• Keep a symptom diary with times and descriptions")
    
    else:
        message = ("ℹ️ **CLINICAL ASSESSMENT** ℹ️\n\n"
                  "Based on your responses, your symptoms appear to be manageable but should be monitored:\n\n"
                  "**Current Assessment:**\n"
                  "• Symptoms are within a range that can be monitored at home\n"
                  "• Continue normal prenatal care schedule\n"
                  "• Watch for any changes or worsening\n\n"
                  "**When to seek care:**\n"
                  "• If symptoms worsen or new symptoms develop\n"
                  "• If you become concerned about any changes\n"
                  "• At your next scheduled prenatal appointment")
    
    return {
        "message": message,
        "recommendations": verification_result["recommendations"],
        "severity": severity.value,
        "requires_followup": severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
    }

# WebSocket for real-time chat
@app.websocket("/consultation/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    session = active_sessions[session_id]
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                continue
            
            # Process message
            session.last_activity = datetime.now()
            
            # Add user message to history
            session.chat_history.append({
                "message": message,
                "message_type": MessageType.USER,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate AI response
            response = chat_assistant.process_message(message, session)
            
            # Add assistant response to history
            session.chat_history.append({
                "message": response["response"],
                "message_type": response["message_type"],
                "timestamp": datetime.now().isoformat(),
                "metadata": response.get("metadata", {})
            })
            
            # Send response back to client
            await websocket.send_json({
                "response": response["response"],
                "message_type": response["message_type"].value,
                "severity": response.get("severity", SeverityLevel.LOW).value,
                "suggestions": response.get("suggestions", []),
                "requires_verification": response.get("requires_verification", False),
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"error": str(e)})

# Session management endpoints
@app.get("/consultation/session/{session_id}")
async def get_session_details(session_id: str):
    """Get consultation session details"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "consultation_type": session.consultation_type,
        "start_time": session.start_time.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "is_active": session.is_active,
        "chat_history_count": len(session.chat_history),
        "verified_concerns_count": len(session.verified_concerns),
        "has_patient_data": session.patient_data is not None
    }

@app.get("/consultation/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """Get chat history for session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Return latest messages (limited)
    recent_history = session.chat_history[-limit:] if len(session.chat_history) > limit else session.chat_history
    
    return {
        "session_id": session_id,
        "chat_history": recent_history,
        "total_messages": len(session.chat_history),
        "returned_messages": len(recent_history)
    }

@app.post("/consultation/end/{session_id}")
async def end_consultation(session_id: str):
    """End consultation session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    session.is_active = False
    
    # Generate session summary
    summary = self._generate_session_summary(session)
    
    # Remove from active sessions (in production, archive to database)
    del active_sessions[session_id]
    
    return {
        "session_ended": True,
        "session_id": session_id,
        "duration_minutes": (datetime.now() - session.start_time).seconds // 60,
        "total_messages": len(session.chat_history),
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }

def _generate_session_summary(self, session: ConsultationSession) -> Dict:
    """Generate summary of consultation session"""
    
    total_messages = len(session.chat_history)
    duration = (datetime.now() - session.start_time).seconds // 60
    
    # Count different message types
    user_messages = sum(1 for msg in session.chat_history if msg["message_type"] == MessageType.USER)
    system_messages = sum(1 for msg in session.chat_history if msg["message_type"] == MessageType.SYSTEM)
    
    # Extract key concerns
    key_concerns = []
    for concern in session.verified_concerns:
        key_concerns.append({
            "type": concern["concern_type"],
            "severity": concern["analysis"]["severity"].value,
            "timestamp": concern["timestamp"]
        })
    
    return {
        "duration_minutes": duration,
        "total_messages": total_messages,
        "user_messages": user_messages,
        "system_alerts": system_messages,
        "verified_concerns": len(session.verified_concerns),
        "key_concerns": key_concerns,
        "consultation_type": session.consultation_type.value,
        "had_emergency_alerts": system_messages > 0
    }

# Health monitoring endpoints
@app.get("/consultation/active-sessions")
async def get_active_sessions():
    """Get all active consultation sessions (for admin/monitoring)"""
    active_count = len(active_sessions)
    
    sessions_summary = []
    for session_id, session in active_sessions.items():
        sessions_summary.append({
            "session_id": session_id,
            "user_id": session.user_id,
            "consultation_type": session.consultation_type.value,
            "start_time": session.start_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "messages_count": len(session.chat_history),
            "concerns_count": len(session.verified_concerns)
        })
    
    return {
        "active_sessions_count": active_count,
        "sessions": sessions_summary,
        "timestamp": datetime.now().isoformat()
    }

# Health check with enhanced status
@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with system status"""
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": risk_model is not None and health_model is not None,
        "chat_assistant_ready": chat_assistant is not None,
        "reasoning_engine_ready": reasoner is not None,
        "active_consultations": len(active_sessions),
        "verification_rules_loaded": len(verification_rules) > 0,
        "version": "2.0.0"
    }

# Root endpoint with enhanced info
@app.get("/")
async def enhanced_root():
    """Enhanced API root endpoint"""
    return {
        "message": "Advanced Maternal Health AI Assistant",
        "version": "2.0.0",
        "features": [
            "AI-powered risk prediction",
            "Real-time chat consultation",
            "Clinical reasoning and verification",
            "Emergency symptom detection",
            "WebSocket support for real-time chat"
        ],
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "enhanced_prediction": "/predict-enhanced",
            "start_consultation": "/consultation/start",
            "chat": "/consultation/chat",
            "websocket": "/consultation/ws/{session_id}"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )