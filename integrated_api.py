import tensorflow as tf
import numpy as np
import pandas as pd
import json
import pickle
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import random
from datetime import datetime, timedelta
import os
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import Dict, List, Optional, Union

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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
    return obj

class AdvancedMaternalHealthAI:
    """
    Comprehensive Maternal Health AI System covering all aspects of maternal care
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.chat_model = None
        self.intents = None
        
    def generate_comprehensive_maternal_dataset(self):
        """Generate comprehensive maternal health dataset covering ALL fields"""
        np.random.seed(42)
        n_samples = 10000
        
        # Generate synthetic data covering all maternal health aspects
        data = {}
        
        # 1. DEMOGRAPHIC DATA
        data['age'] = np.random.normal(28, 6, n_samples).clip(15, 45)
        data['education_level'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        data['income_level'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.35, 0.25, 0.1])
        data['marital_status'] = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
        data['employment'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # 2. MEDICAL HISTORY
        data['previous_pregnancies'] = np.random.poisson(1.5, n_samples).clip(0, 8)
        data['previous_miscarriages'] = np.random.binomial(2, 0.15, n_samples)
        data['diabetes_history'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        data['hypertension_history'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        data['heart_disease'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        data['kidney_disease'] = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
        data['autoimmune_disorders'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        data['mental_health_history'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # 3. CURRENT PREGNANCY DATA
        data['gestational_age'] = np.random.uniform(4, 42, n_samples)
        data['weight_pre_pregnancy'] = np.random.normal(65, 15, n_samples).clip(40, 120)
        data['height'] = np.random.normal(165, 8, n_samples).clip(145, 185)
        data['bmi_pre_pregnancy'] = data['weight_pre_pregnancy'] / ((data['height']/100) ** 2)
        data['weight_gain'] = np.random.normal(12, 8, n_samples).clip(-5, 35)
        
        # 4. VITAL SIGNS & LAB VALUES
        data['systolic_bp'] = np.random.normal(120, 20, n_samples).clip(90, 180)
        data['diastolic_bp'] = np.random.normal(80, 15, n_samples).clip(60, 120)
        data['heart_rate'] = np.random.normal(80, 15, n_samples).clip(50, 120)
        data['hemoglobin'] = np.random.normal(11.5, 2, n_samples).clip(7, 16)
        data['glucose_fasting'] = np.random.normal(90, 20, n_samples).clip(60, 180)
        data['protein_urine'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02])
        data['white_blood_cells'] = np.random.normal(8000, 2000, n_samples).clip(3000, 15000)
        data['platelets'] = np.random.normal(250000, 50000, n_samples).clip(100000, 450000)
        
        # 5. LIFESTYLE FACTORS
        data['smoking'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        data['alcohol'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        data['drug_use'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        data['exercise_level'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.25, 0.05])
        data['stress_level'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.4, 0.2, 0.05])
        data['sleep_hours'] = np.random.normal(7, 2, n_samples).clip(4, 12)
        
        # Calculate risk factors and outcomes
        risk_factors = (
            (data['age'] > 35).astype(int) * 0.3 +
            (data['age'] < 18).astype(int) * 0.4 +
            (data['bmi_pre_pregnancy'] > 30).astype(int) * 0.25 +
            (data['bmi_pre_pregnancy'] < 18.5).astype(int) * 0.2 +
            data['diabetes_history'] * 0.4 +
            data['hypertension_history'] * 0.35 +
            (data['systolic_bp'] > 140).astype(int) * 0.3 +
            data['smoking'] * 0.3 +
            (data['hemoglobin'] < 9).astype(int) * 0.25 +
            (data['protein_urine'] > 1).astype(int) * 0.35
        )
        
        # TARGET VARIABLES
        data['risk_level'] = np.where(risk_factors < 0.5, 'Low',
                                    np.where(risk_factors < 1.0, 'Medium',
                                           np.where(risk_factors < 1.5, 'High', 'Critical')))
        
        data['gestational_diabetes'] = ((data['glucose_fasting'] > 126) | 
                                       (data['bmi_pre_pregnancy'] > 30) | 
                                       (data['age'] > 35)).astype(int)
        
        data['preeclampsia'] = ((data['systolic_bp'] > 140) | 
                               (data['protein_urine'] > 1) |
                               (data['age'] > 35)).astype(int)
        
        data['preterm_birth_risk'] = ((data['previous_miscarriages'] > 1) |
                                     data['smoking'] |
                                     (data['stress_level'] > 3)).astype(int)
        
        return pd.DataFrame(data)
    
    def generate_comprehensive_training_intents(self):
        """Generate comprehensive training data for chat"""
        intents_data = {
            "pregnancy_stages": {
                "patterns": [
                    "first trimester", "second trimester", "third trimester", "early pregnancy",
                    "late pregnancy", "pregnancy stages", "what to expect", "pregnancy timeline",
                    "pregnancy milestones", "fetal development", "pregnancy progression"
                ],
                "responses": [
                    "Pregnancy has three trimesters, each with unique developments. First trimester (0-12 weeks) focuses on organ formation, second (13-27 weeks) on growth and movement, and third (28-40 weeks) on final development. Which stage interests you?",
                    "Each pregnancy stage brings new changes. I can provide detailed information about what to expect during any trimester. What specific stage would you like to know about?",
                    "Pregnancy progression varies, but there are general milestones to track. Would you like information about a specific week or trimester?"
                ]
            },
            
            "nutrition_comprehensive": {
                "patterns": [
                    "nutrition", "diet", "eating", "food", "vitamins", "supplements", "folic acid",
                    "iron", "calcium", "protein", "calories", "weight gain", "healthy eating",
                    "prenatal vitamins", "omega 3", "dha", "meal planning", "food safety"
                ],
                "responses": [
                    "Proper nutrition is crucial during pregnancy. Focus on folate, iron, calcium, and DHA. Eat plenty of fruits, vegetables, lean proteins, and whole grains. Are you looking for specific nutritional guidance?",
                    "A balanced pregnancy diet includes 300-500 extra calories daily, adequate protein, and key nutrients. Prenatal vitamins help fill gaps. What nutrition questions do you have?",
                    "Food safety is important - avoid raw fish, undercooked meat, unpasteurized products, and high-mercury fish. Would you like a detailed food safety guide?"
                ]
            },
            
            "complications_conditions": {
                "patterns": [
                    "complications", "problems", "risks", "gestational diabetes", "preeclampsia",
                    "bleeding", "pain", "high blood pressure", "swelling", "preterm labor",
                    "miscarriage", "placenta", "contractions", "warning signs", "emergency"
                ],
                "responses": [
                    "Pregnancy complications can be managed with proper care. Common ones include gestational diabetes, preeclampsia, and preterm labor. Which concern would you like to discuss?",
                    "Warning signs include severe bleeding, severe headaches, vision changes, severe pain, or decreased fetal movement. Always contact your healthcare provider for concerning symptoms.",
                    "Most pregnancies proceed normally, but it's important to know warning signs and attend regular prenatal visits. What specific concern do you have?"
                ]
            },
            
            "mental_health_comprehensive": {
                "patterns": [
                    "anxiety", "depression", "mood", "stress", "emotional", "worried", "scared",
                    "overwhelmed", "mental health", "counseling", "support", "feelings",
                    "baby blues", "postpartum depression", "coping", "fear", "nervous"
                ],
                "responses": [
                    "Mental health is just as important as physical health during pregnancy. It's normal to feel various emotions, but persistent anxiety or depression needs attention. How are you feeling?",
                    "Pregnancy brings many emotional changes due to hormones and life transitions. Professional support is available and helpful. What specific feelings are you experiencing?",
                    "Taking care of your emotional wellbeing benefits both you and baby. Stress reduction, counseling, and support groups can help. Would you like coping strategies?"
                ]
            },
            
            "labor_delivery": {
                "patterns": [
                    "labor", "delivery", "birth", "contractions", "pain", "epidural", "natural birth",
                    "cesarean", "c-section", "hospital", "birth plan", "pushing", "crowning",
                    "breathing", "positions", "water birth", "midwife", "doula"
                ],
                "responses": [
                    "Labor preparation involves understanding the stages, breathing techniques, and pain management options. Would you like information about natural methods or medical pain relief?",
                    "Birth plans help communicate your preferences. Consider labor positions, pain management, and who you want present. What aspects of birth interest you most?",
                    "Labor signs include regular contractions, water breaking, and bloody show. When contractions are 5 minutes apart for an hour, it's time to contact your provider."
                ]
            },
            
            "emergency_situations": {
                "patterns": [
                    "emergency", "severe bleeding", "severe pain", "can't breathe", "chest pain",
                    "severe headache", "vision problems", "call doctor", "hospital now",
                    "something wrong", "urgent", "help", "911", "immediate"
                ],
                "responses": [
                    "🚨 This sounds urgent. Severe bleeding, intense pain, breathing difficulties, severe headaches, or vision changes need immediate medical attention. Call your doctor or go to the emergency room right away.",
                    "🚨 Don't wait - contact your healthcare provider immediately or go to the nearest emergency room. Trust your instincts when something feels seriously wrong.",
                    "🚨 Emergency warning signs include: heavy bleeding, severe abdominal pain, chest pain, difficulty breathing, severe headaches, or vision changes. Seek immediate medical care."
                ]
            },
            
            "general_support": {
                "patterns": [
                    "hello", "hi", "help", "support", "advice", "guidance", "questions",
                    "information", "pregnant", "pregnancy", "expecting", "baby", "mother"
                ],
                "responses": [
                    "Hello! I'm here to support you through your pregnancy journey. I can help with nutrition, health concerns, labor preparation, and emotional support. What would you like to know?",
                    "Hi there! Congratulations on your pregnancy! I'm here to provide information and support on all aspects of maternal health. How can I help you today?",
                    "Welcome! I can assist with pregnancy stages, nutrition, exercise, complications, mental health, and birth preparation. What questions do you have?"
                ]
            }
        }
        
        return intents_data
    
    def train_comprehensive_models(self, df):
        """Train multiple specialized models using different algorithms"""
        print("Training comprehensive maternal health models...")
        
        # Prepare features - using simplified feature set for API compatibility
        feature_columns = [
            'age', 'gestational_age', 'systolic_bp', 'diastolic_bp', 
            'glucose_fasting', 'heart_rate', 'bmi_pre_pregnancy',
            'previous_pregnancies', 'weight_gain', 'hemoglobin'
        ]
        
        # Ensure all required columns exist
        for col in feature_columns:
            if col not in df.columns:
                if col == 'glucose_fasting':
                    df[col] = df.get('blood_sugar', np.random.normal(90, 20, len(df))).clip(60, 180)
                elif col == 'bmi_pre_pregnancy':
                    df[col] = df.get('bmi', np.random.normal(25, 5, len(df))).clip(15, 50)
                else:
                    df[col] = 0
        
        X = df[feature_columns]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        # Train models for different targets
        models_to_train = {
            'risk_level': (df['risk_level'], 'multiclass'),
            'gestational_diabetes': (df['gestational_diabetes'], 'binary'),
            'preeclampsia': (df['preeclampsia'], 'binary'),
            'preterm_birth_risk': (df['preterm_birth_risk'], 'binary')
        }
        
        for target_name, (y, problem_type) in models_to_train.items():
            print(f"\nTraining models for {target_name}...")
            
            if problem_type == 'multiclass' and y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                self.encoders[target_name] = le
            else:
                y_encoded = y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Use Random Forest as default (good balance of performance and interpretability)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  {target_name} accuracy: {accuracy:.4f}")
            
            # Store the model
            self.models[target_name] = {
                'model': model,
                'algorithm': 'random_forest',
                'accuracy': accuracy,
                'features': feature_columns
            }
        
        # Train chat model
        self.train_chat_model()
        
        print("\n✅ All models trained successfully!")
    
    def train_chat_model(self):
        """Train chat model for conversational AI"""
        print("Training chat model...")
        
        intents_data = self.generate_comprehensive_training_intents()
        
        # Generate training examples
        training_texts = []
        training_labels = []
        
        # Data augmentation
        augmentation_patterns = [
            "I'm experiencing {}", "I have {}", "Tell me about {}",
            "Help with {}", "What about {}", "I'm concerned about {}",
            "Can you help me with {}", "I need information on {}"
        ]
        
        for intent, data in intents_data.items():
            for pattern in data["patterns"]:
                # Add original pattern
                training_texts.append(pattern.lower())
                training_labels.append(intent)
                
                # Add augmented versions
                for aug_pattern in augmentation_patterns:
                    augmented = aug_pattern.format(pattern).lower()
                    training_texts.append(augmented)
                    training_labels.append(intent)
        
        # Create pipeline
        self.chat_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                stop_words='english',
                sublinear_tf=True,
                min_df=2
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train
        self.chat_model.fit(training_texts, training_labels)
        self.intents = intents_data
        
        print("Chat model trained successfully!")
    
    def predict_health_risk(self, health_data):
        """Make comprehensive health predictions"""
        if not self.models:
            return {"error": "Models not trained yet"}
        
        # Prepare input data
        input_df = pd.DataFrame([health_data])
        
        # Get feature columns
        feature_columns = self.models['risk_level']['features']
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        X = input_df[feature_columns]
        X_scaled = self.scalers['main'].transform(X)
        
        predictions = {}
        
        # Make predictions with all models
        for target_name, model_info in self.models.items():
            try:
                model = model_info['model']
                pred = model.predict(X_scaled)[0]
                pred_proba = model.predict_proba(X_scaled)[0]
                
                # Decode if necessary
                if target_name in self.encoders:
                    pred = self.encoders[target_name].inverse_transform([pred])[0]
                
                predictions[target_name] = {
                    'prediction': convert_numpy_types(pred),
                    'confidence': convert_numpy_types(float(max(pred_proba))),
                    'algorithm_used': model_info['algorithm'],
                    'model_accuracy': convert_numpy_types(model_info['accuracy'])
                }
                
            except Exception as e:
                predictions[target_name] = {'error': str(e)}
        
        return convert_numpy_types(predictions)
    
    def get_chat_response(self, user_input, user_id=None):
        """Get chat response from trained model"""
        if not self.chat_model or not self.intents:
            return "I'm still learning. Please train me first."
        
        # Preprocess input
        processed_input = user_input.lower().strip()
        
        # Emergency detection
        emergency_patterns = [
            r'\b(severe|heavy|extreme)\s+(bleeding|pain|headache)\b',
            r'\b(chest pain|difficulty breathing|vision changes)\b',
            r'\b(can\'t breathe|cannot breathe)\b',
            r'\b(emergency|urgent|help|911)\b'
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, processed_input):
                return "🚨 URGENT: This sounds like a medical emergency. Please contact your healthcare provider immediately or go to the nearest emergency room. Don't wait - seek immediate medical attention."
        
        # Predict intent
        try:
            predicted_intent = self.chat_model.predict([processed_input])[0]
            confidence = max(self.chat_model.predict_proba([processed_input])[0])
            
            # Get response
            if predicted_intent in self.intents:
                responses = self.intents[predicted_intent]['responses']
                response = np.random.choice(responses)
                return response
            else:
                return "I understand you have a question about maternal health. Could you please be more specific? I can help with pregnancy stages, nutrition, complications, mental health, and more."
                
        except Exception as e:
            return f"I'm having trouble processing your question. Please try rephrasing it or contact your healthcare provider if it's urgent. Error: {str(e)}"
    
    def save_models(self):
        """Save all trained models"""
        try:
            os.makedirs('maternal_models', exist_ok=True)
            
            # Save main models and components
            for model_name, model_info in self.models.items():
                with open(f'maternal_models/{model_name}_model.pkl', 'wb') as f:
                    pickle.dump(model_info['model'], f)
            
            # Save scalers and encoders
            with open('maternal_models/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            with open('maternal_models/encoders.pkl', 'wb') as f:
                pickle.dump(self.encoders, f)
            
            # Save chat model
            if self.chat_model:
                with open('maternal_models/chat_model.pkl', 'wb') as f:
                    pickle.dump(self.chat_model, f)
            
            # Save intents
            if self.intents:
                with open('maternal_models/intents.json', 'w') as f:
                    json.dump(self.intents, f, indent=2)
            
            # Save model metadata
            model_metadata = {
                'models': {name: {k: convert_numpy_types(v) for k, v in info.items() if k != 'model'} 
                          for name, info in self.models.items()},
                'save_date': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open('maternal_models/metadata.json', 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            print("✅ Models saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load models
            model_files = [f for f in os.listdir('maternal_models') if f.endswith('_model.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '')
                with open(f'maternal_models/{model_file}', 'rb') as f:
                    model = pickle.load(f)
                
                # Load metadata
                with open('maternal_models/metadata.json', 'r') as f:
                    metadata = json.load(f)
                
                if model_name in metadata['models']:
                    self.models[model_name] = metadata['models'][model_name].copy()
                    self.models[model_name]['model'] = model
            
            # Load scalers and encoders
            with open('maternal_models/scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
            with open('maternal_models/encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            
            # Load chat model
            with open('maternal_models/chat_model.pkl', 'rb') as f:
                self.chat_model = pickle.load(f)
            
            # Load intents
            with open('maternal_models/intents.json', 'r') as f:
                self.intents = json.load(f)
            
            print("✅ Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False

# FastAPI Application
app = FastAPI(
    title="Advanced Maternal Health AI",
    description="Comprehensive AI-powered maternal health assessment and support system",
    version="2.0.0"
)

# Global AI system instance
ai_system = None

# Pydantic models for API
class MaternalHealthInput(BaseModel):
    age: int = Field(..., ge=16, le=50, description="Mother's age in years")
    gestational_age: float = Field(..., ge=4, le=42, description="Gestational age in weeks")
    systolic_bp: int = Field(..., ge=80, le=200, description="Systolic blood pressure")
    diastolic_bp: int = Field(..., ge=50, le=120, description="Diastolic blood pressure")
    glucose_fasting: float = Field(..., ge=60, le=250, description="Fasting blood glucose")
    heart_rate: int = Field(..., ge=50, le=150, description="Heart rate")
    bmi_pre_pregnancy: float = Field(..., ge=15, le=50, description="Pre-pregnancy BMI")
    previous_pregnancies: int = Field(..., ge=0, le=10, description="Previous pregnancies")
    weight_gain: float = Field(..., ge=-20, le=80, description="Weight gain in pounds")
    hemoglobin: float = Field(..., ge=7, le=16, description="Hemoglobin level")

class ChatInput(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    user_id: Optional[str] = Field(default="anonymous")

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    user_id: str
    emergency_detected: bool = False

# Load AI system on startup
@app.on_event("startup")
async def startup_event():
    global ai_system
    ai_system = AdvancedMaternalHealthAI()
    
    # Try to load existing models
    if not ai_system.load_models():
        print("Training new models...")
        dataset = ai_system.generate_comprehensive_maternal_dataset()
        ai_system.train_comprehensive_models(dataset)
        ai_system.save_models()
    
    print("✅ AI System ready!")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Advanced Maternal Health AI System",
        "version": "2.0.0",
        "features": [
            "🤖 Conversational AI Support",
            "📊 Risk Assessment",
            "💡 Health Recommendations",
            "🚨 Emergency Detection"
        ]
    }

@app.post("/predict", response_model=Dict)
async def predict_health_risk(input_data: MaternalHealthInput):
    """Make health risk predictions"""
    try:
        if ai_system is None:
            raise HTTPException(status_code=503, detail="AI System not loaded")
        
        # Convert input to dict
        health_data = input_data.dict()
        
        # Make prediction
        predictions = ai_system.predict_health_risk(health_data)
        
        # Generate clinical insights
        insights = []
        if input_data.age > 35:
            insights.append("Advanced maternal age - increased monitoring recommended")
        if input_data.systolic_bp > 140:
            insights.append("Hypertension detected - immediate medical evaluation needed")
        if input_data.glucose_fasting > 140:
            insights.append("High blood glucose - gestational diabetes screening recommended")
        
        result = {
            'predictions': predictions,
            'clinical_insights': insights,
            'timestamp': datetime.now().isoformat()
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(chat_input: ChatInput):
    """Chat with the maternal health AI assistant"""
    try:
        if ai_system is None:
            raise HTTPException(status_code=503, detail="AI System not loaded")
        
        # Get response from AI
        response = ai_system.get_chat_response(chat_input.message, chat_input.user_id)
        
        # Check for emergency
        emergency_keywords = ["🚨", "emergency", "urgent", "immediate"]
        emergency_detected = any(keyword in response.lower() for keyword in emergency_keywords)
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat(),
            user_id=chat_input.user_id,
            emergency_detected=emergency_detected
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrated-consultation")
async def integrated_consultation(health_data: MaternalHealthInput, question: str):
    """Get integrated consultation with predictions and chat"""
    try:
        if ai_system is None:
            raise HTTPException(status_code=503, detail="AI System not loaded")
        
        # Make prediction
        predictions = ai_system.predict_health_risk(health_data.dict())
        
        # Get chat response
        chat_response = ai_system.get_chat_response(question)
        
        # Generate recommendations based on risk level
        recommendations = []
        risk_level = predictions.get('risk_level', {}).get('prediction', 'Unknown')
        
        if risk_level == 'High' or risk_level == 'Critical':
            recommendations.extend([
                "🚨 Schedule immediate consultation with your healthcare provider",
                "📊 Monitor vital signs closely",
                "🏥 Consider high-risk pregnancy specialist consultation"
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                "⚠️ Increase frequency of prenatal visits",
                "📋 Keep daily log of symptoms",
                "🩺 Follow up on any concerning symptoms"
            ])
        else:
            recommendations.extend([
                "✅ Continue with regular prenatal care",
                "🌟 Maintain healthy lifestyle",
                "📅 Attend scheduled prenatal appointments"
            ])
        
        # Add specific recommendations
        if health_data.systolic_bp > 140:
            recommendations.append("🩺 Blood pressure management is crucial")
        if health_data.glucose_fasting > 125:
            recommendations.append("🍎 Consider nutritional counseling")
        if health_data.age > 35:
            recommendations.append("🧬 Discuss genetic screening options")
        
        result = {
            'chat_response': chat_response,
            'predictions': predictions,
            'recommendations': recommendations,
            'risk_summary': {
                'level': risk_level,
                'confidence': predictions.get('risk_level', {}).get('confidence', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        logger.error(f"Integrated consultation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "ai_system_loaded": ai_system is not None,
        "models_available": len(ai_system.models) if ai_system else 0,
        "chat_available": ai_system.chat_model is not None if ai_system else False
    }

# Demo function
def demo_system():
    """Demonstrate the system capabilities"""
    print("🤱 Advanced Maternal Health AI System Demo")
    print("=" * 50)
    
    # Initialize system
    ai_system = AdvancedMaternalHealthAI()
    
    # Try to load or train models
    if not ai_system.load_models():
        print("Training new models...")
        dataset = ai_system.generate_comprehensive_maternal_dataset()
        ai_system.train_comprehensive_models(dataset)
        ai_system.save_models()
    
    # Demo prediction
    print("\n1. 📊 Health Risk Assessment Demo")
    print("-" * 30)
    
    sample_patient = {
        'age': 32,
        'gestational_age': 28,
        'systolic_bp': 135,
        'diastolic_bp': 85,
        'glucose_fasting': 95,
        'heart_rate': 75,
        'bmi_pre_pregnancy': 26.5,
        'previous_pregnancies': 1,
        'weight_gain': 12,
        'hemoglobin': 10.5
    }
    
    predictions = ai_system.predict_health_risk(sample_patient)
    print(f"Patient: 32-year-old, 28 weeks pregnant")
    print("Predictions:")
    for pred_name, pred_info in predictions.items():
        if 'error' not in pred_info:
            print(f"  {pred_name}: {pred_info['prediction']} (Confidence: {pred_info['confidence']:.2f})")
    
    # Demo chat
    print("\n2. 💬 Chat System Demo")
    print("-" * 20)
    
    sample_questions = [
        "I'm having severe headaches, should I worry?",
        "What should I eat during my second trimester?",
        "I'm feeling anxious about labor",
        "Tell me about gestational diabetes",
        "Hello, I'm pregnant and need support"
    ]
    
    for question in sample_questions:
        print(f"\nQ: {question}")
        response = ai_system.get_chat_response(question)
        print(f"A: {response}")
    
    print("\n" + "="*50)
    print("✅ Demo completed! System is ready for use.")
    print("To run the API server, use: uvicorn integrated_maternal_health:app --reload")

if __name__ == "__main__":
    # Check if we should run demo or API
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_system()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)