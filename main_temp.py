#!/usr/bin/env python3
"""
Practical Enhanced Health Consultation Implementation
Integrates advanced reasoning with existing ML models from HealthDataPipeline
Redis dependency removed, using in-memory storage
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, asdict
import json
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthConsultationRequest:
    """Enhanced health consultation request"""
    # Non-default arguments (required fields)
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BloodSugar: float
    BodyTemp: float
    HeartRate: float
    BMI: float
    # Default arguments (optional fields)
    GestationalWeek: float = 0.0
    PreviousPregnancies: int = 0
    WeightGain: float = 0.0
    query: str = ""
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    conversation_context: Optional[Dict] = None
    preferred_language: str = "english"
    urgency_level: Optional[str] = None
    symptoms: List[str] = None
    medical_history: List[str] = None
    current_medications: List[str] = None

# New Pydantic model for /chat endpoint
class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    preferred_language: str = "english"

class AdvancedDataPreprocessor:
    """Replicated from HealthDataPipeline to handle feature engineering"""
    
    def engineer_health_features(self, df):
        """Create health-specific engineered features"""
        logger.info("Engineering health-specific features...")
        
        df_eng = df.copy()
        
        # BMI categories
        if 'BMI' in df_eng.columns:
            df_eng['BMI_Category'] = pd.cut(df_eng['BMI'], 
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            df_eng['BMI_Category'] = df_eng['BMI_Category'].astype(str)
        
        # Blood pressure categories
        if 'SystolicBP' in df_eng.columns and 'DiastolicBP' in df_eng.columns:
            df_eng['BP_Category'] = 'Normal'
            df_eng.loc[(df_eng['SystolicBP'] >= 140) | (df_eng['DiastolicBP'] >= 90), 'BP_Category'] = 'Hypertensive'
            df_eng.loc[(df_eng['SystolicBP'] >= 120) & (df_eng['SystolicBP'] < 140) & 
                      (df_eng['DiastolicBP'] < 90), 'BP_Category'] = 'Prehypertensive'
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
        
        # Risk factor combinations
        risk_factors = []
        if 'BMI' in df_eng.columns:
            risk_factors.append((df_eng['BMI'] > 30).astype(int))
        if 'SystolicBP' in df_eng.columns:
            risk_factors.append((df_eng['SystolicBP'] > 140).astype(int))
        if 'BloodSugar' in df_eng.columns:
            risk_factors.append((df_eng['BloodSugar'] > 126).astype(int))
        
        if risk_factors:
            df_eng['TotalRiskFactors'] = sum(risk_factors)
        
        logger.info(f"Created {len(df_eng.columns) - len(df.columns)} new features")
        return df_eng

class EnhancedHealthConsultationSystem:
    """Enhanced health consultation system with advanced reasoning"""
    
    def __init__(self):
        self.ml_models = self.load_ml_models()
        self.preprocessor = AdvancedDataPreprocessor()
        self.response_generator = ResponseGenerator()
        self.knowledge_base = MedicalKnowledgeBase()
        self.safety_checker = SafetyChecker()
        self.conversation_sessions = {}  # In-memory storage for conversation history
    
    def load_ml_models(self) -> Dict[str, Any]:
        """Load ML models saved by HealthDataPipeline"""
        try:
            models = {
                'best_model': joblib.load('models/best_model.pkl'),
                'scaler': joblib.load('models/scaler.pkl'),
                'target_encoder': joblib.load('models/target_encoder.pkl') if os.path.exists('models/target_encoder.pkl') else None,
                'label_encoders': joblib.load('models/label_encoders.pkl') if os.path.exists('models/label_encoders.pkl') else {},
                'feature_columns': pd.read_csv('models/feature_columns.csv')['features'].tolist()
            }
            with open('models/pipeline_metadata.json', 'r') as f:
                models['metadata'] = json.load(f)
            logger.info("ML models and metadata loaded successfully")
            return models
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return None
    
    async def process_consultation(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Process enhanced health consultation"""
        
        try:
            # 1. Safety check first
            safety_result = await self.safety_checker.check_safety(request)
            if safety_result['requires_immediate_attention']:
                return self.create_emergency_response(safety_result)
            
            # 2. Get ML predictions
            ml_prediction = await self.get_ml_prediction(request)
            
            # 3. Process with reasoning system
            reasoning_result = await self.process_reasoning(request)
            
            # 4. Integrate knowledge base
            knowledge_enhancement = await self.knowledge_base.enhance_response(
                request.query, ml_prediction, reasoning_result
            )
            
            # 5. Generate comprehensive response
            final_response = await self.response_generator.generate_comprehensive_response(
                request, ml_prediction, reasoning_result, knowledge_enhancement
            )
            
            # 6. Store conversation context
            await self.store_conversation_context(request.user_id, request.session_id, 
                                                 request.query, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in consultation processing: {e}")
            return self.create_error_response(str(e))
    
    async def process_chat(self, request: ChatRequest) -> Dict[str, Any]:
        """Process conversational chat queries"""
        try:
            # Create a minimal HealthConsultationRequest for reasoning
            health_request = HealthConsultationRequest(
                Age=30,  # Default values for minimal processing
                SystolicBP=120,
                DiastolicBP=80,
                BloodSugar=100,
                BodyTemp=98.6,
                HeartRate=75,
                BMI=23,
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                preferred_language=request.preferred_language
            )
            
            # Process reasoning and knowledge enhancement
            reasoning_result = await self.process_reasoning(health_request)
            knowledge_enhancement = await self.knowledge_base.enhance_response(
                request.query, {}, reasoning_result
            )
            
            # Generate chat response with health tips
            response = await self.response_generator.generate_chat_response(
                health_request, reasoning_result, knowledge_enhancement
            )
            
            # Store conversation context
            await self.store_conversation_context(
                request.user_id, request.session_id, request.query, response
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return self.create_error_response(str(e))
    
    async def get_ml_prediction(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Get ML model predictions with feature engineering"""
        
        if not self.ml_models:
            return {'error': 'ML models not available'}
        
        # Prepare features as DataFrame
        feature_data = pd.DataFrame([{
            'Age': request.Age,
            'GestationalWeek': request.GestationalWeek,
            'SystolicBP': request.SystolicBP,
            'DiastolicBP': request.DiastolicBP,
            'BloodSugar': request.BloodSugar,
            'BodyTemp': request.BodyTemp,
            'HeartRate': request.HeartRate,
            'BMI': request.BMI,
            'PreviousPregnancies': request.PreviousPregnancies,
            'WeightGain': request.WeightGain
        }])
        
        # Engineer features
        feature_data = self.preprocessor.engineer_health_features(feature_data)
        
        # Handle missing features
        for col in self.ml_models['feature_columns']:
            if col not in feature_data.columns:
                feature_data[col] = 0
        
        # Encode categorical features
        for col, encoder in self.ml_models['label_encoders'].items():
            if col in feature_data.columns:
                try:
                    feature_data[col] = encoder.transform(feature_data[col].astype(str))
                except ValueError:
                    feature_data[col] = encoder.transform([encoder.classes_[0]] * len(feature_data))
        
        # Select and order features
        X = feature_data[self.ml_models['feature_columns']]
        
        # Scale features
        X_scaled = self.ml_models['scaler'].transform(X)
        
        # Get risk prediction
        risk_prediction = self.ml_models['best_model'].predict(X_scaled)[0]
        risk_probability = self.ml_models['best_model'].predict_proba(X_scaled)[0]
        
        # Decode risk level
        risk_level = self.ml_models['target_encoder'].inverse_transform([risk_prediction])[0]
        
        # Map probabilities to class names
        class_names = self.ml_models['metadata']['class_names']
        risk_prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(risk_probability)}
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.ml_models['best_model'], 'feature_importances_'):
            feature_importance = dict(zip(self.ml_models['feature_columns'], 
                                        self.ml_models['best_model'].feature_importances_.tolist()))
        
        return {
            'risk_level': risk_level,
            'risk_probability': risk_prob_dict,
            'recommendation': None,  # No recommendation model in first code
            'feature_importance': feature_importance,
            'confidence': float(np.max(risk_probability))
        }
    
    async def process_reasoning(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Process with reasoning system"""
        
        reasoning = {
            'analytical': await self.analyze_logical_patterns(request.query),
            'causal': await self.analyze_causal_relationships(request),
            'contextual': await self.analyze_context(request),
            'confidence': 0.8
        }
        
        return reasoning
    
    async def analyze_logical_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze logical patterns in the query"""
        
        patterns = {
            'has_conditional': 'if' in query.lower(),
            'has_causal': any(word in query.lower() for word in ['because', 'due to', 'caused by']),
            'has_comparison': any(word in query.lower() for word in ['better', 'worse', 'compared']),
            'has_temporal': any(word in query.lower() for word in ['when', 'after', 'before', 'during']),
        }
        
        complexity_score = sum(patterns.values()) / len(patterns)
        
        return {
            'patterns': patterns,
            'complexity_score': complexity_score,
            'requires_evidence': complexity_score > 0.3
        }
    
    async def analyze_causal_relationships(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Analyze potential causal relationships"""
        
        potential_causes = []
        
        if request.BMI > 30:
            potential_causes.append({'factor': 'obesity', 'strength': 0.8, 'category': 'lifestyle'})
        
        if request.SystolicBP > 140 or request.DiastolicBP > 90:
            potential_causes.append({'factor': 'hypertension', 'strength': 0.9, 'category': 'medical'})
        
        if request.Age > 35:
            potential_causes.append({'factor': 'advanced_age', 'strength': 0.6, 'category': 'demographic'})
        
        if request.GestationalWeek > 20:
            potential_causes.append({'factor': 'pregnancy', 'strength': 0.7, 'category': 'physiological'})
        
        return {
            'potential_causes': potential_causes,
            'causal_complexity': len(potential_causes),
            'primary_cause': max(potential_causes, key=lambda x: x['strength']) if potential_causes else None
        }
    
    async def analyze_context(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Analyze contextual factors"""
        
        conversation_history = await self.get_conversation_history(request.user_id, request.session_id)
        
        urgency_keywords = {
            'high': ['emergency', 'severe', 'critical', 'urgent', 'immediately'],
            'moderate': ['concerned', 'worried', 'unusual', 'significant'],
            'low': ['mild', 'slight', 'minor', 'routine']
        }
        
        urgency_level = 'low'
        query_lower = request.query.lower()
        
        for level, keywords in urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                urgency_level = level
                break
        
        emotional_indicators = {
            'anxiety': ['worried', 'anxious', 'scared', 'nervous'],
            'frustration': ['frustrated', 'annoyed', 'upset'],
            'confusion': ['confused', 'don\'t understand', 'unclear']
        }
        
        emotions = {}
        for emotion, keywords in emotional_indicators.items():
            emotions[emotion] = any(keyword in query_lower for keyword in keywords)
        
        return {
            'urgency_level': urgency_level,
            'emotional_context': emotions,
            'conversation_length': len(conversation_history) if conversation_history else 0,
            'topic_consistency': self.assess_topic_consistency(conversation_history, request.query)
        }
    
    async def get_conversation_history(self, user_id: str, session_id: str) -> List[Dict]:
        """Get conversation history from in-memory storage"""
        
        key = f"{user_id}:{session_id or f'{user_id}_default'}"
        return self.conversation_sessions.get(key, [])
    
    async def store_conversation_context(self, user_id: str, session_id: str, 
                                       query: str, response: Dict):
        """Store conversation context in memory"""
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_summary': response.get('summary', ''),
            'risk_level': response.get('risk_assessment', {}).get('level', 'unknown')
        }
        
        key = f"{user_id}:{session_id or f'{user_id}_default'}"
        if key not in self.conversation_sessions:
            self.conversation_sessions[key] = []
        self.conversation_sessions[key].append(conversation_entry)
        
        # Keep only last 20 entries
        self.conversation_sessions[key] = self.conversation_sessions[key][-20:]
    
    def assess_topic_consistency(self, history: List[Dict], current_query: str) -> float:
        """Assess topic consistency in the conversation"""
        
        if not history:
            return 1.0
        
        current_keywords = set(current_query.lower().split())
        
        consistency_scores = []
        for entry in history[-3:]:
            past_keywords = set(entry.get('query', '').lower().split())
            if past_keywords:
                overlap = len(current_keywords & past_keywords)
                total = len(current_keywords | past_keywords)
                consistency_scores.append(overlap / total if total > 0 else 0)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def create_emergency_response(self, safety_result: Dict) -> Dict[str, Any]:
        """Create emergency response for urgent situations"""
        
        return {
            'response_type': 'emergency',
            'message': 'Based on the information provided, this appears to require immediate medical attention.',
            'recommendations': [
                'Seek emergency medical care immediately',
                'Contact your healthcare provider',
                'If symptoms worsen, call emergency services'
            ],
            'risk_assessment': {
                'level': 'critical',
                'requires_immediate_attention': True,
                'safety_concerns': safety_result.get('concerns', [])
            },
            'confidence': 1.0,
            'disclaimer': 'This is not a substitute for professional medical advice. Seek immediate medical attention.'
        }
    
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        
        return {
            'response_type': 'error',
            'message': 'I apologize, but I encountered an issue processing your request.',
            'error': error_message,
            'recommendations': [
                'Please try rephrasing your question',
                'Contact support if the issue persists',
                'For urgent medical concerns, consult a healthcare provider immediately'
            ],
            'confidence': 0.0
        }

class ResponseGenerator:
    """Generate comprehensive, intelligent responses"""
    
    def __init__(self):
        self.response_templates = self.load_response_templates()
        self.medical_knowledge = MedicalKnowledgeBase()
    
    async def generate_comprehensive_response(self, request: HealthConsultationRequest,
                                            ml_prediction: Dict, reasoning_result: Dict,
                                            knowledge_enhancement: Dict) -> Dict[str, Any]:
        """Generate comprehensive response integrating all analysis"""
        
        strategy = self._determine_response_strategy(request, ml_prediction, reasoning_result)
        
        main_response = await self._create_main_response(strategy, request, ml_prediction, reasoning_result)
        educational_content = await self._generate_educational_content(request, ml_prediction, reasoning_result)
        recommendations = await self._generate_actionable_recommendations(request, ml_prediction, reasoning_result)
        follow_up_plan = await self._create_follow_up_plan(request, ml_prediction, reasoning_result)
        
        overall_confidence = self._calculate_overall_confidence(ml_prediction, reasoning_result)
        
        return {
            'response_type': strategy,
            'main_response': main_response,
            'risk_assessment': {
                'ml_risk_level': ml_prediction.get('risk_level', 'Unknown'),
                'ml_confidence': ml_prediction.get('confidence', 0.0),
                'reasoning_risk_factors': reasoning_result.get('causal', {}).get('potential_causes', []),
                'combined_risk_score': self._calculate_combined_risk_score(ml_prediction, reasoning_result),
                'urgency_level': reasoning_result.get('contextual', {}).get('urgency_level', 'low')
            },
            'recommendations': recommendations,
            'educational_content': educational_content,
            'follow_up_plan': follow_up_plan,
            'reasoning_transparency': self._explain_reasoning(reasoning_result),
            'confidence_metrics': {
                'overall_confidence': overall_confidence,
                'ml_confidence': ml_prediction.get('confidence', 0.0),
                'reasoning_confidence': reasoning_result.get('confidence', 0.0)
            },
            'clinical_insights': self._generate_clinical_insights(request, ml_prediction, reasoning_result),
            'safety_alerts': await self._generate_safety_alerts(request, ml_prediction, reasoning_result)
        }
    
    async def generate_chat_response(self, request: HealthConsultationRequest,
                                   reasoning_result: Dict, knowledge_enhancement: Dict) -> Dict[str, Any]:
        """Generate response for chat endpoint"""
        
        strategy = self._determine_response_strategy(request, {}, reasoning_result)
        
        main_response = await self._create_main_response(strategy, request, {}, reasoning_result)
        educational_content = await self._generate_educational_content(request, {}, reasoning_result)
        recommendations = await self._generate_actionable_recommendations(request, {}, reasoning_result)
        
        overall_confidence = reasoning_result.get('confidence', 0.8)
        
        return {
            'response_type': strategy,
            'main_response': main_response,
            'health_tips': educational_content['health_tips'],
            'lifestyle_recommendations': educational_content['lifestyle_recommendations'],
            'educational_content': educational_content['key_concepts'],
            'recommendations': recommendations,
            'confidence': overall_confidence,
            'disclaimer': 'This is general health information and not a substitute for professional medical advice.'
        }
    
    def _determine_response_strategy(self, request: HealthConsultationRequest, 
                                   ml_prediction: Dict, reasoning_result: Dict) -> str:
        """Determine the most appropriate response strategy"""
        
        if self._is_emergency_case(request, ml_prediction, reasoning_result):
            return 'emergency'
        
        if self._is_high_risk_case(request, ml_prediction, reasoning_result):
            return 'high_risk_guidance'
        
        if self._needs_education(request, reasoning_result):
            return 'educational_supportive'
        
        if self._needs_monitoring(request, ml_prediction):
            return 'monitoring_guidance'
        
        return 'standard_consultation'
    
    async def _create_main_response(self, strategy: str, request: HealthConsultationRequest,
                                  ml_prediction: Dict, reasoning_result: Dict) -> str:
        """Create the main response based on strategy and analysis"""
        
        response_parts = []
        
        contextual = reasoning_result.get('contextual', {})
        if contextual.get('emotional_context'):
            emotions = contextual['emotional_context']
            if emotions.get('anxiety', False):
                response_parts.append("I understand you're feeling concerned about your health. ")
            elif emotions.get('confusion', False):
                response_parts.append("Let me help clarify your health situation. ")
        
        if strategy == 'emergency':
            response_parts.append(self._create_emergency_response_text(request, ml_prediction, reasoning_result))
        
        elif strategy == 'high_risk_guidance':
            response_parts.append(self._create_high_risk_response_text(request, ml_prediction, reasoning_result))
        
        elif strategy == 'educational_supportive':
            response_parts.append(self._create_educational_response_text(request, ml_prediction, reasoning_result))
        
        elif strategy == 'monitoring_guidance':
            response_parts.append(self._create_monitoring_response_text(request, ml_prediction, reasoning_result))
        
        else:
            response_parts.append(self._create_standard_response_text(request, ml_prediction, reasoning_result))
        
        causal_analysis = reasoning_result.get('causal', {})
        if causal_analysis.get('potential_causes'):
            strongest_cause = max(causal_analysis['potential_causes'], 
                               key=lambda x: x['strength'])
            cause_factor = strongest_cause.get('factor', '')
            if cause_factor:
                response_parts.append(f"The analysis suggests that {cause_factor} may be a significant factor in your current health status. ")
        
        if ml_prediction.get('feature_importance'):
            top_factors = sorted(ml_prediction['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:2]
            factor_names = [self._humanize_feature_name(factor[0]) for factor in top_factors]
            if factor_names:
                response_parts.append(f"Based on the data analysis, {' and '.join(factor_names)} appear to be the most important factors for your assessment. ")
        
        return "".join(response_parts)
    
    def _create_emergency_response_text(self, request: HealthConsultationRequest,
                                      ml_prediction: Dict, reasoning_result: Dict) -> str:
        """Create emergency response text"""
        
        response = "Based on the information provided, I'm very concerned about your current symptoms. "
        
        risk_factors = []
        if request.SystolicBP > 160 or request.DiastolicBP > 100:
            risk_factors.append("severely elevated blood pressure")
        if request.BloodSugar > 180:
            risk_factors.append("very high blood sugar levels")
        if request.BodyTemp > 101:
            risk_factors.append("high fever")
        
        if risk_factors:
            response += f"The concerning factors include: {', '.join(risk_factors)}. "
        
        response += "This situation requires immediate medical attention. Please seek emergency care or contact your healthcare provider immediately. "
        
        return response
    
    def _create_high_risk_response_text(self, request: HealthConsultationRequest,
                                      ml_prediction: Dict, reasoning_result: Dict) -> str:
        """Create high-risk response text"""
        
        response = "Your current health indicators suggest an elevated risk level that requires attention. "
        
        risk_level = ml_prediction.get('risk_level', 'Unknown')
        confidence = ml_prediction.get('confidence', 0.0)
        response += f"The predictive analysis indicates a {risk_level.lower()} risk level with {confidence:.0%} confidence. "
        
        causal = reasoning_result.get('causal', {})
        if causal.get('potential_causes'):
            primary_causes = [cause['factor'] for cause in causal['potential_causes'][:2]]
            response += f"Key contributing factors appear to be: {', '.join(primary_causes)}. "
        
        response += "I strongly recommend consulting with your healthcare provider for a comprehensive evaluation and personalized treatment plan. "
        
        return response
    
    def _create_educational_response_text(self, request: HealthConsultationRequest,
                                        ml_prediction: Dict, reasoning_result: Dict) -> str:
        """Create educational response text"""
        
        response = "Thank you for sharing your health information. Let me provide some insights based on your data. "
        
        analytical = reasoning_result.get('analytical', {})
        if analytical.get('requires_evidence'):
            response += "To give you the most accurate guidance, it would be helpful to understand more about your symptoms and their timeline. "
        
        if request.BMI > 25:
            response += "Your BMI indicates that weight management could be beneficial for your overall health. "
        
        if request.GestationalWeek > 0:
            stage = "early" if request.GestationalWeek < 20 else "later"
            response += f"Since you're in the {stage} stages of pregnancy, monitoring key health indicators is especially important. "
        
        return response
    
    def _create_monitoring_response_text(self, request: HealthConsultationRequest,
                                       ml_prediction: Dict, reasoning_result: Dict) -> str:
        """Create monitoring guidance response text"""
        
        response = "Based on your health profile, regular monitoring of certain indicators would be beneficial. "
        
        monitoring_areas = []
        if request.SystolicBP > 130 or request.DiastolicBP > 85:
            monitoring_areas.append("blood pressure")
        if request.BloodSugar > 125:
            monitoring_areas.append("blood glucose levels")
        if request.GestationalWeek > 0 and request.Age > 35:
            monitoring_areas.append("pregnancy progression")
        
        if monitoring_areas:
            response += f"I recommend focusing on: {', '.join(monitoring_areas)}. "
        
        return response
    
    def _create_standard_response_text(self, request: HealthConsultationRequest,
                                     ml_prediction: Dict, reasoning_result: Dict) -> str:
        """Create standard consultation response text"""
        
        response = "Thank you for providing your health information. Based on the comprehensive analysis, here's what I found: "
        
        risk_level = ml_prediction.get('risk_level', '').lower()
        if 'low' in risk_level:
            response += "Your overall health indicators look encouraging. "
        
        improvement_areas = self._identify_improvement_areas(request)
        if improvement_areas:
            response += f"There are opportunities to enhance your health in areas such as: {', '.join(improvement_areas)}. "
        
        return response
    
    async def _generate_educational_content(self, request: HealthConsultationRequest,
                                          ml_prediction: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """Generate educational content with enhanced health tips"""
        
        educational_content = {
            'key_concepts': [],
            'health_tips': [],
            'lifestyle_recommendations': [],
            'when_to_seek_help': []
        }
        
        # General health tips based on query
        query_lower = request.query.lower()
        if 'blood pressure' in query_lower or request.SystolicBP > 130 or request.DiastolicBP > 85:
            educational_content['key_concepts'].append({
                'topic': 'Blood Pressure Management',
                'explanation': 'Blood pressure readings above 130/85 may indicate prehypertension or hypertension. Regular monitoring and lifestyle changes can help manage blood pressure effectively.',
                'importance': 'high'
            })
            educational_content['health_tips'].extend([
                'Monitor blood pressure at the same time daily using a validated home device.',
                'Practice stress-reduction techniques like deep breathing or meditation.',
                'Limit caffeine intake, as it can temporarily elevate blood pressure.'
            ])
            educational_content['lifestyle_recommendations'].extend([
                'Reduce sodium intake to less than 2,300 mg per day, ideally 1,500 mg.',
                'Engage in 150 minutes of moderate aerobic activity per week, like brisk walking.',
                'Maintain a healthy weight to reduce strain on your cardiovascular system.'
            ])
        
        if 'blood sugar' in query_lower or request.BloodSugar > 125:
            educational_content['key_concepts'].append({
                'topic': 'Blood Glucose Control',
                'explanation': 'Elevated blood glucose levels may indicate prediabetes or diabetes. Proper management through diet and lifestyle is crucial.',
                'importance': 'high'
            })
            educational_content['health_tips'].extend([
                'Check blood sugar levels as recommended, typically before and after meals.',
                'Eat smaller, frequent meals to stabilize blood glucose levels.',
                'Choose low-glycemic-index foods like whole grains and vegetables.'
            ])
            educational_content['lifestyle_recommendations'].extend([
                'Follow a balanced diet with complex carbohydrates and controlled portions.',
                'Incorporate regular physical activity to improve insulin sensitivity.',
                'Work with a dietitian to create a personalized meal plan.'
            ])
        
        if 'weight' in query_lower or request.BMI > 30:
            educational_content['key_concepts'].append({
                'topic': 'Weight Management',
                'explanation': 'A BMI over 30 indicates obesity, which increases risks for heart disease, diabetes, and other conditions. Gradual, sustainable weight loss is beneficial.',
                'importance': 'moderate'
            })
            educational_content['health_tips'].extend([
                'Track daily calorie intake using a food diary or app.',
                'Incorporate strength training to build muscle and boost metabolism.',
                'Set realistic weight loss goals, aiming for 0.5-1 kg per week.'
            ])
            educational_content['lifestyle_recommendations'].extend([
                'Combine 150 minutes of moderate exercise with a balanced, calorie-controlled diet.',
                'Replace sugary drinks with water or unsweetened teas.',
                'Plan meals ahead to avoid impulsive eating choices.'
            ])
        
        if 'pregnancy' in query_lower or request.GestationalWeek > 0:
            educational_content['key_concepts'].append({
                'topic': 'Prenatal Health',
                'explanation': 'During pregnancy, regular monitoring of health indicators is essential for maternal and fetal wellbeing.',
                'importance': 'high'
            })
            educational_content['health_tips'].extend([
                'Take prenatal vitamins as prescribed, including folic acid.',
                'Stay active with pregnancy-safe exercises like walking or yoga.',
                'Report any unusual symptoms, such as swelling or severe headaches, to your doctor.'
            ])
            educational_content['lifestyle_recommendations'].extend([
                'Follow a nutrient-rich diet with adequate protein, iron, and calcium.',
                'Get 7-9 hours of sleep nightly to support pregnancy health.',
                'Attend all scheduled prenatal appointments for regular monitoring.'
            ])
        
        # General wellness tips for all users
        educational_content['health_tips'].extend([
            'Stay hydrated by drinking 8-10 glasses of water daily.',
            'Aim for 7-9 hours of quality sleep each night to support overall health.',
            'Practice stress management through mindfulness or hobbies.'
        ])
        educational_content['lifestyle_recommendations'].extend([
            'Incorporate a variety of fruits and vegetables into your daily diet.',
            'Limit processed foods high in sugar, salt, and unhealthy fats.',
            'Schedule regular health check-ups to catch issues early.'
        ])
        
        educational_content['when_to_seek_help'] = [
            'Blood pressure consistently above 140/90 mmHg.',
            'Blood sugar levels consistently above 180 mg/dL.',
            'Unusual symptoms like chest pain, shortness of breath, or severe headaches.',
            'Any concerning changes during pregnancy, such as reduced fetal movement.',
            'Persistent fatigue, dizziness, or other unexplained symptoms.'
        ]
        
        return educational_content
    
    async def _generate_actionable_recommendations(self, request: HealthConsultationRequest,
                                                 ml_prediction: Dict, reasoning_result: Dict) -> List[Dict]:
        """Generate specific, actionable recommendations"""
        
        recommendations = []
        
        priority_issues = self._identify_priority_issues(request, ml_prediction, reasoning_result)
        
        for issue in priority_issues:
            if issue['type'] == 'blood_pressure':
                recommendations.append({
                    'category': 'immediate',
                    'action': 'Monitor blood pressure daily',
                    'details': 'Record readings at the same time each day and track patterns',
                    'timeline': 'Start immediately',
                    'importance': 'high'
                })
                recommendations.append({
                    'category': 'lifestyle',
                    'action': 'Reduce sodium intake',
                    'details': 'Aim for less than 2,300mg sodium per day, ideally 1,500mg',
                    'timeline': 'Begin within this week',
                    'importance': 'high'
                })
            
            elif issue['type'] == 'blood_sugar':
                recommendations.append({
                    'category': 'immediate',
                    'action': 'Monitor blood glucose levels',
                    'details': 'Check as recommended by healthcare provider, typically before meals',
                    'timeline': 'Start immediately',
                    'importance': 'high'
                })
                recommendations.append({
                    'category': 'dietary',
                    'action': 'Implement carbohydrate counting',
                    'details': 'Work with a dietitian to learn proper portion control and meal planning',
                    'timeline': 'Schedule within 2 weeks',
                    'importance': 'high'
                })
            
            elif issue['type'] == 'weight_management':
                recommendations.append({
                    'category': 'lifestyle',
                    'action': 'Create a structured exercise plan',
                    'details': 'Start with 150 minutes of moderate activity per week, as tolerated',
                    'timeline': 'Begin gradually over next 2 weeks',
                    'importance': 'moderate'
                })
            
            elif issue['type'] == 'prenatal_care':
                recommendations.append({
                    'category': 'medical',
                    'action': 'Schedule prenatal appointment',
                    'details': 'Discuss current health indicators and any concerns with your OB/GYN',
                    'timeline': 'Within 1-2 weeks',
                    'importance': 'high'
                })
        
        recommendations.append({
            'category': 'wellness',
            'action': 'Maintain regular sleep schedule',
            'details': 'Aim for 7-9 hours of quality sleep per night',
            'timeline': 'Ongoing',
            'importance': 'moderate'
        })
        
        recommendations.append({
            'category': 'wellness',
            'action': 'Stay hydrated',
            'details': 'Drink adequate water throughout the day, approximately 8 glasses',
            'timeline': 'Daily',
            'importance': 'moderate'
        })
        
        return recommendations
    
    async def _create_follow_up_plan(self, request: HealthConsultationRequest,
                                   ml_prediction: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """Create a comprehensive follow-up plan"""
        
        follow_up_plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_objectives': [],
            'monitoring_schedule': {},
            'next_consultation': {}
        }
        
        urgency = reasoning_result.get('contextual', {}).get('urgency_level', 'low')
        if urgency == 'high':
            follow_up_plan['immediate_actions'].append('Contact healthcare provider today')
            follow_up_plan['immediate_actions'].append('Monitor symptoms closely')
        elif urgency == 'moderate':
            follow_up_plan['immediate_actions'].append('Schedule appointment with healthcare provider within 1 week')
        
        if request.SystolicBP > 140:
            follow_up_plan['short_term_goals'].append('Achieve blood pressure below 140/90')
        if request.BloodSugar > 140:
            follow_up_plan['short_term_goals'].append('Stabilize blood glucose levels')
        if request.BMI > 30:
            follow_up_plan['short_term_goals'].append('Begin structured weight management program')
        
        follow_up_plan['long_term_objectives'].append('Maintain healthy lifestyle habits')
        if request.GestationalWeek > 0:
            follow_up_plan['long_term_objectives'].append('Ensure healthy pregnancy progression')
        
        if request.SystolicBP > 130:
            follow_up_plan['monitoring_schedule']['blood_pressure'] = 'Daily for 2 weeks, then weekly'
        if request.BloodSugar > 125:
            follow_up_plan['monitoring_schedule']['blood_glucose'] = 'As directed by healthcare provider'
        
        risk_level = ml_prediction.get('risk_level', '').lower()
        if 'high' in risk_level:
            follow_up_plan['next_consultation'] = {
                'timeframe': '1-2 weeks',
                'focus': 'Review progress and adjust treatment plan'
            }
        elif 'medium' in risk_level:
            follow_up_plan['next_consultation'] = {
                'timeframe': '2-4 weeks',
                'focus': 'Monitor improvements and provide ongoing guidance'
            }
        else:
            follow_up_plan['next_consultation'] = {
                'timeframe': '1-3 months',
                'focus': 'Routine check-in and preventive care'
            }
        
        return follow_up_plan
    
    def _explain_reasoning(self, reasoning_result: Dict) -> Dict[str, Any]:
        """Provide transparent explanation of the reasoning process"""
        
        explanation = {
            'analysis_summary': {},
            'key_findings': [],
            'confidence_factors': {},
            'limitations': []
        }
        
        analytical = reasoning_result.get('analytical', {})
        if analytical:
            explanation['analysis_summary']['logical_patterns'] = analytical.get('patterns', {})
            explanation['analysis_summary']['complexity_assessment'] = analytical.get('complexity_score', 0)
        
        causal = reasoning_result.get('causal', {})
        if causal:
            explanation['key_findings'].append(f"Identified {len(causal.get('potential_causes', []))} potential contributing factors")
            if causal.get('primary_cause'):
                primary = causal['primary_cause']
                explanation['key_findings'].append(f"Primary factor appears to be: {primary['factor']} (strength: {primary['strength']:.1f})")
        
        contextual = reasoning_result.get('contextual', {})
        if contextual:
            urgency = contextual.get('urgency_level', 'low')
            explanation['key_findings'].append(f"Urgency level assessed as: {urgency}")
        
        explanation['confidence_factors'] = {
            'data_completeness': 'High - comprehensive health indicators provided',
            'pattern_recognition': 'Moderate - based on established clinical patterns',
            'individual_variation': 'Consider - individual factors may influence outcomes'
        }
        
        explanation['limitations'] = [
            'This analysis is based on provided data and established patterns',
            'Individual medical history and context may affect recommendations',
            'Professional medical evaluation is recommended for definitive diagnosis',
            'Recommendations are general guidance, not specific medical advice'
        ]
        
        return explanation
    
    def _calculate_overall_confidence(self, ml_prediction: Dict, reasoning_result: Dict) -> float:
        """Calculate overall confidence score"""
        
        ml_confidence = ml_prediction.get('confidence', 0.0)
        reasoning_confidence = reasoning_result.get('confidence', 0.0)
        
        if ml_confidence > 0.8:
            return (ml_confidence * 0.7) + (reasoning_confidence * 0.3)
        else:
            return (ml_confidence * 0.5) + (reasoning_confidence * 0.5)
    
    def _calculate_combined_risk_score(self, ml_prediction: Dict, reasoning_result: Dict) -> float:
        """Calculate combined risk score from ML and reasoning"""
        
        ml_risk = ml_prediction.get('risk_level', '').lower()
        ml_score = 0.3 if 'low' in ml_risk else 0.6 if 'medium' in ml_risk else 0.9
        
        urgency = reasoning_result.get('contextual', {}).get('urgency_level', 'low')
        urgency_score = 0.2 if urgency == 'low' else 0.5 if urgency == 'moderate' else 0.8
        
        causal_complexity = len(reasoning_result.get('causal', {}).get('potential_causes', []))
        complexity_score = min(causal_complexity * 0.15, 0.6)
        
        combined_score = (ml_score * 0.5) + (urgency_score * 0.3) + (complexity_score * 0.2)
        
        return min(combined_score, 1.0)
    
    def _generate_clinical_insights(self, request: HealthConsultationRequest,
                                  ml_prediction: Dict, reasoning_result: Dict) -> List[str]:
        """Generate clinical insights based on comprehensive analysis"""
        
        insights = []
        
        if request.SystolicBP > 140 or request.DiastolicBP > 90:
            insights.append("🔍 Hypertension detected - this significantly increases cardiovascular risk")
        elif request.SystolicBP > 130 or request.DiastolicBP > 85:
            insights.append("⚠️ Pre-hypertension identified - lifestyle modifications can prevent progression")
        
        if request.BloodSugar > 140:
            insights.append("🔍 Hyperglycemia present - diabetes screening and management needed")
        elif request.BloodSugar > 125:
            insights.append("⚠️ Elevated glucose - prediabetes possible, dietary changes recommended")
        
        if request.BMI > 30:
            insights.append("⚠️ Obesity classification - increased risk for multiple health conditions")
        elif request.BMI < 18.5:
            insights.append("⚠️ Underweight classification - nutritional assessment recommended")
        
        if request.GestationalWeek > 0:
            if request.Age > 35:
                insights.append("ℹ️ Advanced maternal age - enhanced prenatal monitoring recommended")
            if request.GestationalWeek > 37:
                insights.append("ℹ️ Near term pregnancy - delivery planning should be discussed")
            if request.BloodSugar > 125 and request.GestationalWeek > 20:
                insights.append("🔍 Possible gestational diabetes - glucose tolerance test needed")
        
        if request.HeartRate > 100:
            insights.append("⚠️ Tachycardia present - underlying cause should be investigated")
        elif request.HeartRate < 60:
            insights.append("ℹ️ Bradycardia noted - may be normal for athletes or indicate underlying condition")
        
        if request.BodyTemp > 100.4:
            insights.append("🔍 Fever detected - possible infection or inflammatory process")
        
        causal = reasoning_result.get('causal', {})
        if causal.get('causal_complexity', 0) > 3:
            insights.append("🔍 Multiple risk factors present - comprehensive management approach needed")
        
        return insights
    
    async def _generate_safety_alerts(self, request: HealthConsultationRequest,
                                     ml_prediction: Dict, reasoning_result: Dict) -> List[Dict]:
        """Generate safety alerts for critical situations"""
        
        alerts = []
        
        if request.SystolicBP > 180 or request.DiastolicBP > 110:
            alerts.append({
                'level': 'critical',
                'message': 'Hypertensive crisis - seek emergency medical care immediately',
                'action': 'Go to emergency room or call 911'
            })
        
        if request.BloodSugar > 250:
            alerts.append({
                'level': 'critical',
                'message': 'Severe hyperglycemia - immediate medical attention required',
                'action': 'Contact healthcare provider or emergency services'
            })
        
        if request.BodyTemp > 103:
            alerts.append({
                'level': 'urgent',
                'message': 'High fever present - medical evaluation needed',
                'action': 'Contact healthcare provider today'
            })
        
        if request.GestationalWeek > 20:
            if request.SystolicBP > 140 and request.DiastolicBP > 90:
                alerts.append({
                    'level': 'urgent',
                    'message': 'Possible pregnancy-induced hypertension',
                    'action': 'Contact OB/GYN immediately'
                })
        
        risk_factors = sum([
            request.SystolicBP > 140,
            request.DiastolicBP > 90,
            request.BloodSugar > 140,
            request.BMI > 35,
            request.BodyTemp > 100.4
        ])
        
        if risk_factors >= 3:
            alerts.append({
                'level': 'urgent',
                'message': 'Multiple high-risk factors present',
                'action': 'Schedule comprehensive medical evaluation within 24-48 hours'
            })
        
        return alerts
    
    def _identify_priority_issues(self, request: HealthConsultationRequest,
                                ml_prediction: Dict, reasoning_result: Dict) -> List[Dict]:
        """Identify priority health issues to address"""
        
        issues = []
        
        if request.SystolicBP > 130 or request.DiastolicBP > 85:
            issues.append({
                'type': 'blood_pressure',
                'severity': 'high' if request.SystolicBP > 140 else 'moderate',
                'priority': 1
            })
        
        if request.BloodSugar > 125:
            issues.append({
                'type': 'blood_sugar',
                'severity': 'high' if request.BloodSugar > 140 else 'moderate',
                'priority': 1 if request.BloodSugar > 140 else 2
            })
        
        if request.BMI > 30 or request.BMI < 18.5:
            issues.append({
                'type': 'weight_management',
                'severity': 'moderate',
                'priority': 3
            })
        
        if request.GestationalWeek > 0:
            issues.append({
                'type': 'prenatal_care',
                'severity': 'high' if request.Age > 35 or any([
                    request.SystolicBP > 140,
                    request.BloodSugar > 125,
                    request.BMI > 30
                ]) else 'moderate',
                'priority': 1 if request.Age > 35 else 2
            })
        
        return sorted(issues, key=lambda x: x['priority'])
    
    def _identify_improvement_areas(self, request: HealthConsultationRequest) -> List[str]:
        """Identify areas for health improvement"""
        
        areas = []
        
        if request.BMI > 25:
            areas.append("weight management")
        
        if request.SystolicBP > 120 or request.DiastolicBP > 80:
            areas.append("blood pressure control")
        
        if request.BloodSugar > 100:
            areas.append("glucose management")
        
        if request.HeartRate > 90:
            areas.append("cardiovascular fitness")
        
        return areas
    
    def _humanize_feature_name(self, feature_name: str) -> str:
        """Convert feature names to human-readable format"""
        
        name_mapping = {
            'Age': 'age',
            'GestationalWeek': 'gestational week',
            'SystolicBP': 'systolic blood pressure',
            'DiastolicBP': 'diastolic blood pressure',
            'BloodSugar': 'blood sugar levels',
            'BodyTemp': 'body temperature',
            'HeartRate': 'heart rate',
            'BMI': 'body mass index',
            'PreviousPregnancies': 'pregnancy history',
            'WeightGain': 'weight changes',
            'BMI_Category': 'BMI category',
            'BP_Category': 'blood pressure category',
            'PulsePressure': 'pulse pressure',
            'AgeGroup': 'age group',
            'IsPregnant': 'pregnancy status',
            'Trimester': 'pregnancy trimester',
            'BloodSugar_Category': 'blood sugar category',
            'HeartRate_Pct_Max': 'heart rate percentage of maximum',
            'HR_Zone': 'heart rate zone',
            'TotalRiskFactors': 'total risk factors'
        }
        
        return name_mapping.get(feature_name, feature_name.lower())
    
    def _is_emergency_case(self, request: HealthConsultationRequest,
                          ml_prediction: Dict, reasoning_result: Dict) -> bool:
        """Determine if this is an emergency case"""
        
        emergency_indicators = [
            request.SystolicBP > 180 or request.DiastolicBP > 110,
            request.BloodSugar > 250,
            request.BodyTemp > 103,
            reasoning_result.get('contextual', {}).get('urgency_level') == 'high'
        ]
        
        return any(emergency_indicators)
    
    def _is_high_risk_case(self, request: HealthConsultationRequest,
                          ml_prediction: Dict, reasoning_result: Dict) -> bool:
        """Determine if this is a high-risk case"""
        
        high_risk_indicators = [
            ml_prediction.get('risk_level', '').lower() == 'high',
            request.SystolicBP > 140 or request.DiastolicBP > 90,
            request.BloodSugar > 140,
            request.GestationalWeek > 0 and request.Age > 35,
            reasoning_result.get('contextual', {}).get('urgency_level') == 'moderate'
        ]
        
        return any(high_risk_indicators)
    
    def _needs_education(self, request: HealthConsultationRequest, reasoning_result: Dict) -> bool:
        """Determine if educational content is needed"""
        
        analytical = reasoning_result.get('analytical', {})
        return (
            analytical.get('requires_evidence', False) or
            request.query and '?' in request.query or
            'help' in request.query.lower() or
            'understand' in request.query.lower()
        )
    
    def _needs_monitoring(self, request: HealthConsultationRequest, ml_prediction: Dict) -> bool:
        """Determine if ongoing monitoring is needed"""
        
        return (
            ml_prediction.get('risk_level', '').lower() == 'medium' or
            request.SystolicBP > 130 or
            request.BloodSugar > 100 or
            request.GestationalWeek > 0
        )
    
    def load_response_templates(self) -> Dict[str, str]:
        """Load response templates for different scenarios"""
        
        return {
            'emergency': "Based on your symptoms, this requires immediate medical attention. {specific_concerns} Please seek emergency care or contact your healthcare provider immediately.",
            'high_risk': "Your health indicators suggest an elevated risk that needs attention. {risk_factors} I recommend consulting with your healthcare provider for a comprehensive evaluation.",
            'educational': "Thank you for sharing your health information. {educational_content} Here's what I found based on your data:",
            'monitoring': "Your health profile suggests that regular monitoring would be beneficial. {monitoring_recommendations}",
            'standard': "Based on your comprehensive health assessment, {assessment_summary}"
        }

class MedicalKnowledgeBase:
    """Medical knowledge base for enhanced responses"""
    
    def __init__(self):
        self.condition_database = self.load_condition_database()
        self.treatment_guidelines = self.load_treatment_guidelines()
        self.risk_factors = self.load_risk_factors()
    
    async def enhance_response(self, query: str, ml_prediction: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """Enhance response with medical knowledge"""
        
        enhancement = {
            'relevant_conditions': await self.identify_relevant_conditions(query, ml_prediction),
            'clinical_guidelines': await self.get_clinical_guidelines(ml_prediction),
            'risk_stratification': await self.perform_risk_stratification(ml_prediction, reasoning_result),
            'evidence_based_recommendations': await self.get_evidence_based_recommendations(ml_prediction)
        }
        
        return enhancement
    
    async def identify_relevant_conditions(self, query: str, ml_prediction: Dict) -> List[Dict]:
        """Identify relevant medical conditions"""
        
        conditions = []
        risk_level = ml_prediction.get('risk_level', '').lower()
        
        if 'pregnancy' in query.lower() or 'pregnant' in query.lower():
            conditions.append({
                'condition': 'Pregnancy Management',
                'relevance': 'high',
                'description': 'Comprehensive prenatal care and monitoring'
            })
        
        if 'blood pressure' in query.lower() or risk_level == 'high':
            conditions.append({
                'condition': 'Hypertension',
                'relevance': 'moderate',
                'description': 'Elevated blood pressure requiring management'
            })
        
        if 'blood sugar' in query.lower():
            conditions.append({
                'condition': 'Diabetes Risk',
                'relevance': 'moderate',
                'description': 'Potential for prediabetes or diabetes based on glucose concerns'
            })
        
        if 'weight' in query.lower():
            conditions.append({
                'condition': 'Weight Management',
                'relevance': 'moderate',
                'description': 'Strategies for maintaining a healthy weight'
            })
        
        return conditions
    
    async def get_clinical_guidelines(self, ml_prediction: Dict) -> Dict[str, Any]:
        """Get relevant clinical guidelines"""
        
        guidelines = {}
        risk_level = ml_prediction.get('risk_level', '').lower()
        
        if 'high' in risk_level:
            guidelines['hypertension'] = {
                'source': 'AHA/ACC Guidelines',
                'recommendation': 'Blood pressure goal <130/80 mmHg for most adults',
                'evidence_level': 'Class I, Level of Evidence A'
            }
        
        return guidelines
    
    async def perform_risk_stratification(self, ml_prediction: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        """Perform clinical risk stratification"""
        
        stratification = {
            'cardiovascular_risk': 'low',
            'diabetes_risk': 'low',
            'pregnancy_risk': 'low'
        }
        
        return stratification
    
    async def get_evidence_based_recommendations(self, ml_prediction: Dict) -> List[Dict]:
        """Get evidence-based recommendations"""
        
        recommendations = []
        
        recommendations.append({
            'intervention': 'Lifestyle Modification',
            'evidence': 'Multiple RCTs show 5-10 mmHg reduction in systolic BP',
            'strength': 'Strong recommendation',
            'applicability': 'All patients with elevated BP'
        })
        
        return recommendations
    
    def load_condition_database(self) -> Dict[str, Any]:
        """Load medical condition database"""
        return {
            'hypertension': {
                'definition': 'Blood pressure ≥130/80 mmHg',
                'risk_factors': ['age', 'obesity', 'diabetes', 'smoking'],
                'complications': ['stroke', 'heart_attack', 'kidney_disease']
            },
            'diabetes': {
                'definition': 'Fasting blood glucose ≥126 mg/dL',
                'risk_factors': ['obesity', 'family_history', 'sedentary_lifestyle'],
                'complications': ['neuropathy', 'retinopathy', 'cardiovascular_disease']
            },
            'obesity': {
                'definition': 'BMI ≥30',
                'risk_factors': ['poor_diet', 'lack_of_exercise', 'genetics'],
                'complications': ['diabetes', 'hypertension', 'joint_problems']
            }
        }
    
    def load_treatment_guidelines(self) -> Dict[str, Any]:
        """Load treatment guidelines"""
        return {
            'hypertension': {
                'first_line': ['ACE inhibitors', 'ARBs', 'thiazide diuretics', 'CCBs'],
                'lifestyle': ['diet', 'exercise', 'weight_loss', 'smoking_cessation']
            },
            'diabetes': {
                'first_line': ['metformin', 'insulin'],
                'lifestyle': ['dietary_changes', 'regular_exercise', 'weight_management']
            }
        }
    
    def load_risk_factors(self) -> Dict[str, Any]:
        """Load risk factor database"""
        return {
            'cardiovascular': {
                'major': ['hypertension', 'diabetes', 'smoking', 'dyslipidemia'],
                'minor': ['obesity', 'sedentary_lifestyle', 'stress']
            },
            'diabetes': {
                'major': ['obesity', 'family_history', 'gestational_diabetes'],
                'minor': ['sedentary_lifestyle', 'poor_diet']
            }
        }

class SafetyChecker:
    """Safety checker for critical health situations"""
    
    def __init__(self):
        self.emergency_thresholds = self.load_emergency_thresholds()
        self.warning_signs = self.load_warning_signs()
    
    async def check_safety(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Comprehensive safety check"""
        
        safety_result = {
            'requires_immediate_attention': False,
            'warning_level': 'low',
            'concerns': [],
            'recommended_actions': []
        }
        
        if await self.check_critical_vitals(request):
            safety_result['requires_immediate_attention'] = True
            safety_result['warning_level'] = 'critical'
            safety_result['concerns'].append('Critical vital signs detected')
            safety_result['recommended_actions'].append('Seek emergency medical care immediately')
        
        if request.GestationalWeek > 0:
            pregnancy_safety = await self.check_pregnancy_safety(request)
            if pregnancy_safety['high_risk']:
                safety_result['warning_level'] = max(safety_result['warning_level'], 'high')
                safety_result['concerns'].extend(pregnancy_safety['concerns'])
                safety_result['recommended_actions'].extend(pregnancy_safety['actions'])
        
        combination_risk = await self.check_dangerous_combinations(request)
        if combination_risk['risk_detected']:
            safety_result['warning_level'] = max(safety_result['warning_level'], 'moderate')
            safety_result['concerns'].extend(combination_risk['concerns'])
        
        return safety_result
    
    async def check_critical_vitals(self, request: HealthConsultationRequest) -> bool:
        """Check for critically abnormal vital signs"""
        
        critical_conditions = [
            request.SystolicBP > 180 or request.DiastolicBP > 110,
            request.BloodSugar > 250,
            request.BodyTemp > 103,
            request.HeartRate > 120 or request.HeartRate < 50
        ]
        
        return any(critical_conditions)
    
    async def check_pregnancy_safety(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Check pregnancy-specific safety concerns"""
        
        safety = {
            'high_risk': False,
            'concerns': [],
            'actions': []
        }
        
        if request.SystolicBP > 140 and request.DiastolicBP > 90 and request.GestationalWeek > 20:
            safety['high_risk'] = True
            safety['concerns'].append('Possible preeclampsia')
            safety['actions'].append('Contact OB/GYN immediately')
        
        if request.BloodSugar > 140 and request.GestationalWeek > 24:
            safety['concerns'].append('Possible gestational diabetes')
            safety['actions'].append('Schedule glucose tolerance test')
        
        return safety
    
    async def check_dangerous_combinations(self, request: HealthConsultationRequest) -> Dict[str, Any]:
        """Check for dangerous combinations of factors"""
        
        risk_assessment = {
            'risk_detected': False,
            'concerns': []
        }
        
        cv_risks = sum([
            request.SystolicBP > 140,
            request.BloodSugar > 125,
            request.BMI > 30,
            request.Age > 45
        ])
        
        if cv_risks >= 3:
            risk_assessment['risk_detected'] = True
            risk_assessment['concerns'].append('Multiple cardiovascular risk factors present')
        
        return risk_assessment
    
    def load_emergency_thresholds(self) -> Dict[str, Dict]:
        """Load emergency threshold values"""
        return {
            'blood_pressure': {'systolic': 180, 'diastolic': 110},
            'blood_sugar': 250,
            'temperature': 103,
            'heart_rate': {'max': 120, 'min': 50}
        }
    
    def load_warning_signs(self) -> List[str]:
        """Load warning signs to watch for"""
        return [
            'chest_pain',
            'difficulty_breathing',
            'severe_headache',
            'vision_changes',
            'severe_abdominal_pain'
        ]

# FastAPI Application Setup
app = FastAPI(title="Enhanced Health Consultation API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
enhanced_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global enhanced_system
    
    logger.info("Initializing Enhanced Health Consultation System...")
    enhanced_system = EnhancedHealthConsultationSystem()
    
    if not enhanced_system.ml_models:
        logger.warning("ML models not loaded - some features may be limited")
    
    logger.info("System ready!")
    
    yield
    
    logger.info("Shutting down Enhanced Health Consultation System...")

app.router.lifespan_context = lifespan

# API Endpoints
@app.post("/enhanced-consultation")
async def enhanced_consultation(request: HealthConsultationRequest):
    """Enhanced health consultation endpoint with advanced reasoning"""
    
    try:
        if not enhanced_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        result = await enhanced_system.process_consultation(request)
        
        return {
            "status": "success",
            "consultation_id": f"consult_{datetime.now().timestamp()}",
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This consultation is for informational purposes only and does not replace professional medical advice."
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced consultation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for conversational health queries"""
    
    try:
        if not enhanced_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        result = await enhanced_system.process_chat(request)
        
        return {
            "status": "success",
            "chat_id": f"chat_{datetime.now().timestamp()}",
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This is general health information and not a substitute for professional medical advice."
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quick-assessment")
async def quick_assessment(
    Age: float,
    SystolicBP: float,
    DiastolicBP: float,
    BloodSugar: float,
    BMI: float,
    query: str = "General health assessment"
):
    """Quick health assessment endpoint"""
    
    request = HealthConsultationRequest(
        Age=Age,
        SystolicBP=SystolicBP,
        DiastolicBP=DiastolicBP,
        BloodSugar=BloodSugar,
        BodyTemp=98.6,
        HeartRate=75,
        BMI=BMI,
        query=query
    )
    
    return await enhanced_consultation(request)

@app.get("/conversation-history/{user_id}")
async def get_conversation_history(user_id: str, session_id: str = None):
    """Get conversation history for a user"""
    
    try:
        if not enhanced_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        history = await enhanced_system.get_conversation_history(user_id, session_id)
        
        return {
            "user_id": user_id,
            "session_id": session_id,
            "history": history,
            "total_conversations": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reasoning-analysis")
async def reasoning_analysis(query: str, context: Dict = None):
    """Standalone reasoning analysis endpoint"""
    
    try:
        if not enhanced_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        request = HealthConsultationRequest(
            Age=30,
            SystolicBP=120,
            DiastolicBP=80,
            BloodSugar=100,
            BodyTemp=98.6,
            HeartRate=75,
            BMI=23,
            query=query
        )
        
        reasoning_result = await enhanced_system.process_reasoning(request)
        
        return {
            "query": query,
            "reasoning_analysis": reasoning_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in reasoning analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    system_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "enhanced_system": enhanced_system is not None,
            "ml_models": enhanced_system.ml_models is not None if enhanced_system else False
        }
    }
    
    return system_status

@app.get("/")
async def root():
    """Root endpoint with API information"""
    
    return {
        "message": "Enhanced Health Consultation API",
        "version": "2.0.0",
        "features": [
            "Advanced reasoning capabilities",
            "Multi-modal analysis",
            "Contextual understanding",
            "Safety checking",
            "Educational content generation",
            "Personalized recommendations",
            "Conversational chat interface"
        ],
        "endpoints": {
            "enhanced_consultation": "/enhanced-consultation",
            "chat": "/chat",
            "quick_assessment": "/quick-assessment",
            "reasoning_analysis": "/reasoning-analysis",
            "conversation_history": "/conversation-history/{user_id}",
            "health_check": "/health"
        }
    }

# Demo and Testing Functions
async def demo_enhanced_system():
    """Demonstrate the enhanced system capabilities"""
    
    print("\n" + "="*80)
    print("ENHANCED HEALTH CONSULTATION SYSTEM DEMO")
    print("="*80)
    
    system = EnhancedHealthConsultationSystem()
    
    test_cases = [
        {
            'name': 'Simple Health Check',
            'request': HealthConsultationRequest(
                Age=28,
                SystolicBP=118,
                DiastolicBP=76,
                BloodSugar=95,
                BodyTemp=98.6,
                HeartRate=72,
                BMI=22.5,
                query="I feel fine but want to check my overall health status",
                user_id="demo_user_1"
            )
        },
        {
            'name': 'Complex Pregnancy Case',
            'request': HealthConsultationRequest(
                Age=35,
                GestationalWeek=32,
                SystolicBP=145,
                DiastolicBP=92,
                BloodSugar=155,
                BodyTemp=99.1,
                HeartRate=95,
                BMI=28.5,
                PreviousPregnancies=1,
                WeightGain=38,
                query="I'm concerned about my blood pressure and blood sugar during pregnancy. I've been feeling more tired lately and sometimes get headaches.",
                user_id="demo_user_2",
                symptoms=["fatigue", "headaches", "elevated BP"],
                medical_history=["gestational diabetes in previous pregnancy"]
            )
        },
        {
            'name': 'High-Risk Cardiovascular',
            'request': HealthConsultationRequest(
                Age=52,
                SystolicBP=168,
                DiastolicBP=98,
                BloodSugar=185,
                BodyTemp=98.8,
                HeartRate=88,
                BMI=32.1,
                query="My doctor said I need to monitor my blood pressure and diabetes better. What should I be worried about and what can I do?",
                user_id="demo_user_3",
                medical_history=["type 2 diabetes", "hypertension", "family history of heart disease"],
                current_medications=["metformin", "lisinopril"]
            )
        },
        {
            'name': 'Chat Health Tips',
            'request': ChatRequest(
                query="What are some tips for managing stress during pregnancy?",
                user_id="demo_user_4"
            )
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i}: {case['name']} {'='*20}")
        
        if isinstance(case['request'], HealthConsultationRequest):
            result = await system.process_consultation(case['request'])
            
            print(f"\nUser Query: {case['request'].query}")
            print(f"\nMain Response: {result['main_response']}")
            
            print(f"\nRisk Assessment:")
            risk = result['risk_assessment']
            print(f"  ML Risk Level: {risk['ml_risk_level']} (confidence: {risk['ml_confidence']:.1%})")
            print(f"  Urgency Level: {risk['urgency_level']}")
            print(f"  Combined Risk Score: {risk['combined_risk_score']:.2f}")
            
            if result['clinical_insights']:
                print(f"\nClinical Insights:")
                for insight in result['clinical_insights']:
                    print(f"  {insight}")
            
            if result['safety_alerts']:
                print(f"\nSafety Alerts:")
                for alert in result['safety_alerts']:
                    print(f"  [{alert['level'].upper()}] {alert['message']}")
            
            print(f"\nHealth Tips:")
            for tip in result['educational_content']['health_tips'][:3]:
                print(f"  • {tip}")
            
            print(f"\nRecommendations:")
            for rec in result['recommendations'][:3]:
                print(f"  • {rec['action']} ({rec['category']}) - {rec['importance']} priority")
            
            print(f"\nFollow-up Plan:")
            follow_up = result['follow_up_plan']
            if follow_up['immediate_actions']:
                print(f"  Immediate: {', '.join(follow_up['immediate_actions'])}")
            print(f"  Next consultation: {follow_up['next_consultation']['timeframe']}")
            
            print(f"\nOverall Confidence: {result['confidence_metrics']['overall_confidence']:.1%}")
        
        else:  # ChatRequest
            result = await system.process_chat(case['request'])
            
            print(f"\nUser Query: {case['request'].query}")
            print(f"\nMain Response: {result['main_response']}")
            
            print(f"\nHealth Tips:")
            for tip in result['health_tips']:
                print(f"  • {tip}")
            
            print(f"\nLifestyle Recommendations:")
            for rec in result['lifestyle_recommendations'][:3]:
                print(f"  • {rec}")
            
            print(f"\nOverall Confidence: {result['confidence']:.1%}")
        
        print("\n" + "-"*80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        asyncio.run(demo_enhanced_system())
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)