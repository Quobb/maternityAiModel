import tensorflow as tf
import numpy as np
import pandas as pd
import json
import pickle
import re
import spacy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import random
from datetime import datetime, timedelta
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from textblob import TextBlob
from imblearn.over_sampling import SMOTE


warnings.filterwarnings('ignore')
class AdvancedMaternalHealthAI:
    """
    Integrated Advanced Maternal Health AI with human-like conversation capabilities.
    Combines comprehensive maternal health analysis with enhanced, empathetic chat features.
    """
   
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.chat_model = None
        self.intents = None
        self.conversation_memory = {}
        self.personality_traits = {
            'empathy_level': 0.8,
            'formality_level': 0.4, # 0 = very casual, 1 = very formal
            'supportiveness': 0.9,
            'medical_precision': 0.7
        }
       
        # Load spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy not available. Using basic NLP processing.")
            self.nlp = None
   
    def generate_comprehensive_maternal_dataset(self):
        """
        Enhanced dataset generation with more realistic patterns (from HumanLike model)
        """
        np.random.seed(42)
        n_samples = 12000 # Increased sample size
       
        data = {}
       
        # Demographics with more realistic distributions
        age_distribution = np.random.beta(2, 3, n_samples) * 30 + 15 # More realistic age distribution
        data['age'] = age_distribution.clip(15, 45)
       
        data['education_level'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.08, 0.15, 0.35, 0.27, 0.15])
        data['income_level'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.25, 0.35, 0.28, 0.12])
        data['marital_status'] = np.random.choice([0, 1], n_samples, p=[0.15, 0.85])
        data['employment'] = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
       
        # Medical history with age correlations
        age_factor = (data['age'] - 25) / 20 # Normalize age factor
        data['previous_pregnancies'] = np.random.poisson(np.maximum(0, age_factor), n_samples).clip(0, 8)
        data['previous_miscarriages'] = np.random.binomial(2, 0.10 + 0.05 * age_factor, n_samples).clip(0, 3)
       
        # Health conditions with realistic correlations
        diabetes_risk = 0.08 + 0.15 * (data['age'] > 35).astype(int)
        data['diabetes_history'] = np.random.binomial(1, diabetes_risk, n_samples)
       
        hypertension_risk = 0.12 + 0.20 * (data['age'] > 35).astype(int)
        data['hypertension_history'] = np.random.binomial(1, hypertension_risk, n_samples)
       
        data['heart_disease'] = np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
        data['kidney_disease'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        data['autoimmune_disorders'] = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
        data['mental_health_history'] = np.random.choice([0, 1], n_samples, p=[0.72, 0.28])
       
        # Current pregnancy with gestational age effects
        data['gestational_age'] = np.random.uniform(6, 40, n_samples)
        data['weight_pre_pregnancy'] = np.random.normal(68, 18, n_samples).clip(45, 130)
        data['height'] = np.random.normal(163, 8, n_samples).clip(140, 185)
        data['bmi_pre_pregnancy'] = data['weight_pre_pregnancy'] / ((data['height']/100) ** 2)
       
        # Weight gain based on gestational age and BMI
        expected_gain = (data['gestational_age'] / 40) * 12 # Base weight gain
        bmi_adjustment = np.where(data['bmi_pre_pregnancy'] < 18.5, 1.5,
                                 np.where(data['bmi_pre_pregnancy'] > 25, 0.7, 1.0))
        data['weight_gain'] = (expected_gain * bmi_adjustment +
                              np.random.normal(0, 3, n_samples)).clip(-2, 25)
       
        # Vital signs with pregnancy-related changes
        gestational_bp_increase = (data['gestational_age'] / 40) * 10
        data['systolic_bp'] = (np.random.normal(115, 18, n_samples) +
                              gestational_bp_increase +
                              data['hypertension_history'] * 20).clip(85, 170)
        data['diastolic_bp'] = (np.random.normal(75, 12, n_samples) +
                               gestational_bp_increase * 0.6 +
                               data['hypertension_history'] * 15).clip(55, 110)
       
        # Heart rate increases during pregnancy
        hr_increase = (data['gestational_age'] / 40) * 15
        data['heart_rate'] = (np.random.normal(72, 12, n_samples) + hr_increase).clip(50, 110)
       
        # Lab values with pregnancy-related changes
        data['hemoglobin'] = np.random.normal(11.8 - (data['gestational_age']/40) * 1.5, 1.5, n_samples).clip(7, 16)
       
        glucose_base = 88 + data['diabetes_history'] * 25
        data['glucose_fasting'] = np.random.normal(glucose_base, 15, n_samples).clip(65, 160)
       
        data['protein_urine'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.75, 0.18, 0.05, 0.02])
        data['white_blood_cells'] = np.random.normal(9000 + (data['gestational_age']/40) * 2000, 2500, n_samples).clip(4000, 16000)
        data['platelets'] = np.random.normal(230000 - (data['gestational_age']/40) * 30000, 45000, n_samples).clip(120000, 400000)
       
        # Lifestyle factors with realistic correlations
        smoking_risk = 0.15 - 0.05 * (data['education_level'] / 5)
        data['smoking'] = np.random.binomial(1, smoking_risk, n_samples)
       
        alcohol_risk = 0.08 - 0.03 * (data['education_level'] / 5)
        data['alcohol'] = np.random.binomial(1, alcohol_risk, n_samples)
       
        data['drug_use'] = np.random.choice([0, 1], n_samples, p=[0.96, 0.04])
        data['exercise_level'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.25, 0.35, 0.30, 0.10])
       
        stress_base = 2.5 + 0.5 * (data['income_level'] == 1).astype(int)
        data['stress_level'] = np.random.poisson(stress_base, n_samples).clip(1, 5)
       
        sleep_reduction = (data['gestational_age'] / 40) * 1.5
        data['sleep_hours'] = np.random.normal(7.5 - sleep_reduction, 1.2, n_samples).clip(4, 10)
       
        # Nutritional status
        data['vitamin_d'] = np.random.normal(28 + 5 * (data['prenatal_vitamins'] if 'prenatal_vitamins' in data else 1), 12, n_samples).clip(8, 70)
        data['iron_levels'] = np.random.normal(14 - (data['gestational_age']/40) * 3, 4, n_samples).clip(6, 25)
       
        supplement_likelihood = 0.8 + 0.15 * (data['education_level'] / 5)
        data['folic_acid_intake'] = np.random.binomial(1, supplement_likelihood, n_samples)
        data['prenatal_vitamins'] = np.random.binomial(1, supplement_likelihood + 0.05, n_samples)
       
        # Social determinants
        healthcare_access = np.minimum(4, np.maximum(1, data['income_level'] + np.random.normal(0, 0.5, n_samples)))
        data['access_to_healthcare'] = healthcare_access.astype(int)
       
        data['social_support'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.08, 0.17, 0.50, 0.25])
        data['transportation_access'] = np.random.choice([0, 1], n_samples, p=[0.15, 0.85])
        data['insurance_coverage'] = np.random.choice([0, 1], n_samples, p=[0.12, 0.88])
       
        # Environmental factors
        data['air_quality_index'] = np.random.gamma(2, 25, n_samples).clip(15, 180)
        data['water_quality'] = np.random.choice([1, 2, 3], n_samples, p=[0.75, 0.20, 0.05])
        data['housing_quality'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.08, 0.15, 0.55, 0.22])
       
        # Fetal measurements
        fetal_growth_factor = np.random.normal(1, 0.15, n_samples)
        data['fetal_weight_percentile'] = (50 * fetal_growth_factor).clip(5, 95)
        data['amniotic_fluid_level'] = np.random.choice([1, 2, 3], n_samples, p=[0.12, 0.76, 0.12])
        data['placental_position'] = np.random.choice([1, 2, 3], n_samples, p=[0.04, 0.93, 0.03])
       
        # Calculate risk factors with more sophisticated logic
        risk_score = (
            ((data['age'] > 35) | (data['age'] < 18)).astype(int) * 0.3 +
            (data['bmi_pre_pregnancy'] > 30).astype(int) * 0.25 +
            (data['bmi_pre_pregnancy'] < 18.5).astype(int) * 0.2 +
            data['diabetes_history'] * 0.4 +
            data['hypertension_history'] * 0.35 +
            (data['systolic_bp'] > 140).astype(int) * 0.3 +
            data['smoking'] * 0.3 +
            (data['hemoglobin'] < 10).astype(int) * 0.25 +
            (data['protein_urine'] > 1).astype(int) * 0.35 +
            (data['stress_level'] > 3).astype(int) * 0.2
        )
       
        # Target variables with more nuanced thresholds
        data['risk_level'] = np.where(risk_score < 0.4, 'Low',
                                     np.where(risk_score < 0.8, 'Medium',
                                            np.where(risk_score < 1.2, 'High', 'Critical')))
       
        # Specific conditions with realistic probabilities
        gd_risk = ((data['glucose_fasting'] > 100) * 0.6 +
                   (data['bmi_pre_pregnancy'] > 25) * 0.3 +
                   (data['age'] > 30) * 0.2 +
                   data['diabetes_history'] * 0.8 +
                   np.random.random(n_samples) * 0.3)
        data['gestational_diabetes'] = (gd_risk > 0.7).astype(int)
       
        pe_risk = ((data['systolic_bp'] > 130) * 0.5 +
                   (data['protein_urine'] > 0) * 0.4 +
                   (data['age'] > 35) * 0.3 +
                   data['hypertension_history'] * 0.6 +
                   np.random.random(n_samples) * 0.3)
        data['preeclampsia'] = (pe_risk > 0.8).astype(int)
       
        ptb_risk = ((data['previous_miscarriages'] > 0) * 0.4 +
                    data['smoking'] * 0.5 +
                    (data['stress_level'] > 3) * 0.3 +
                    (data['bmi_pre_pregnancy'] < 18.5) * 0.2 +
                    np.random.random(n_samples) * 0.4)
        data['preterm_birth_risk'] = (ptb_risk > 0.6).astype(int)
       
        ppd_risk = (data['mental_health_history'] * 0.6 +
                    (data['social_support'] < 2) * 0.4 +
                    (data['stress_level'] > 3) * 0.3 +
                    np.random.random(n_samples) * 0.4)
        data['postpartum_depression_risk'] = (ppd_risk > 0.5).astype(int)
       
        cs_risk = ((data['age'] > 35) * 0.3 +
                   (data['bmi_pre_pregnancy'] > 30) * 0.3 +
                   (data['fetal_weight_percentile'] > 85) * 0.4 +
                   np.random.random(n_samples) * 0.4)
        data['cesarean_risk'] = (cs_risk > 0.6).astype(int)
       
        data['birth_weight_category'] = np.where(data['fetal_weight_percentile'] < 25, 'Low',
                                       np.where(data['fetal_weight_percentile'] > 75, 'High', 'Normal'))
       
        return pd.DataFrame(data)
   
    def generate_comprehensive_training_intents(self):
        """
        Merged intents from both models for comprehensive and human-like coverage
        """
        # Human-like intents
        human_like_intents = {
            "casual_pregnancy_chat": {
                "patterns": [
                    "how are you feeling", "how's the pregnancy going", "how are things",
                    "what's up", "how's baby", "everything okay", "feeling alright",
                    "how's it going", "sup", "wassup", "hey there", "how you doing"
                ],
                "responses": [
                    "I'm here to help with whatever's on your mind about your pregnancy! How are you feeling today?",
                    "Every pregnancy journey is unique. What's been on your mind lately?",
                    "I hope you're doing well! Is there anything about your pregnancy you'd like to talk about?"
                ]
            },
            "expressing_emotions": {
                "patterns": [
                    "I'm scared", "I'm worried", "I'm excited", "I'm nervous",
                    "feeling anxious", "so happy", "terrified", "overwhelmed",
                    "can't sleep", "stressed out", "feeling down", "mood swings",
                    "crying a lot", "emotional rollercoaster", "hormones crazy"
                ],
                "responses": [
                    "It sounds like you're going through a lot emotionally right now, and that's completely normal during pregnancy. Your feelings are valid. What's been weighing on your mind the most?",
                    "Pregnancy brings such a mix of emotions - excitement, worry, joy, fear - sometimes all at once! What you're feeling is so common. Would you like to talk about what's making you feel this way?",
                    "I hear you, and I want you to know that what you're experiencing emotionally is part of many women's pregnancy journeys. You're not alone in this. Can you tell me more about how you've been feeling?"
                ]
            },
            "simple_symptoms": {
                "patterns": [
                    "feel sick", "throwing up", "can't keep food down", "nauseous",
                    "back hurts", "tired all the time", "can't sleep", "headache",
                    "heartburn", "swollen feet", "need to pee constantly",
                    "breasts hurt", "round ligament pain", "hip pain"
                ],
                "responses": [
                    "That sounds uncomfortable. Many pregnant women experience similar symptoms. When did you first notice this, and how has it been affecting your daily routine?",
                    "I understand how frustrating these symptoms can be. Let's talk about what you're experiencing - every detail can help us figure out the best way to help you feel better.",
                    "Pregnancy symptoms can really impact how you're feeling day to day. You're dealing with a lot right now. Can you describe what's been bothering you the most?"
                ]
            },
            "relationship_concerns": {
                "patterns": [
                    "partner doesn't understand", "husband not supportive",
                    "relationship changed", "intimacy issues", "partner stressed",
                    "don't feel connected", "arguing more", "he doesn't get it",
                    "feels distant", "relationship problems", "communication issues"
                ],
                "responses": [
                    "Pregnancy can put stress on relationships, and it sounds like you're feeling unsupported right now. That must be really hard to deal with on top of everything else. What's been the most challenging part?",
                    "Relationship changes during pregnancy are more common than people talk about. You deserve to feel supported and understood. What would help you feel more connected with your partner?",
                    "It's tough when you feel like your partner isn't on the same page during such an important time. Your feelings matter, and it's okay to need more support. Have you been able to talk about how you're feeling?"
                ]
            },
            "practical_life_concerns": {
                "patterns": [
                    "work stress", "boss not understanding", "maternity leave",
                    "money worries", "can't afford", "insurance problems",
                    "childcare concerns", "preparing nursery", "baby stuff expensive",
                    "time management", "too busy", "don't have time"
                ],
                "responses": [
                    "Balancing pregnancy with work and life responsibilities is really challenging. It sounds like you're juggling a lot right now. What's feeling most overwhelming to you?",
                    "The practical side of preparing for a baby can feel daunting, especially when you're dealing with work stress too. You're handling a lot. What's your biggest concern right now?",
                    "It's completely normal to feel stressed about the practical aspects of having a baby. These are real concerns that many expectant parents face. Let's talk about what's worrying you most."
                ]
            },
            "body_changes_concerns": {
                "patterns": [
                    "don't recognize my body", "feel huge", "stretch marks",
                    "weight gain worries", "don't feel attractive", "clothes don't fit",
                    "body changing so fast", "self conscious", "feel fat",
                    "partner finds me attractive", "body image issues"
                ],
                "responses": [
                    "Your body is doing something incredible right now, but I understand that all these changes can feel overwhelming. It's normal to have mixed feelings about how your body is changing. How are you feeling about these changes?",
                    "Pregnancy brings such dramatic body changes, and it's completely understandable to feel uncertain about them. Your feelings about your changing body are valid. What's been the hardest part for you?",
                    "Many women struggle with body image during pregnancy - you're definitely not alone in feeling this way. Your body is working so hard to grow your baby. What would help you feel more comfortable with these changes?"
                ]
            },
            "food_and_cravings": {
                "patterns": [
                    "weird cravings", "can't stop eating", "craving ice cream",
                    "want pickles", "food aversions", "everything tastes weird",
                    "gained too much weight", "eating too much", "food guilt",
                    "healthy eating hard", "junk food cravings"
                ],
                "responses": [
                    "Pregnancy cravings and food changes are fascinating and can be so strong! It sounds like your relationship with food has really shifted. What kinds of things have you been craving or avoiding?",
                    "Food during pregnancy can be such a rollercoaster - suddenly loving things you used to hate, or not being able to stand your favorite foods. What's been the strangest change for you?",
                    "Your body is telling you what it needs in some pretty interesting ways! Don't feel guilty about cravings - they're part of the experience. What food changes have surprised you the most?"
                ]
            },
            "sleep_and_energy": {
                "patterns": [
                    "can't get comfortable", "up all night peeing", "insomnia",
                    "exhausted but can't sleep", "weird dreams", "nightmares",
                    "tossing and turning", "partner snoring", "too hot to sleep",
                    "tired all day", "no energy", "need naps"
                ],
                "responses": [
                    "Sleep during pregnancy can be so elusive! It's frustrating when you're exhausted but can't get comfortable. What's been making it hardest for you to get good rest?",
                    "The pregnancy sleep struggle is real - your body needs rest, but so many things make it difficult to actually sleep well. What's been keeping you up most?",
                    "Sleep issues during pregnancy are incredibly common, but that doesn't make it any less exhausting for you. What time of night tends to be most difficult?"
                ]
            },
            "birth_fears": {
                "patterns": [
                    "scared of labor", "pain worries", "what if something goes wrong",
                    "fear of childbirth", "epidural or natural", "cesarean scary",
                    "don't know what to expect", "birth plan confusion",
                    "horror stories", "traumatic birth fears"
                ],
                "responses": [
                    "It's completely natural to feel scared about labor and delivery - you're facing something unknown and intense. Many women share these fears. What specifically worries you the most about giving birth?",
                    "Birth fears are so common and completely understandable. Knowledge and preparation can help, but it's okay to still feel nervous. What aspects of labor and delivery feel most scary to you?",
                    "The unknown aspects of childbirth can feel overwhelming, especially with all the stories you hear. Your fears are valid. What would help you feel more prepared or confident about the birth?"
                ]
            },
            "acknowledgments": {
                "patterns": [
                    "okay", "thanks", "got it", "makes sense", "that helps",
                    "feel better", "good to know", "appreciate it", "understood",
                    "yeah", "right", "exactly", "true", "I see"
                ],
                "responses": [
                    "I'm glad that resonated with you. Is there anything else about your pregnancy journey you'd like to talk about?",
                    "You're so welcome. How else can I support you today?",
                    "I'm here whenever you need to talk more about anything pregnancy-related. How are you feeling overall?"
                ]
            }
        }
       
        # Comprehensive intents from Advanced model
        comprehensive_intents = {
            "pregnancy_stages": {
                "patterns": [
                    "first trimester", "second trimester", "third trimester", "early pregnancy",
                    "late pregnancy", "pregnancy stages", "what to expect", "pregnancy timeline",
                    "pregnancy milestones", "fetal development stages", "pregnancy progression"
                ],
                "responses": [
                    "Pregnancy has three trimesters, each with unique developments and changes. Which trimester are you in or curious about?",
                    "Each stage of pregnancy brings new developments for both you and baby. The first trimester focuses on organ formation, second on growth and movement, and third on final development and birth preparation.",
                    "Pregnancy progression varies for everyone, but there are general milestones. What specific stage would you like to know about?"
                ]
            },
            "nutrition_comprehensive": {
                "patterns": [
                    "nutrition plan", "meal planning", "supplements", "prenatal vitamins", "folic acid",
                    "iron deficiency", "vitamin d", "calcium", "omega 3", "dha", "protein needs",
                    "calorie requirements", "weight gain", "healthy recipes", "food safety",
                    "listeria", "toxoplasmosis", "mercury in fish", "caffeine limit", "alcohol pregnancy"
                ],
                "responses": [
                    "Proper nutrition is crucial for both you and baby's health. A balanced diet with key nutrients like folate, iron, calcium, and DHA supports healthy development. What specific nutrition questions do you have?",
                    "Prenatal nutrition involves getting adequate calories, protein, vitamins, and minerals. I can help you understand requirements for your stage of pregnancy and any dietary restrictions.",
                    "Food safety during pregnancy is important to prevent infections. This includes avoiding certain foods and proper food handling. Would you like specific guidance?"
                ]
            },
            "complications_conditions": {
                "patterns": [
                    "gestational diabetes", "preeclampsia", "placenta previa", "preterm labor",
                    "miscarriage", "ectopic pregnancy", "hyperemesis gravidarum", "anemia",
                    "blood pressure", "swelling", "protein in urine", "bleeding", "contractions",
                    "cervical insufficiency", "placental abruption", "intrauterine growth restriction"
                ],
                "responses": [
                    "Pregnancy complications can be concerning, but many are manageable with proper care. It's important to work closely with your healthcare provider for monitoring and treatment. Which condition are you concerned about?",
                    "While complications can occur, early detection and management often lead to good outcomes. Regular prenatal care helps identify and address issues early. What symptoms or conditions are you experiencing?",
                    "Each pregnancy complication has specific signs, risks, and treatments. I can provide information, but medical evaluation is essential for proper diagnosis and care."
                ]
            },
            "mental_health_comprehensive": {
                "patterns": [
                    "anxiety", "depression", "mood swings", "emotional changes", "stress management",
                    "postpartum depression", "baby blues", "prenatal depression", "counseling",
                    "support groups", "coping strategies", "overwhelmed", "scared", "worried",
                    "bonding with baby", "body image", "relationship changes", "fear of labor"
                ],
                "responses": [
                    "Mental health is just as important as physical health during pregnancy. It's normal to experience various emotions, but persistent feelings need attention. How are you feeling emotionally?",
                    "Pregnancy can bring many emotional changes due to hormones, life changes, and concerns. Professional support is available and can be very helpful. What specific feelings are you experiencing?",
                    "Taking care of your emotional wellbeing benefits both you and your baby. There are many resources and strategies available. Would you like to discuss specific concerns or coping methods?"
                ]
            },
            "labor_delivery_comprehensive": {
                "patterns": [
                    "labor signs", "birth plan", "delivery options", "natural birth", "epidural",
                    "cesarean section", "water birth", "home birth", "hospital birth", "midwife",
                    "doula", "labor positions", "breathing techniques", "pain management",
                    "inducing labor", "overdue", "early labor", "active labor", "transition"
                ],
                "responses": [
                    "Preparing for labor and delivery involves understanding your options and creating a birth plan that works for you. What aspects of birth preparation interest you most?",
                    "There are many approaches to labor and delivery, from natural methods to medical interventions. The key is being informed about your choices. What would you like to know about?",
                    "Birth preparation includes physical, emotional, and practical aspects. From labor positions to pain management options, there's much to consider. How can I help you prepare?"
                ]
            },
            "postpartum_comprehensive": {
                "patterns": [
                    "postpartum recovery", "after birth", "healing", "breastfeeding", "bottle feeding",
                    "newborn care", "sleep deprivation", "postpartum bleeding", "episiotomy care",
                    "c-section recovery", "returning to work", "contraception", "future pregnancies",
                    "postpartum checkup", "baby blues", "postpartum depression"
                ],
                "responses": [
                    "Postpartum recovery is an important time for both physical healing and emotional adjustment. Recovery looks different for everyone. What aspects of postpartum care are you concerned about?",
                    "The postpartum period involves healing, learning to care for your baby, and adjusting to new routines. Support and information are key. How can I help you prepare for or navigate this time?",
                    "Postpartum care includes physical recovery, emotional wellbeing, infant care, and family adjustment. It's a significant transition that deserves attention and support."
                ]
            },
            "high_risk_pregnancy": {
                "patterns": [
                    "high risk pregnancy", "advanced maternal age", "multiple pregnancy", "twins",
                    "triplets", "chronic conditions", "diabetes", "hypertension", "autoimmune",
                    "previous complications", "fertility treatments", "bed rest", "specialist care",
                    "maternal fetal medicine", "genetic testing", "amniocentesis"
                ],
                "responses": [
                    "High-risk pregnancies require specialized care and monitoring, but many result in healthy outcomes. Working with maternal-fetal medicine specialists can provide the best care. What makes your pregnancy high-risk?",
                    "While a high-risk designation can be concerning, it simply means you need extra monitoring and care. Many factors can contribute to this classification. How can I help you understand your specific situation?",
                    "High-risk pregnancies benefit from individualized care plans and close monitoring. The goal is to optimize outcomes for both mother and baby through specialized attention."
                ]
            },
            "lifestyle_wellness": {
                "patterns": [
                    "exercise pregnancy", "prenatal yoga", "swimming", "walking", "travel",
                    "work during pregnancy", "sleep", "sexual health", "beauty treatments",
                    "hair dye", "massage", "spa treatments", "stress reduction", "meditation",
                    "relaxation techniques", "partner support", "family preparation"
                ],
                "responses": [
                    "Maintaining a healthy lifestyle during pregnancy supports your wellbeing and baby's development. This includes safe exercise, good sleep, stress management, and self-care. What lifestyle aspects interest you?",
                    "Pregnancy doesn't mean putting life on hold, but some modifications may be needed for safety and comfort. I can help you understand what's safe and beneficial during pregnancy.",
                    "Wellness during pregnancy encompasses physical activity, mental health, relationships, and self-care. Creating a balanced approach supports your journey. What would you like to focus on?"
                ]
            },
            "emergency_situations": {
                "patterns": [
                    "emergency", "severe bleeding", "severe pain", "vision changes", "severe headache",
                    "difficulty breathing", "chest pain", "fever", "decreased fetal movement",
                    "water breaking", "contractions early", "call doctor", "go to hospital",
                    "urgent care", "when to worry", "warning signs"
                ],
                "responses": [
                    "⚠️ This sounds like it could require immediate medical attention. Please contact your healthcare provider right away or go to the emergency room if you're experiencing severe symptoms like heavy bleeding, severe pain, vision changes, or difficulty breathing.",
                    "⚠️ Some pregnancy symptoms require urgent medical care. If you're experiencing severe or concerning symptoms, don't hesitate to seek immediate medical attention. Trust your instincts - it's better to be safe.",
                    "⚠️ Emergency warning signs in pregnancy include: severe bleeding, severe headaches, vision problems, severe abdominal pain, fever over 100.4°F, or significant decrease in baby's movements. Seek immediate care for these symptoms."
                ]
            },
            "family_partner_support": {
                "patterns": [
                    "partner support", "family involvement", "preparing siblings", "grandparents",
                    "relationship changes", "communication", "intimacy", "support system",
                    "involving father", "single pregnancy", "family dynamics", "cultural considerations",
                    "extended family", "preparing children", "family planning"
                ],
                "responses": [
                    "Having a strong support system during pregnancy and after birth is incredibly valuable. This can include partners, family, friends, and community resources. How is your support system?",
                    "Pregnancy affects the whole family, and preparing everyone can help with the transition. Communication and involvement are key. What aspects of family support would you like to discuss?",
                    "Building and maintaining supportive relationships during pregnancy benefits everyone involved. This includes educating partners and family about pregnancy and preparing for the new baby."
                ]
            }
        }
       
        # Merge both
        merged_intents = {**human_like_intents, **comprehensive_intents}
        return merged_intents
   
    def enhance_text_preprocessing(self, text):
        """
        Enhanced text preprocessing with emotion detection, context understanding, and abbreviations expansion
        """
        # Convert to lowercase and strip
        text = text.lower().strip()
       
        # Expand contractions (from HumanLike)
        contractions = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "i'm": "i am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
       
        # Expand abbreviations (from Advanced)
        abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'bpm': 'beats per minute',
            'c-section': 'cesarean section',
            'ob': 'obstetrician',
            'gyn': 'gynecologist',
            'ob-gyn': 'obstetrician gynecologist',
            'nicu': 'neonatal intensive care unit',
            'iugr': 'intrauterine growth restriction',
            'ppd': 'postpartum depression',
            'gd': 'gestational diabetes',
            'hcg': 'human chorionic gonadotropin'
        }
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
       
        # Detect emotional indicators
        emotion_markers = {
            'anxiety': ['worried', 'scared', 'anxious', 'nervous', 'terrified', 'panic'],
            'sadness': ['sad', 'depressed', 'down', 'crying', 'upset', 'heartbroken'],
            'excitement': ['excited', 'happy', 'thrilled', 'amazing', 'wonderful'],
            'confusion': ['confused', 'unsure', 'dont know', 'not sure', 'unclear'],
            'pain': ['hurt', 'pain', 'ache', 'sore', 'uncomfortable', 'burning']
        }
       
        detected_emotions = []
        for emotion, markers in emotion_markers.items():
            if any(marker in text for marker in markers):
                detected_emotions.append(emotion)
       
        # Use spaCy for better preprocessing if available
        entities = []
        if self.nlp:
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents]
       
        return {
            'processed_text': text,
            'emotions': detected_emotions,
            'entities': entities
        }
   
    def generate_empathetic_response(self, base_response, user_emotions, conversation_context=None):
        """
        Generate more empathetic and human-like responses
        """
        empathy_prefixes = {
            'anxiety': [
                "I can hear that you're feeling anxious about this, and that's completely understandable.",
                "It sounds like you're really worried, and I want you to know that your concerns are valid.",
                "I can sense you're feeling nervous about this, which is so normal."
            ],
            'sadness': [
                "I can tell you're going through a difficult time right now.",
                "It sounds like you're feeling really down about this, and I'm sorry you're struggling.",
                "I hear the sadness in what you're sharing, and I want you to know I'm here to support you."
            ],
            'excitement': [
                "I love hearing the excitement in your message!",
                "It's wonderful to hear how happy you're feeling about this!",
                "Your enthusiasm is contagious!"
            ],
            'confusion': [
                "I can tell you're feeling uncertain about this, which is completely normal.",
                "It sounds like you're looking for some clarity, and I'm here to help.",
                "Pregnancy can be confusing with so much information out there."
            ],
            'pain': [
                "I'm sorry you're dealing with discomfort right now.",
                "That sounds really uncomfortable, and I want to help you feel better.",
                "Physical symptoms during pregnancy can be so challenging to deal with."
            ]
        }
       
        # Add empathetic prefix based on detected emotions
        if user_emotions:
            primary_emotion = user_emotions[0] # Take the first detected emotion
            if primary_emotion in empathy_prefixes:
                empathy_prefix = random.choice(empathy_prefixes[primary_emotion])
                base_response = f"{empathy_prefix} {base_response}"
       
        # Add conversational connectors
        connectors = [
            "Let me help you with that.",
            "Here's what I'm thinking:",
            "From what you're describing,",
            "Based on what you've shared,",
            "It sounds like"
        ]
       
        # Sometimes add a connector for more natural flow
        if random.random() < 0.3: # 30% chance
            connector = random.choice(connectors)
            base_response = f"{connector.lower()} {base_response}"
       
        return base_response
   
    def maintain_conversation_context(self, user_id, user_input, response, emotions):
        """
        Maintain conversation context and memory for more human-like interactions
        """
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = {
                'history': [],
                'topics_discussed': set(),
                'emotional_state': [],
                'concerns': [],
                'gestational_stage': None,
                'last_interaction': None
            }
       
        # Update conversation memory
        memory = self.conversation_memory[user_id]
        memory['history'].append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'response': response,
            'emotions': emotions
        })
       
        # Keep only last 10 interactions to manage memory
        memory['history'] = memory['history'][-10:]
       
        # Track emotional patterns
        memory['emotional_state'].extend(emotions)
        memory['emotional_state'] = memory['emotional_state'][-20:] # Keep recent emotions
       
        # Update last interaction
        memory['last_interaction'] = datetime.now()
   
    def get_contextual_followup_questions(self, intent, user_emotions, conversation_history):
        """
        Generate contextual follow-up questions to keep conversation flowing
        """
        followup_questions = {
            'casual_pregnancy_chat': [
                "What's been the most surprising thing about your pregnancy so far?",
                "How has your partner been handling the pregnancy news?",
                "Are you finding out the baby's gender, or keeping it a surprise?"
            ],
            'expressing_emotions': [
                "Have you been able to talk to anyone else about how you're feeling?",
                "What usually helps you feel better when you're going through tough emotions?",
                "Do you think these feelings are connected to any specific concerns about the pregnancy?"
            ],
            'simple_symptoms': [
                "Have you mentioned this to your doctor yet?",
                "How long have you been dealing with this?",
                "Is this interfering with your daily activities or sleep?"
            ],
            'pregnancy_stages': [
                "What symptoms are you experiencing in this stage?",
                "How can I help with preparation for the next trimester?"
            ],
            'nutrition_comprehensive': [
                "Do you have any dietary restrictions I should consider?",
                "Would you like sample meal ideas?"
            ],
            'complications_conditions': [
                "Have you discussed this with your healthcare provider?",
                "What symptoms are you currently experiencing?"
            ],
            'mental_health_comprehensive': [
                "Would you like resources for professional support?",
                "What coping strategies have you tried?"
            ],
            'labor_delivery_comprehensive': [
                "Have you started creating a birth plan?",
                "What are your preferences for pain management?"
            ],
            'postpartum_comprehensive': [
                "How are you planning for newborn care?",
                "What support do you have lined up after birth?"
            ],
            'high_risk_pregnancy': [
                "What specific monitoring are you receiving?",
                "How can I help with understanding your care plan?"
            ],
            'lifestyle_wellness': [
                "What activities are you currently doing?",
                "Any specific lifestyle changes you're considering?"
            ],
            'family_partner_support': [
                "How can I help with communicating with your partner?",
                "Would you like tips for involving family?"
            ]
        }
       
        questions = followup_questions.get(intent, [
            "What else has been on your mind?",
            "How can I best support you right now?",
            "Is there anything specific you'd like to know more about?"
        ])
       
        return random.choice(questions)
   
    def generate_personalized_response_style(self, user_profile=None):
        """
        Adjust response style based on user preferences or profile
        """
        if user_profile:
            # Adjust personality traits based on user preferences
            if user_profile.get('prefers_formal'):
                self.personality_traits['formality_level'] = 0.7
            if user_profile.get('needs_high_support'):
                self.personality_traits['supportiveness'] = 1.0
            if user_profile.get('medical_background'):
                self.personality_traits['medical_precision'] = 0.9
   
    def get_advanced_chat_response(self, user_input, user_context=None, user_id="anonymous"):
        """
        Enhanced human-like chat response integrating both models' capabilities
        """
        if not self.chat_model:
            return {
                'response': "I'm still learning how to have conversations. Please bear with me as I get better at understanding and responding to you.",
                'intent': 'system',
                'confidence': 0.0,
                'emergency': False
            }
       
        # Enhanced preprocessing with emotion detection
        processed_input = self.enhance_text_preprocessing(user_input)
        text = processed_input['processed_text']
        emotions = processed_input['emotions']
       
        # Check conversation history for context
        conversation_history = self.conversation_memory.get(user_id, {}).get('history', [])
       
        # Emergency detection with merged patterns
        emergency_patterns = [
            r'\b(severe|heavy|extreme|intense|unbearable|excruciating)\s+(bleeding|pain|headache|cramping)\b',
            r'\b(chest pain|difficulty breathing|vision changes|blurred vision)\b',
            r'\b(water broke|water breaking|contractions)\b.*\b(early|preterm|not due)\b',
            r'\b(fever|high fever|temperature)\b.*\b(pregnancy|pregnant)\b',
            r'\b(decreased|less|no|can\'t feel)\s+(fetal movement|baby movement|baby moving)\b',
            r'\b(passing out|fainting|dizzy|lightheaded)\b.*\b(severe|really|very)\b',
            r'\b(something\'s wrong|emergency|help|urgent|scared|terrified)\b.*\b(bleeding|pain|baby)\b'
        ]
       
        for pattern in emergency_patterns:
            if re.search(pattern, text):
                emergency_response = self.generate_emergency_response(emotions)
                return {
                    'intent': 'emergency_situations',
                    'response': emergency_response,
                    'confidence': 1.0,
                    'emergency': True,
                    'followup': "Please don't hesitate to reach out again if you need more support or have other concerns."
                }
       
        try:
            # Predict intent
            predicted_intent = self.chat_model.predict([text])[0]
            confidence = max(self.chat_model.predict_proba([text])[0])
           
            # Get base response
            if predicted_intent in self.intents:
                responses = self.intents[predicted_intent]['responses']
                base_response = random.choice(responses)
               
                # Make response more empathetic and human-like
                human_response = self.generate_empathetic_response(
                    base_response, emotions, conversation_history
                )
               
                # Add contextual information if available
                if user_context:
                    human_response = self.add_contextual_information(
                        human_response, user_context, predicted_intent
                    )
               
                # Adapt to user style
                human_response = self.adapt_response_to_user_style(human_response, user_id)
               
                # Generate follow-up question
                followup = self.get_contextual_followup_questions(
                    predicted_intent, emotions, conversation_history
                )
               
                # Maintain conversation memory
                self.maintain_conversation_context(user_id, user_input, human_response, emotions)
               
                # Add personality-based variations
                if random.random() < 0.2: # 20% chance to add casual elements
                    casual_additions = [
                        " I hope that helps!",
                        " Let me know what you think.",
                        " Does that make sense?",
                        " I'm here if you need to talk more about this."
                    ]
                    human_response += random.choice(casual_additions)
               
                return {
                    'intent': predicted_intent,
                    'response': human_response,
                    'confidence': float(confidence),
                    'emergency': False,
                    'emotions_detected': emotions,
                    'followup': followup,
                    'suggestions': self.get_related_suggestions(predicted_intent)
                }
           
            else:
                # Handle unknown intents more gracefully
                fallback_responses = [
                    f"I want to make sure I understand what you're going through. Could you tell me a bit more about {' and '.join(emotions) if emotions else 'whats on your mind'}?",
                    "I'm here to help with whatever pregnancy concerns you have. Sometimes it helps to break things down - what's the main thing that's been worrying you?",
                    "Every pregnancy journey is different, and I want to give you the most helpful information. Can you help me understand what specific area you'd like support with?"
                ]
               
                response = random.choice(fallback_responses)
               
                if emotions:
                    response = self.generate_empathetic_response(response, emotions)
               
                return {
                    'intent': 'unknown',
                    'response': response,
                    'confidence': 0.0,
                    'emergency': False,
                    'emotions_detected': emotions,
                    'followup': "What's been the most challenging part of your pregnancy experience so far?"
                }
               
        except Exception as e:
            error_responses = [
                "I'm having a moment where I can't quite process what you're saying, but I'm still here to help. Could you try rephrasing that?",
                "Something got a bit jumbled on my end - could you tell me again what's on your mind?",
                "I want to make sure I give you the best response, but I need you to help me understand better. What's the main concern you have right now?"
            ]
           
            return {
                'intent': 'error',
                'response': random.choice(error_responses),
                'confidence': 0.0,
                'emergency': False,
                'emotions_detected': emotions,
                'error_details': str(e)
            }
   
    def generate_emergency_response(self, emotions):
        """
        Generate empathetic emergency responses
        """
        base_emergency = "🚨 What you're describing sounds like it could need immediate medical attention. Please contact your healthcare provider right away or go to the nearest emergency room."
       
        if 'anxiety' in emotions:
            return f"I can hear how scared you must be right now, and I want you to get the help you need immediately. {base_emergency} You're not alone in this - medical professionals are there to help you and your baby."
        elif 'pain' in emotions:
            return f"I'm so sorry you're in pain right now. {base_emergency} Don't try to tough this out - you and your baby deserve immediate care."
        else:
            return f"{base_emergency} Trust your instincts - if something feels wrong, it's always better to get checked out. Your health and your baby's health are the priority right now."
   
    def train_advanced_chat_model(self):
        """
        Train the chat model with merged human-like and comprehensive data
        """
        print("Training advanced human-like chat model...")
       
        intents_data = self.generate_comprehensive_training_intents()
       
        # Generate training examples with more natural variations
        training_texts = []
        training_labels = []
       
        # Natural conversation starters and variations
        conversation_starters = [
            "hey", "hi", "hello", "so", "well", "um", "actually",
            "i was wondering", "can you help me with", "i need to talk about",
            "i'm having trouble with", "what do you think about",
            "is it normal that", "should i be worried about"
        ]
       
        emotional_modifiers = [
            "really", "very", "so", "extremely", "quite", "pretty",
            "kinda", "sorta", "a bit", "a little", "totally"
        ]
       
        for intent, data in intents_data.items():
            for pattern in data["patterns"]:
                # Add original pattern
                training_texts.append(pattern.lower())
                training_labels.append(intent)
               
                # Add variations with conversation starters
                for starter in conversation_starters[:3]: # Use subset to avoid explosion
                    varied_pattern = f"{starter} {pattern}".lower()
                    training_texts.append(varied_pattern)
                    training_labels.append(intent)
               
                # Add emotional variations
                for modifier in emotional_modifiers[:2]:
                    if random.random() < 0.3: # 30% chance
                        emotional_pattern = pattern.replace("i'm", f"i'm {modifier}").lower()
                        training_texts.append(emotional_pattern)
                        training_labels.append(intent)
               
                # Add typo variations (common misspellings)
                typo_variations = {
                    "worried": "worreid", "scared": "scarred", "tired": "tierd",
                    "pregnant": "pregnent", "nauseous": "nauseus"
                }
               
                modified_pattern = pattern.lower()
                for correct, typo in typo_variations.items():
                    if correct in modified_pattern:
                        typo_pattern = modified_pattern.replace(correct, typo)
                        training_texts.append(typo_pattern)
                        training_labels.append(intent)
                        break # Only one typo per pattern
       
        # Create enhanced pipeline
        self.chat_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=15000,
                stop_words='english',
                sublinear_tf=True,
                min_df=1,
                lowercase=True,
                strip_accents='ascii'
            )),
            ('classifier', MultinomialNB(alpha=0.01))
        ])
       
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            training_texts, training_labels, test_size=0.2, random_state=42,
            stratify=training_labels
        )
       
        self.chat_model.fit(X_train, y_train)
       
        # Evaluate
        y_pred = self.chat_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Human-like chat model accuracy: {accuracy:.4f}")
       
        self.intents = intents_data
   
    def add_contextual_information(self, base_response, context, intent):
        """
        Merged contextual information addition
        """
        contextual_additions = {
            'expressing_emotions': {
                'first_pregnancy': " Since this is your first pregnancy, know that all these intense emotions are completely normal - you're navigating something brand new.",
                'high_risk': " I know having a high-risk pregnancy can add extra worry to what you're already feeling.",
                'previous_loss': " Given your previous experience with pregnancy loss, it's completely understandable that you might feel extra anxious."
            },
            'simple_symptoms': {
                'first_trimester': " In the first trimester, your body is going through massive changes, so symptoms like this are really common.",
                'third_trimester': " You're in the home stretch now! Third trimester symptoms can be intense as your body prepares for birth.",
                'high_risk': " Since you're having a high-risk pregnancy, it's especially important to keep track of symptoms like this."
            },
            'birth_fears': {
                'first_pregnancy': " It's so normal to feel scared about your first birth experience - you're about to do something amazing but unknown.",
                'previous_traumatic_birth': " After a difficult previous birth experience, it makes complete sense that you'd have fears about this delivery."
            },
            'pregnancy_stages': {
                'trimester_1': "Since you're in your first trimester, focus on taking prenatal vitamins, avoiding harmful substances, and managing early pregnancy symptoms.",
                'trimester_2': "Being in your second trimester, this is often the most comfortable time with increased energy and visible baby growth.",
                'trimester_3': "In your third trimester, prepare for birth, monitor baby's movements, and watch for labor signs."
            },
            'nutrition_comprehensive': {
                'gestational_diabetes': "With gestational diabetes, focus on complex carbohydrates, protein, and monitoring blood sugar levels.",
                'anemia': "For iron deficiency anemia, include iron-rich foods and consider iron supplements as recommended by your doctor."
            },
            'mental_health_comprehensive': {
                'first_pregnancy': "As a first-time mother, it's completely normal to feel anxious or overwhelmed. Many resources are available to support you.",
                'previous_loss': "Having experienced pregnancy loss before can increase anxiety. Consider counseling and extra emotional support."
            }
        }
       
        if intent in contextual_additions:
            for condition, addition in contextual_additions[intent].items():
                if condition in context:
                    base_response += addition
                    break # Only add one contextual piece per response
       
        return base_response
   
    def get_related_suggestions(self, intent):
        """
        Merged suggestions from both models
        """
        suggestions_map = {
            'casual_pregnancy_chat': [
                "Want to talk about how you're preparing for baby?",
                "Curious about what to expect in your current trimester?",
                "Need some tips for dealing with pregnancy symptoms?"
            ],
            'expressing_emotions': [
                "Would it help to talk about coping strategies for tough emotions?",
                "Want to explore what support options might work for you?",
                "Interested in learning about pregnancy mood changes?"
            ],
            'simple_symptoms': [
                "Want some tips for managing this symptom?",
                "Curious about when symptoms like this typically improve?",
                "Need help figuring out when to call your doctor?"
            ],
            'relationship_concerns': [
                "Want to talk about ways to communicate better with your partner?",
                "Interested in tips for keeping your relationship strong during pregnancy?",
                "Need ideas for helping your partner understand what you're going through?"
            ],
            'pregnancy_stages': [
                "Learn about fetal development milestones",
                "Understand common symptoms by trimester",
                "Explore prenatal testing options"
            ],
            'nutrition_comprehensive': [
                "Create a personalized meal plan",
                "Learn about food safety during pregnancy",
                "Understand supplement requirements"
            ],
            'complications_conditions': [
                "Know warning signs to watch for",
                "Understand risk factors",
                "Learn about prevention strategies"
            ],
            'mental_health_comprehensive': [
                "Explore stress management techniques",
                "Find local support groups",
                "Learn about counseling options"
            ],
            'labor_delivery_comprehensive': [
                "Create your birth plan",
                "Learn labor breathing techniques",
                "Understand pain management options"
            ],
            'postpartum_comprehensive': [
                "Plan for recovery support",
                "Learn about newborn care",
                "Understand breastfeeding basics"
            ]
        }
       
        return suggestions_map.get(intent, [
            "What else can I help you with today?",
            "Any other pregnancy questions on your mind?",
            "How else can I support you right now?"
        ])
   
    def analyze_conversation_patterns(self, user_id):
        """
        Analyze conversation patterns to provide better, more personalized responses
        """
        if user_id not in self.conversation_memory:
            return {}
       
        memory = self.conversation_memory[user_id]
        history = memory['history']
       
        if not history:
            return {}
       
        # Analyze emotional patterns
        all_emotions = []
        for interaction in history:
            all_emotions.extend(interaction.get('emotions', []))
       
        # Count emotion frequencies
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
       
        # Identify conversation themes
        common_topics = []
        for interaction in history:
            user_text = interaction['user_input'].lower()
            if any(word in user_text for word in ['sleep', 'tired', 'exhausted']):
                common_topics.append('sleep_issues')
            if any(word in user_text for word in ['scared', 'worried', 'nervous']):
                common_topics.append('anxiety')
            if any(word in user_text for word in ['partner', 'husband', 'relationship']):
                common_topics.append('relationship')
       
        return {
            'dominant_emotions': sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'common_topics': list(set(common_topics)),
            'interaction_count': len(history),
            'conversation_style': self.detect_conversation_style(history)
        }
   
    def detect_conversation_style(self, history):
        """
        Detect user's preferred conversation style
        """
        if not history:
            return 'unknown'
       
        total_words = 0
        casual_indicators = 0
        formal_indicators = 0
       
        for interaction in history:
            user_text = interaction['user_input'].lower()
            words = user_text.split()
            total_words += len(words)
           
            # Check for casual language
            casual_words = ['yeah', 'ok', 'gonna', 'wanna', 'kinda', 'sorta', 'sup', 'hey']
            casual_indicators += sum(1 for word in words if word in casual_words)
           
            # Check for formal language
            formal_words = ['however', 'therefore', 'furthermore', 'regarding', 'concerning']
            formal_indicators += sum(1 for word in words if word in formal_words)
       
        if casual_indicators > formal_indicators:
            return 'casual'
        elif formal_indicators > casual_indicators:
            return 'formal'
        else:
            return 'balanced'
   
    def adapt_response_to_user_style(self, response, user_id):
        """
        Adapt response style based on user's conversation patterns
        """
        patterns = self.analyze_conversation_patterns(user_id)
        style = patterns.get('conversation_style', 'balanced')
       
        if style == 'casual':
            # Make response more casual
            response = response.replace('It is important', 'It\'s really important')
            response = response.replace('You should', 'You might want to')
            response = response.replace('I recommend', 'I\'d suggest')
           
        elif style == 'formal':
            # Keep response more formal
            response = response.replace('It\'s', 'It is')
            response = response.replace('You\'re', 'You are')
            response = response.replace('can\'t', 'cannot')
       
        return response
   
    def train_comprehensive_models(self, df):
        """
        Enhanced model training (from HumanLike model)
        """
        print("Training comprehensive maternal health models...")
       
        feature_columns = [col for col in df.columns if col not in [
            'risk_level', 'gestational_diabetes', 'preeclampsia', 'preterm_birth_risk',
            'postpartum_depression_risk', 'cesarean_risk', 'birth_weight_category'
        ]]
       
        X = df[feature_columns]
       
        # Enhanced feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
       
        models_to_train = {
            'risk_level': (df['risk_level'], 'multiclass'),
            'gestational_diabetes': (df['gestational_diabetes'], 'binary'),
            'preeclampsia': (df['preeclampsia'], 'binary'),
            'preterm_birth_risk': (df['preterm_birth_risk'], 'binary'),
            'postpartum_depression_risk': (df['postpartum_depression_risk'], 'binary'),
            'cesarean_risk': (df['cesarean_risk'], 'binary'),
            'birth_weight_category': (df['birth_weight_category'], 'multiclass')
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
           
            # Enhanced algorithms with hyperparameter tuning
            algorithms = {
                'random_forest': RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=5,
                    class_weight='balanced',
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'svm': SVC(
                    probability=True,
                    class_weight='balanced',
                    gamma='scale',
                    random_state=42
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    max_iter=2000,
                    alpha=0.001,
                    early_stopping=True,
                    random_state=42
                ),
                'logistic_regression': LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )
            }
           
            best_model = None
            best_score = 0
            best_algorithm = None
           
            for alg_name, model in algorithms.items():
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
                    avg_score = np.mean(scores)
                   
                    print(f" {alg_name}: {avg_score:.4f} (+/- {scores.std() * 2:.4f})")
                   
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_algorithm = alg_name
                       
                except Exception as e:
                    print(f" {alg_name}: Error - {str(e)}")
           
            if best_model is not None:
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
               
                print(f" Best: {best_algorithm} (Test accuracy: {test_accuracy:.4f})")
               
                self.models[target_name] = {
                    'model': best_model,
                    'algorithm': best_algorithm,
                    'accuracy': test_accuracy,
                    'features': feature_columns
                }
       
        # Train enhanced chat model
        self.train_advanced_chat_model()
       
        print("\n✅ All models trained successfully!")
   
    def predict_comprehensive_health_risk(self, health_data):
        """
        Make comprehensive predictions using all trained models
        """
        if not self.models:
            return {"error": "Models not trained yet"}
       
        # Prepare input data
        input_df = pd.DataFrame([health_data])
       
        # Get feature columns used in training
        feature_columns = self.models['risk_level']['features']
       
        # Ensure all required features are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0 # Default value
       
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
                    'prediction': pred,
                    'confidence': float(max(pred_proba)),
                    'algorithm_used': model_info['algorithm'],
                    'model_accuracy': model_info['accuracy']
                }
               
            except Exception as e:
                predictions[target_name] = {'error': str(e)}
       
        return predictions
   
    def generate_personalized_recommendations(self, health_data):
        """Generate personalized health recommendations"""
        recommendations = []
       
        # Risk-based recommendations
        risk_predictions = self.predict_comprehensive_health_risk(health_data)
       
        if 'risk_level' in risk_predictions:
            risk_level = risk_predictions['risk_level']['prediction']
           
            if risk_level in ['High', 'Critical']:
                recommendations.append({
                    'category': 'Medical Care',
                    'priority': 'High',
                    'recommendation': 'Schedule frequent prenatal visits and consider high-risk pregnancy specialist consultation',
                    'reasoning': f'Based on your risk level assessment: {risk_level}'
                })
       
        # Age-based recommendations
        age = health_data.get('age', 0)
        if age > 35:
            recommendations.append({
                'category': 'Genetic Testing',
                'priority': 'Medium',
                'recommendation': 'Discuss genetic screening options with your healthcare provider',
                'reasoning': 'Advanced maternal age increases certain genetic risks'
            })
        elif age < 18:
            recommendations.append({
                'category': 'Nutrition',
                'priority': 'High',
                'recommendation': 'Focus on adequate nutrition for both your growth and baby\'s development',
                'reasoning': 'Teenage pregnancy requires extra nutritional support'
            })
       
        # BMI-based recommendations
        bmi = health_data.get('bmi_pre_pregnancy', 0)
        if bmi > 30:
            recommendations.append({
                'category': 'Weight Management',
                'priority': 'High',
                'recommendation': 'Work with a nutritionist for healthy weight gain during pregnancy',
                'reasoning': 'High BMI increases risk of complications'
            })
        elif bmi < 18.5:
            recommendations.append({
                'category': 'Nutrition',
                'priority': 'High',
                'recommendation': 'Focus on healthy weight gain with nutrient-dense foods',
                'reasoning': 'Low BMI may require additional weight gain for healthy pregnancy'
            })
       
        # Lifestyle recommendations
        if health_data.get('smoking', 0) == 1:
            recommendations.append({
                'category': 'Smoking Cessation',
                'priority': 'Critical',
                'recommendation': 'Quit smoking immediately and join a smoking cessation program',
                'reasoning': 'Smoking significantly increases pregnancy complications and fetal risks'
            })
       
        if health_data.get('exercise_level', 0) < 2:
            recommendations.append({
                'category': 'Exercise',
                'priority': 'Medium',
                'recommendation': 'Start a gentle exercise routine like prenatal yoga or walking',
                'reasoning': 'Regular exercise improves pregnancy outcomes and reduces complications'
            })
       
        # Nutritional recommendations
        if health_data.get('folic_acid_intake', 0) == 0:
            recommendations.append({
                'category': 'Supplements',
                'priority': 'High',
                'recommendation': 'Start taking folic acid supplements (400-800 mcg daily)',
                'reasoning': 'Folic acid is crucial for preventing neural tube defects'
            })
       
        if health_data.get('iron_levels', 0) < 12:
            recommendations.append({
                'category': 'Nutrition',
                'priority': 'Medium',
                'recommendation': 'Increase iron-rich foods and consider iron supplementation',
                'reasoning': 'Low iron levels can lead to anemia during pregnancy'
            })
       
        # Mental health recommendations
        if health_data.get('stress_level', 0) > 3:
            recommendations.append({
                'category': 'Mental Health',
                'priority': 'Medium',
                'recommendation': 'Practice stress reduction techniques and consider counseling support',
                'reasoning': 'High stress levels can affect both maternal and fetal health'
            })
       
        return recommendations
   
    def generate_comprehensive_health_report(self, health_data):
        """Generate comprehensive health assessment report"""
        report = {
            'assessment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patient_summary': {},
            'risk_predictions': {},
            'recommendations': [],
            'monitoring_schedule': {},
            'emergency_contacts': [],
            'educational_resources': []
        }
       
        # Patient summary
        report['patient_summary'] = {
            'age': health_data.get('age', 'Not provided'),
            'gestational_age': f"{health_data.get('gestational_age', 0):.1f} weeks",
            'pregnancy_number': health_data.get('previous_pregnancies', 0) + 1,
            'pre_pregnancy_bmi': f"{health_data.get('bmi_pre_pregnancy', 0):.1f}",
            'medical_history_flags': self.get_medical_history_flags(health_data)
        }
       
        # Risk predictions
        report['risk_predictions'] = self.predict_comprehensive_health_risk(health_data)
       
        # Personalized recommendations
        report['recommendations'] = self.generate_personalized_recommendations(health_data)
       
        # Monitoring schedule
        report['monitoring_schedule'] = self.generate_monitoring_schedule(health_data)
       
        # Educational resources
        report['educational_resources'] = self.get_educational_resources(health_data)
       
        return report
   
    def get_medical_history_flags(self, health_data):
        """Get relevant medical history flags"""
        flags = []
       
        if health_data.get('diabetes_history', 0) == 1:
            flags.append('Diabetes History')
        if health_data.get('hypertension_history', 0) == 1:
            flags.append('Hypertension History')
        if health_data.get('heart_disease', 0) == 1:
            flags.append('Heart Disease')
        if health_data.get('previous_miscarriages', 0) > 0:
            flags.append(f"Previous Miscarriages: {health_data['previous_miscarriages']}")
        if health_data.get('mental_health_history', 0) == 1:
            flags.append('Mental Health History')
       
        return flags if flags else ['No significant medical history']
   
    def generate_monitoring_schedule(self, health_data):
        """Generate personalized monitoring schedule"""
        gestational_age = health_data.get('gestational_age', 0)
        risk_predictions = self.predict_comprehensive_health_risk(health_data)
       
        schedule = {
            'next_appointment': 'As recommended by healthcare provider',
            'recommended_tests': [],
            'monitoring_frequency': 'Standard prenatal schedule'
        }
       
        # Adjust based on gestational age
        if gestational_age < 12:
            schedule['recommended_tests'].extend([
                'First trimester screening (10-13 weeks)',
                'Dating ultrasound',
                'Complete blood count',
                'Urine analysis'
            ])
        elif gestational_age < 28:
            schedule['recommended_tests'].extend([
                'Anatomy scan (18-22 weeks)',
                'Glucose screening (24-28 weeks)',
                'Blood pressure monitoring'
            ])
        else:
            schedule['recommended_tests'].extend([
                'Group B strep test (35-37 weeks)',
                'Non-stress tests if indicated',
                'Regular fetal movement monitoring'
            ])
       
        # Adjust based on risk level
        if 'risk_level' in risk_predictions:
            risk_level = risk_predictions['risk_level']['prediction']
            if risk_level in ['High', 'Critical']:
                schedule['monitoring_frequency'] = 'Increased frequency - every 1-2 weeks'
                schedule['recommended_tests'].append('Additional specialist consultations')
       
        return schedule
   
    def get_educational_resources(self, health_data):
        """Get personalized educational resources"""
        resources = []
        gestational_age = health_data.get('gestational_age', 0)
       
        # Stage-specific resources
        if gestational_age < 12:
            resources.extend([
                'First Trimester: What to Expect',
                'Prenatal Vitamin Guide',
                'Early Pregnancy Symptoms Management'
            ])
        elif gestational_age < 28:
            resources.extend([
                'Second Trimester: Baby Development',
                'Prenatal Testing Options',
                'Exercise During Pregnancy'
            ])
        else:
            resources.extend([
                'Third Trimester: Preparing for Birth',
                'Labor Signs and When to Call Doctor',
                'Breastfeeding Preparation'
            ])
       
        # Risk-specific resources
        risk_predictions = self.predict_comprehensive_health_risk(health_data)
       
        if risk_predictions.get('gestational_diabetes', {}).get('prediction') == 1:
            resources.append('Managing Gestational Diabetes')
       
        if risk_predictions.get('preeclampsia', {}).get('prediction') == 1:
            resources.append('Preeclampsia: Warning Signs and Management')
       
        if risk_predictions.get('postpartum_depression_risk', {}).get('prediction') == 1:
            resources.append('Postpartum Mental Health Support')
       
        return resources
   
    def save_models(self):
        """Save all trained models and components"""
        try:
            # Create directory if it doesn't exist
            os.makedirs('maternal_models', exist_ok=True)
           
            # Save individual models
            for model_name, model_info in self.models.items():
                with open(f'maternal_models/{model_name}_model.pkl', 'wb') as f:
                    pickle.dump(model_info['model'], f)
           
            # Save scalers
            with open('maternal_models/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
           
            # Save encoders
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
                'models': {name: {k: v for k, v in info.items() if k != 'model'}
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
        """Load all trained models and components"""
        try:
            # Load individual models
            for model_file in os.listdir('maternal_models'):
                if model_file.endswith('_model.pkl'):
                    model_name = model_file.replace('_model.pkl', '')
                    with open(f'maternal_models/{model_file}', 'rb') as f:
                        model = pickle.load(f)
                   
                    # Load metadata to reconstruct model info
                    with open('maternal_models/metadata.json', 'r') as f:
                        metadata = json.load(f)
                   
                    if model_name in metadata['models']:
                        self.models[model_name] = metadata['models'][model_name].copy()
                        self.models[model_name]['model'] = model
           
            # Load scalers
            with open('maternal_models/scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
           
            # Load encoders
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
def main():
    """
    Main function to demonstrate the integrated Advanced Maternal Health AI System
    """
    print("🤱 Integrated Advanced Maternal Health AI System with Human-like Chat")
    print("=" * 60)
   
    # Initialize the system
    ai_system = AdvancedMaternalHealthAI()
   
    # Try to load existing models first
    models_loaded = ai_system.load_models()
   
    if not models_loaded:
        print("\nNo existing models found. Training new models...")
        print("This may take a few minutes...\n")
       
        # Generate dataset
        print("📊 Generating comprehensive maternal health dataset...")
        dataset = ai_system.generate_comprehensive_maternal_dataset()
        print(f"Generated dataset with {len(dataset)} samples and {len(dataset.columns)} features")
       
        # Train models
        ai_system.train_comprehensive_models(dataset)
   
    # Demo interactions
    print("\n" + "="*60)
    print("🎯 DEMO: Integrated AI Capabilities")
    print("="*60)
   
    # 1. Demo comprehensive health assessment
    print("\n1. 📋 Comprehensive Health Risk Assessment")
    print("-" * 40)
   
    # Sample patient data
    sample_patient = {
        'age': 32,
        'gestational_age': 28,
        'bmi_pre_pregnancy': 26.5,
        'systolic_bp': 135,
        'diastolic_bp': 85,
        'glucose_fasting': 95,
        'hemoglobin': 10.5,
        'previous_pregnancies': 1,
        'diabetes_history': 0,
        'hypertension_history': 0,
        'smoking': 0,
        'exercise_level': 2,
        'stress_level': 3,
        'folic_acid_intake': 1,
        'prenatal_vitamins': 1
    }
   
    # Fill in default values for other required features
    default_values = {
        'education_level': 3, 'income_level': 2, 'marital_status': 1, 'employment': 1,
        'previous_miscarriages': 0, 'heart_disease': 0, 'kidney_disease': 0,
        'autoimmune_disorders': 0, 'mental_health_history': 0,
        'weight_pre_pregnancy': 70, 'height': 165, 'weight_gain': 10,
        'heart_rate': 75, 'protein_urine': 0, 'white_blood_cells': 7000,
        'platelets': 200000, 'alcohol': 0, 'drug_use': 0, 'sleep_hours': 7,
        'vitamin_d': 25, 'iron_levels': 12, 'access_to_healthcare': 3,
        'social_support': 3, 'transportation_access': 1, 'insurance_coverage': 1,
        'air_quality_index': 45, 'water_quality': 2, 'housing_quality': 3,
        'fetal_weight_percentile': 50, 'amniotic_fluid_level': 2, 'placental_position': 2
    }
   
    # Merge sample patient data with defaults
    complete_patient_data = {**default_values, **sample_patient}
   
    # Make predictions
    predictions = ai_system.predict_comprehensive_health_risk(complete_patient_data)
   
    print(f"Patient: 32-year-old, 28 weeks pregnant")
    print(f"Key indicators: BP 135/85, BMI 26.5, Hemoglobin 10.5")
    print("\nPredictions:")
    for pred_name, pred_info in predictions.items():
        if 'error' not in pred_info:
            print(f" {pred_name}: {pred_info['prediction']} "
                  f"(Confidence: {pred_info['confidence']:.2f}, "
                  f"Algorithm: {pred_info['algorithm_used']})")
   
    # 2. Generate comprehensive report
    print("\n2. 📄 Comprehensive Health Report")
    print("-" * 40)
   
    health_report = ai_system.generate_comprehensive_health_report(complete_patient_data)
   
    print(f"Assessment Date: {health_report['assessment_date']}")
    print(f"Gestational Age: {health_report['patient_summary']['gestational_age']}")
    print(f"Pregnancy Number: {health_report['patient_summary']['pregnancy_number']}")
    print(f"Pre-pregnancy BMI: {health_report['patient_summary']['pre_pregnancy_bmi']}")
   
    print(f"\nMedical History: {', '.join(health_report['patient_summary']['medical_history_flags'])}")
   
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(health_report['recommendations'][:3], 1):
        print(f" {i}. [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
   
    # 3. Demo human-like chat system
    print("\n3. 💬 Human-like Chat System Demo")
    print("-" * 40)
   
    sample_questions = [
        "hey, i'm freaking out a bit about this whole pregnancy thing",
        "ugh my back is killing me and i can barely sleep",
        "What should I eat during my second trimester?",
        "I'm feeling anxious about labor and delivery",
        "I'm experiencing severe headaches and blurred vision",
        "Tell me about gestational diabetes",
        "my partner just doesn't get how tired i am all the time"
    ]
   
    for question in sample_questions:
        print(f"\nQ: {question}")
        response = ai_system.get_advanced_chat_response(question,
                                                      {'trimester_2': True, 'first_pregnancy': True},
                                                      user_id="demo_user")
       
        print(f"A: {response['response']}")
        print(f" Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
       
        if response.get('emotions_detected'):
            print(f" Emotions: {', '.join(response['emotions_detected'])}")
       
        if response.get('emergency'):
            print(" 🚨 EMERGENCY DETECTED")
       
        if response.get('followup'):
            print(f" Follow-up: {response['followup']}")
       
        if response.get('suggestions'):
            print(f" Suggestions: {', '.join(response['suggestions'][:2])}")
   
    print("\n" + "="*60)
    print("✅ Demo completed! The integrated system is ready for use.")
    print("="*60)
if __name__ == "__main__":
    main()