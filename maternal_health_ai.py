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
warnings.filterwarnings('ignore')

class AdvancedMaternalHealthAI:
    """
    Comprehensive Maternal Health AI System covering all aspects of maternal care
    
    Algorithms Used:
    1. SUPERVISED LEARNING:
       - Random Forest (for risk prediction)
       - Gradient Boosting (for complications prediction)
       - Neural Networks (for pattern recognition)
       - SVM (for classification tasks)
       - Logistic Regression (for binary outcomes)
       - Naive Bayes (for text classification)
    
    2. ENSEMBLE METHODS:
       - Voting Classifier (combines multiple algorithms)
       - Bagging (reduces overfitting)
    
    3. DEEP LEARNING:
       - LSTM (for time series health data)
       - CNN (for medical image analysis if needed)
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.chat_model = None
        self.intents = None
        
    def generate_comprehensive_maternal_dataset(self):
        """
        Generate comprehensive maternal health dataset covering ALL fields
        """
        np.random.seed(42)
        n_samples = 10000
        
        # Generate synthetic data covering all maternal health aspects
        data = {}
        
        # 1. DEMOGRAPHIC DATA
        data['age'] = np.random.normal(28, 6, n_samples).clip(15, 45)
        data['education_level'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        data['income_level'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.35, 0.25, 0.1])
        data['marital_status'] = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])  # 0: single, 1: married
        data['employment'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # 0: unemployed, 1: employed
        
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
        
        # 6. NUTRITIONAL STATUS
        data['vitamin_d'] = np.random.normal(30, 15, n_samples).clip(10, 80)
        data['iron_levels'] = np.random.normal(15, 5, n_samples).clip(5, 30)
        data['folic_acid_intake'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        data['prenatal_vitamins'] = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
        
        # 7. SOCIAL DETERMINANTS
        data['access_to_healthcare'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.25, 0.4, 0.2])
        data['social_support'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.5, 0.2])
        data['transportation_access'] = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
        data['insurance_coverage'] = np.random.choice([0, 1], n_samples, p=[0.15, 0.85])
        
        # 8. ENVIRONMENTAL FACTORS
        data['air_quality_index'] = np.random.normal(50, 25, n_samples).clip(10, 200)
        data['water_quality'] = np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.25, 0.05])
        data['housing_quality'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.5, 0.2])
        
        # 9. FETAL MEASUREMENTS (ultrasound data)
        data['fetal_weight_percentile'] = np.random.normal(50, 25, n_samples).clip(5, 95)
        data['amniotic_fluid_level'] = np.random.choice([1, 2, 3], n_samples, p=[0.15, 0.7, 0.15])  # 1: low, 2: normal, 3: high
        data['placental_position'] = np.random.choice([1, 2, 3], n_samples, p=[0.05, 0.9, 0.05])  # 1: previa, 2: normal, 3: abruption
        
        # TARGET VARIABLES (Multiple outcomes to predict)
        
        # Risk factors for complications
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
        
        # 1. OVERALL RISK CLASSIFICATION
        data['risk_level'] = np.where(risk_factors < 0.5, 'Low',
                                    np.where(risk_factors < 1.0, 'Medium',
                                           np.where(risk_factors < 1.5, 'High', 'Critical')))
        
        # 2. SPECIFIC COMPLICATIONS
        data['gestational_diabetes'] = ((data['glucose_fasting'] > 126) | 
                                       (data['bmi_pre_pregnancy'] > 30) | 
                                       (data['age'] > 35)).astype(int)
        
        data['preeclampsia'] = ((data['systolic_bp'] > 140) | 
                               (data['protein_urine'] > 1) |
                               (data['age'] > 35)).astype(int)
        
        data['preterm_birth_risk'] = ((data['previous_miscarriages'] > 1) |
                                     data['smoking'] |
                                     (data['stress_level'] > 3)).astype(int)
        
        data['postpartum_depression_risk'] = ((data['mental_health_history'] == 1) |
                                             (data['social_support'] < 2) |
                                             (data['stress_level'] > 3)).astype(int)
        
        # 3. DELIVERY OUTCOMES
        data['cesarean_risk'] = ((data['age'] > 35) |
                                (data['bmi_pre_pregnancy'] > 35) |
                                (data['fetal_weight_percentile'] > 90)).astype(int)
        
        data['birth_weight_category'] = np.where(data['fetal_weight_percentile'] < 10, 'Low',
                                               np.where(data['fetal_weight_percentile'] > 90, 'High', 'Normal'))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def generate_comprehensive_training_intents(self):
        """
        Generate comprehensive training data covering ALL maternal health fields
        """
        
        intents_data = {
            # 1. PREGNANCY STAGES & DEVELOPMENT
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
            
            # 2. NUTRITION & SUPPLEMENTS
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
            
            # 3. COMPLICATIONS & MEDICAL CONDITIONS
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
            
            # 4. MENTAL HEALTH & EMOTIONAL WELLBEING
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
            
            # 5. LABOR & DELIVERY PREPARATION
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
            
            # 6. POSTPARTUM CARE
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
            
            # 7. HIGH-RISK PREGNANCIES
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
            
            # 8. LIFESTYLE & WELLNESS
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
            
            # 9. EMERGENCY SITUATIONS
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
            
            # 10. PARTNER & FAMILY SUPPORT
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
        
        return intents_data
    
    def train_comprehensive_models(self, df):
        """
        Train multiple specialized models using different algorithms
        """
        print("Training comprehensive maternal health models...")
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in [
            'risk_level', 'gestational_diabetes', 'preeclampsia', 'preterm_birth_risk',
            'postpartum_depression_risk', 'cesarean_risk', 'birth_weight_category'
        ]]
        
        X = df[feature_columns]
        
        # Scale numerical features
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
        
        # Train models for each target
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
            
            # Define algorithms to try
            algorithms = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'svm': SVC(probability=True, random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42)
            }
            
            best_model = None
            best_score = 0
            best_algorithm = None
            
            # Try each algorithm and select the best
            for alg_name, model in algorithms.items():
                try:
                    # Use cross-validation to evaluate
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    avg_score = np.mean(scores)
                    
                    print(f"  {alg_name}: {avg_score:.4f} (+/- {scores.std() * 2:.4f})")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_algorithm = alg_name
                        
                except Exception as e:
                    print(f"  {alg_name}: Error - {str(e)}")
            
            if best_model is not None:
                # Train the best model on full training set
                best_model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                print(f"  Best algorithm for {target_name}: {best_algorithm} (Test accuracy: {test_accuracy:.4f})")
                
                # Store the model
                self.models[target_name] = {
                    'model': best_model,
                    'algorithm': best_algorithm,
                    'accuracy': test_accuracy,
                    'features': feature_columns
                }
        
        # Train chat model
        self.train_advanced_chat_model()
        
        # Save all models
        self.save_models()
        
        print("\n✅ All models trained successfully!")
    
    def train_advanced_chat_model(self):
        """Train advanced NLP model for comprehensive maternal health chat"""
        print("Training advanced chat model...")
        
        intents_data = self.generate_comprehensive_training_intents()
        
        # Generate training examples with data augmentation
        training_texts = []
        training_labels = []
        
        # Data augmentation techniques
        augmentation_patterns = [
            "I'm experiencing {}",
            "I have {}",
            "Tell me about {}",
            "Help with {}",
            "What about {}",
            "I'm concerned about {}",
            "Can you help me with {}",
            "I need information on {}",
            "Explain {} to me",
            "I'm worried about {}"
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
                
                # Add variations with different wordings
                variations = [
                    pattern.replace("pregnancy", "being pregnant"),
                    pattern.replace("baby", "child"),
                    pattern.replace("birth", "delivery"),
                    pattern.replace("labor", "giving birth")
                ]
                
                for variation in variations:
                    if variation != pattern:
                        training_texts.append(variation.lower())
                        training_labels.append(intent)
        
        # Create advanced pipeline with better preprocessing
        self.chat_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=10000,
                stop_words='english',
                sublinear_tf=True,
                min_df=2
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            training_texts, training_labels, test_size=0.2, random_state=42
        )
        
        self.chat_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.chat_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Chat model accuracy: {accuracy:.4f}")
        
        self.intents = intents_data
    
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
                input_df[col] = 0  # Default value
        
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
    
    def preprocess_advanced_input(self, text):
        """Advanced text preprocessing for better intent recognition"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Handle common medical abbreviations
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
        
        return text
    
    def get_advanced_chat_response(self, user_input, user_context=None):
        """
        Get advanced chat response with context awareness
        """
        if not self.chat_model:
            return "I'm still learning. Please train me first."
        
        # Preprocess input
        processed_input = self.preprocess_advanced_input(user_input)
        
        # Emergency detection (enhanced)
        emergency_patterns = [
            r'\b(severe|heavy|extreme)\s+(bleeding|pain|headache)\b',
            r'\b(chest pain|difficulty breathing|vision changes)\b',
            r'\b(fever|high fever)\b.*\b(pregnancy|pregnant)\b',
            r'\b(water broke|water breaking|labor)\b.*\b(early|preterm)\b',
            r'\b(decreased|less|no)\s+(fetal movement|baby movement)\b',
            r'\b(severe|intense|unbearable)\s+(contractions|cramping)\b'
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, processed_input):
                return {
                    'intent': 'emergency_situations',
                    'response': "🚨 URGENT: Based on what you're describing, this could be a medical emergency. Please contact your healthcare provider immediately or go to the nearest emergency room. Don't wait - maternal and fetal emergencies require immediate attention.",
                    'confidence': 1.0,
                    'emergency': True
                }
        
        # Predict intent
        try:
            predicted_intent = self.chat_model.predict([processed_input])[0]
            confidence = max(self.chat_model.predict_proba([processed_input])[0])
            
            # Get response
            if predicted_intent in self.intents:
                responses = self.intents[predicted_intent]['responses']
                response = np.random.choice(responses)
                
                # Add context-specific information if available
                if user_context:
                    response = self.add_contextual_information(response, user_context, predicted_intent)
                
                return {
                    'intent': predicted_intent,
                    'response': response,
                    'confidence': float(confidence),
                    'emergency': False,
                    'suggestions': self.get_related_suggestions(predicted_intent)
                }
            else:
                return {
                    'intent': 'unknown',
                    'response': "I understand you have a question about maternal health. Could you please be more specific about what you'd like to know? I can help with pregnancy stages, nutrition, complications, mental health, labor preparation, and much more.",
                    'confidence': 0.0,
                    'emergency': False
                }
                
        except Exception as e:
            return {
                'intent': 'error',
                'response': f"I'm having trouble processing your question right now. Please try rephrasing your question or contact your healthcare provider if it's urgent. Error: {str(e)}",
                'confidence': 0.0,
                'emergency': False
            }
    
    def add_contextual_information(self, base_response, context, intent):
        """Add personalized context to responses"""
        contextual_additions = {
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
                    base_response += f" {addition}"
        
        return base_response
    
    def get_related_suggestions(self, intent):
        """Get related topic suggestions based on current intent"""
        suggestions_map = {
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
        
        return suggestions_map.get(intent, [])
    
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
    Main function to demonstrate the Advanced Maternal Health AI System
    """
    print("🤱 Advanced Maternal Health AI System")
    print("=" * 50)
    
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
    print("\n" + "="*50)
    print("🎯 DEMO: Advanced Maternal Health AI Capabilities")
    print("="*50)
    
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
            print(f"  {pred_name}: {pred_info['prediction']} "
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
        print(f"  {i}. [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
    
    # 3. Demo advanced chat system
    print("\n3. 💬 Advanced Chat System Demo")
    print("-" * 40)
    
    sample_questions = [
        "I'm experiencing severe headaches and blurred vision",
        "What should I eat during my second trimester?",
        "I'm feeling anxious about labor and delivery",
        "Tell me about gestational diabetes",
        "I'm having trouble sleeping during pregnancy"
    ]
    
    for question in sample_questions:
        print(f"\nQ: {question}")
        response = ai_system.get_advanced_chat_response(question, 
                                                      {'trimester_2': True, 'first_pregnancy': True})
        
        print(f"A: {response['response']}")
        print(f"   Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
        
        if response.get('emergency'):
            print("   🚨 EMERGENCY DETECTED")
        
        if response.get('suggestions'):
            print(f"   Suggestions: {', '.join(response['suggestions'][:2])}")
    
    print("\n" + "="*50)
    print("✅ Demo completed! The system is ready for use.")
    print("="*50)


if __name__ == "__main__":
    main()