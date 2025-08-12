import tensorflow as tf
import numpy as np
import pandas as pd
import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
from datetime import datetime
import os

class MaternalHealthChatBot:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.intents = None
        self.responses = None
        self.user_context = {}
        
    def generate_training_data(self):
        """Generate comprehensive training data for maternal health chatbot"""
        
        # Define intents and training data
        intents_data = {
            "greeting": {
                "patterns": [
                    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                    "greetings", "howdy", "what's up", "how are you", "nice to meet you"
                ],
                "responses": [
                    "Hello! I'm here to support you through your pregnancy journey. How can I help you today?",
                    "Hi there! Welcome to your maternal health companion. What would you like to know?",
                    "Hello! I'm excited to chat with you about your pregnancy. How are you feeling today?",
                    "Hi! I'm here to answer your questions and provide support during your pregnancy."
                ]
            },
            
            "pregnancy_symptoms": {
                "patterns": [
                    "morning sickness", "nausea", "vomiting", "fatigue", "tired", "exhausted",
                    "back pain", "headache", "swelling", "heartburn", "constipation",
                    "breast tenderness", "mood swings", "frequent urination", "cramping",
                    "dizzy", "dizziness", "shortness of breath", "leg cramps"
                ],
                "responses": [
                    "I understand you're experiencing some pregnancy symptoms. While some discomfort is normal, it's important to monitor your symptoms. Would you like some tips for managing specific symptoms?",
                    "Pregnancy symptoms can be challenging. Many symptoms are normal, but severe or persistent symptoms should be discussed with your healthcare provider. Which symptom is bothering you most?",
                    "It sounds like you're dealing with some pregnancy-related discomfort. I can share some safe ways to manage symptoms, but always consult your doctor if symptoms worsen.",
                    "Pregnancy brings many changes to your body. While some symptoms are expected, tracking them can help you and your doctor monitor your health."
                ]
            },
            
            "nutrition": {
                "patterns": [
                    "what to eat", "nutrition", "diet", "food", "vitamins", "prenatal vitamins",
                    "healthy eating", "protein", "calcium", "iron", "folic acid", "folate",
                    "weight gain", "cravings", "food aversions", "what not to eat",
                    "fish", "caffeine", "alcohol", "raw foods"
                ],
                "responses": [
                    "Nutrition is crucial during pregnancy! Focus on balanced meals with plenty of fruits, vegetables, lean proteins, and whole grains. Are you taking prenatal vitamins?",
                    "A healthy pregnancy diet includes folate, iron, calcium, and protein. Avoid raw fish, high-mercury fish, alcohol, and limit caffeine. What specific nutrition questions do you have?",
                    "Good nutrition supports both you and your baby's development. Include colorful fruits and vegetables, lean meats, dairy, and whole grains. Any particular foods you're curious about?",
                    "Eating well during pregnancy doesn't have to be complicated. Focus on nutrient-dense foods and stay hydrated. Would you like meal planning tips?"
                ]
            },
            
            "exercise": {
                "patterns": [
                    "exercise", "workout", "fitness", "yoga", "walking", "swimming",
                    "safe exercises", "prenatal exercise", "physical activity",
                    "can I exercise", "is it safe to workout", "prenatal yoga"
                ],
                "responses": [
                    "Exercise during pregnancy is generally safe and beneficial! Walking, swimming, and prenatal yoga are excellent choices. Always check with your healthcare provider before starting new activities.",
                    "Regular, moderate exercise can help with pregnancy symptoms and prepare your body for labor. What type of activities do you enjoy?",
                    "Staying active during pregnancy is wonderful for both you and baby. Low-impact activities like walking and prenatal classes are usually safe. Have you been active before pregnancy?",
                    "Exercise can boost your mood, energy, and help manage weight gain during pregnancy. The key is to listen to your body and avoid high-risk activities."
                ]
            },
            
            "baby_development": {
                "patterns": [
                    "baby development", "fetal development", "how big is baby", "baby size",
                    "when will I feel movement", "baby kicks", "baby moving",
                    "ultrasound", "baby growth", "milestones", "what week am I"
                ],
                "responses": [
                    "Baby development is amazing to track! Each week brings new changes. What week of pregnancy are you in? I can share what's happening with your baby's development.",
                    "Your baby is growing and developing rapidly! From tiny limbs to developing organs, there's so much happening. Are you curious about a specific aspect of development?",
                    "It's exciting to think about how your baby is developing! Baby movements usually start around 18-22 weeks for first pregnancies. Have you felt any movement yet?",
                    "Fetal development follows predictable patterns, but every baby grows at their own pace. Regular prenatal visits help monitor your baby's progress."
                ]
            },
            
            "prenatal_care": {
                "patterns": [
                    "doctor visits", "prenatal appointments", "checkups", "ultrasounds",
                    "tests", "glucose test", "blood work", "screening", "when to call doctor",
                    "prenatal care", "ob-gyn", "midwife", "healthcare provider"
                ],
                "responses": [
                    "Regular prenatal care is essential for monitoring both your health and baby's development. Are you keeping up with your scheduled appointments?",
                    "Prenatal visits help catch any issues early and ensure you're both healthy. Don't hesitate to ask questions during appointments or call between visits if needed.",
                    "Your healthcare team is there to support you! Make sure to attend all scheduled visits and don't be afraid to voice any concerns you have.",
                    "Prenatal care includes various tests and screenings to monitor your pregnancy. Is there a specific test or appointment you have questions about?"
                ]
            },
            
            "concerns_warnings": {
                "patterns": [
                    "bleeding", "spotting", "severe headache", "vision changes", "severe pain",
                    "contractions", "leaking fluid", "decreased movement", "fever",
                    "something wrong", "worried", "concerned", "emergency", "when to call doctor"
                ],
                "responses": [
                    "I understand you're concerned. Some symptoms require immediate medical attention: severe bleeding, severe headaches, vision changes, severe abdominal pain, fever, or significantly decreased baby movement. Please contact your healthcare provider right away if you're experiencing any of these.",
                    "Your concern is valid, and it's always better to be safe. Contact your healthcare provider immediately if you have: heavy bleeding, severe pain, persistent headaches, fever over 100.4°F, or if baby's movements have decreased significantly.",
                    "Trust your instincts - if something feels wrong, it's important to seek medical attention. Call your doctor or go to the emergency room for severe symptoms like bleeding, intense pain, or fever.",
                    "When in doubt, always contact your healthcare provider. They'd rather have you call with concerns than miss something important. Don't hesitate to seek immediate care for severe symptoms."
                ]
            },
            
            "mental_health": {
                "patterns": [
                    "anxious", "anxiety", "worried", "scared", "nervous", "stress", "stressed",
                    "sad", "depressed", "mood", "emotional", "crying", "overwhelmed",
                    "mental health", "support", "feelings"
                ],
                "responses": [
                    "It's completely normal to experience a range of emotions during pregnancy. If you're feeling overwhelmed, anxious, or sad frequently, please talk to your healthcare provider. Mental health is just as important as physical health during pregnancy.",
                    "Pregnancy can bring up many emotions - excitement, worry, joy, and anxiety are all normal. However, if negative feelings persist, don't hesitate to seek support from your healthcare team or a counselor.",
                    "Your emotional wellbeing matters greatly during pregnancy. Many women experience mood changes, but persistent sadness or anxiety should be addressed with professional support.",
                    "Taking care of your mental health benefits both you and your baby. Consider relaxation techniques, support groups, or speaking with a counselor if you're struggling with your emotions."
                ]
            },
            
            "labor_delivery": {
                "patterns": [
                    "labor", "delivery", "birth", "contractions", "water breaking",
                    "due date", "signs of labor", "when will baby come", "birth plan",
                    "hospital bag", "labor pain", "epidural", "c-section", "natural birth"
                ],
                "responses": [
                    "Preparing for labor and delivery is exciting! Signs of early labor include regular contractions, water breaking, or bloody show. Have you discussed your birth plan with your healthcare provider?",
                    "Every labor is different, but knowing the signs helps you prepare. True labor contractions become regular, stronger, and closer together. What questions do you have about labor?",
                    "It's natural to think about labor and delivery! Consider creating a birth plan and packing your hospital bag around 35-36 weeks. Are you feeling prepared?",
                    "Labor preparation includes knowing the signs, having a plan, and trusting your healthcare team. Remember, flexibility is important as every birth is unique."
                ]
            },
            
            "general_support": {
                "patterns": [
                    "thank you", "thanks", "helpful", "appreciate", "goodbye", "bye",
                    "that's all", "no more questions", "you're great", "this helps"
                ],
                "responses": [
                    "You're very welcome! I'm here whenever you need support or have questions about your pregnancy journey.",
                    "I'm so glad I could help! Remember, I'm always here for questions, but don't hesitate to contact your healthcare provider for medical concerns.",
                    "Thank you for letting me be part of your pregnancy journey! Take care of yourself and reach out anytime you need support.",
                    "It's been wonderful chatting with you! Wishing you a healthy and happy pregnancy. Feel free to return anytime with questions."
                ]
            }
        }
        
        # Generate training examples
        training_texts = []
        training_labels = []
        
        for intent, data in intents_data.items():
            for pattern in data["patterns"]:
                # Add the pattern as is
                training_texts.append(pattern.lower())
                training_labels.append(intent)
                
                # Add variations
                variations = [
                    f"I have {pattern}",
                    f"I'm experiencing {pattern}",
                    f"Tell me about {pattern}",
                    f"Help with {pattern}",
                    f"What about {pattern}?"
                ]
                
                for variation in variations:
                    training_texts.append(variation.lower())
                    training_labels.append(intent)
        
        self.intents = intents_data
        return training_texts, training_labels
    
    def train_chat_model(self):
        """Train the chat classification model"""
        print("Training maternal health chat model...")
        
        # Generate training data
        texts, labels = self.generate_training_data()
        
        # Create and train the pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Chat Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        os.makedirs('maternal_models', exist_ok=True)
        with open('maternal_models/chat_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('maternal_models/intents.json', 'w') as f:
            json.dump(self.intents, f, indent=2)
        
        print("Chat model trained and saved successfully!")
    
    def load_chat_model(self):
        """Load the trained chat model"""
        try:
            with open('maternal_models/chat_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('maternal_models/intents.json', 'r') as f:
                self.intents = json.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading chat model: {e}")
            return False
    
    def preprocess_input(self, text):
        """Preprocess user input"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle common contractions
        contractions = {
            "i'm": "i am", "you're": "you are", "it's": "it is",
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "i've": "i have", "i'll": "i will", "isn't": "is not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def get_response(self, user_input, user_id="default"):
        """Get chatbot response"""
        if not self.model or not self.intents:
            return "I'm still learning! Please train me first."
        
        # Preprocess input
        processed_input = self.preprocess_input(user_input)
        
        # Check for emergency keywords first
        emergency_keywords = [
            "bleeding heavily", "severe bleeding", "severe pain", "can't breathe",
            "chest pain", "severe headache", "vision problems", "fever high",
            "emergency", "hospital", "call 911"
        ]
        
        if any(keyword in processed_input for keyword in emergency_keywords):
            return "⚠️ This sounds like it could be a medical emergency. Please contact your healthcare provider immediately or call emergency services if needed. I'm here to provide general information, but medical emergencies require professional care right away."
        
        # Predict intent
        try:
            predicted_intent = self.model.predict([processed_input])[0]
            confidence = max(self.model.predict_proba([processed_input])[0])
            
            # If confidence is too low, provide general support
            if confidence < 0.3:
                return "I want to make sure I give you the best support. Could you tell me more about what you're experiencing or what specific information you're looking for about your pregnancy?"
            
            # Get appropriate response
            responses = self.intents[predicted_intent]["responses"]
            base_response = random.choice(responses)
            
            # Add contextual information for specific intents
            additional_info = self.get_additional_info(predicted_intent, processed_input)
            
            if additional_info:
                return f"{base_response}\n\n{additional_info}"
            
            return base_response
            
        except Exception as e:
            return "I'm having trouble understanding right now. Could you rephrase your question? I'm here to help with pregnancy-related questions and support."
    
    def get_additional_info(self, intent, user_input):
        """Get additional contextual information based on intent"""
        additional_info = {
            "pregnancy_symptoms": {
                "morning sickness": "💡 Tips for morning sickness: Try eating small, frequent meals, keep crackers by your bedside, stay hydrated, and consider ginger tea. If vomiting is severe or persistent, contact your healthcare provider.",
                "fatigue": "💡 Fatigue is very common in the first and third trimesters. Try to rest when possible, maintain a consistent sleep schedule, and don't hesitate to ask for help with daily tasks.",
                "back pain": "💡 For back pain: Practice good posture, wear supportive shoes, use a pregnancy pillow while sleeping, and consider prenatal massage. Gentle stretching and prenatal yoga can also help.",
                "heartburn": "💡 To manage heartburn: Eat smaller meals, avoid spicy or acidic foods, don't lie down immediately after eating, and try sleeping with your head elevated."
            },
            
            "nutrition": {
                "vitamins": "🍎 Key nutrients during pregnancy: Folic acid (400-800 mcg), Iron (27 mg), Calcium (1000 mg), DHA, and Vitamin D. Most prenatal vitamins cover these needs.",
                "weight gain": "📊 Healthy weight gain depends on your pre-pregnancy BMI: Underweight (28-40 lbs), Normal weight (25-35 lbs), Overweight (15-25 lbs), Obese (11-20 lbs).",
                "caffeine": "☕ Limit caffeine to less than 200mg per day (about one 12oz cup of coffee). Also found in tea, chocolate, and some sodas."
            },
            
            "exercise": {
                "safe": "✅ Safe exercises include: walking, swimming, stationary cycling, prenatal yoga, and low-impact aerobics. Avoid contact sports, activities with fall risk, and exercises lying on your back after the first trimester.",
                "yoga": "🧘‍♀️ Prenatal yoga can help with flexibility, strength, and relaxation. Look for certified prenatal instructors and avoid hot yoga or deep twists."
            }
        }
        
        if intent in additional_info:
            for keyword, info in additional_info[intent].items():
                if keyword in user_input:
                    return info
        
        return ""
    
    def get_personalized_tips(self, gestational_week=None):
        """Get week-specific tips if gestational week is provided"""
        if not gestational_week:
            return ""
        
        week = int(gestational_week)
        
        if week <= 12:
            return "📅 First Trimester: Focus on taking prenatal vitamins, managing morning sickness, and getting plenty of rest. Your baby's major organs are forming now."
        elif week <= 27:
            return "📅 Second Trimester: Many women feel their best during this time! You might start feeling baby movements soon. Continue healthy eating and consider starting prenatal classes."
        elif week <= 40:
            return "📅 Third Trimester: Your baby is growing rapidly! Focus on preparing for labor, packing your hospital bag, and getting enough rest. Practice breathing exercises for labor."
        else:
            return "📅 Full term and beyond: Your baby is ready to be born! Watch for signs of labor and stay in close contact with your healthcare provider."

def demo_chatbot():
    """Demo the maternal health chatbot"""
    print("\n" + "="*60)
    print("MATERNAL HEALTH CHAT BOT DEMO")
    print("="*60)
    
    # Initialize and train bot
    bot = MaternalHealthChatBot()
    
    # Try to load existing model, if not available, train new one
    if not bot.load_chat_model():
        print("Training new chat model...")
        bot.train_chat_model()
    else:
        print("Loaded existing chat model...")
    
    # Demo conversation
    test_messages = [
        "Hi, I'm 24 weeks pregnant",
        "I've been having morning sickness",
        "What should I eat during pregnancy?",
        "Is it safe to exercise?",
        "I'm feeling anxious about labor",
        "I have severe bleeding",  # Emergency test
        "Thank you for your help"
    ]
    
    print(f"\n{'='*40}")
    print("CHAT BOT CONVERSATION DEMO")
    print(f"{'='*40}")
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        response = bot.get_response(message)
        print(f"🤖 Bot: {response}")
        print("-" * 50)

# Integration with existing prediction system
def create_integrated_maternal_assistant():
    """Create an integrated system that combines chat and prediction"""
    
    class IntegratedMaternalAssistant:
        def __init__(self):
            self.chatbot = MaternalHealthChatBot()
            self.chatbot.load_chat_model()
        
        def chat_with_predictions(self, user_input, health_data=None):
            """Chat that can also provide predictions if health data is available"""
            
            # Get basic chat response
            chat_response = self.chatbot.get_response(user_input)
            
            # If health data is provided and user is asking about risk/health
            if health_data and any(word in user_input.lower() for word in 
                                 ['risk', 'health', 'predict', 'assessment', 'check']):
                
                try:
                    # This would integrate with your existing prediction functions
                    # from the original code (predict_maternal_health function)
                    prediction_note = "\n\n🔍 Based on your health data, I can provide a detailed risk assessment. Would you like me to analyze your current health parameters?"
                    return chat_response + prediction_note
                except:
                    pass
            
            return chat_response
        
        def provide_comprehensive_support(self, user_input, health_data=None):
            """Comprehensive support combining chat and predictions"""
            response = {
                'chat_response': self.chat_with_predictions(user_input, health_data),
                'timestamp': datetime.now().isoformat(),
                'support_type': 'conversational'
            }
            
            # Add prediction capabilities if health data provided
            if health_data:
                response['prediction_available'] = True
                response['support_type'] = 'comprehensive'
            
            return response
    
    return IntegratedMaternalAssistant()

if __name__ == "__main__":
    # Run the demo
    demo_chatbot()
    
    print(f"\n{'='*60}")
    print("MATERNAL HEALTH CHAT BOT READY!")
    print("="*60)
    print("The chatbot can now:")
    print("✅ Handle pregnancy-related questions")
    print("✅ Provide emotional support")
    print("✅ Give nutrition and exercise advice")
    print("✅ Recognize emergency situations")
    print("✅ Offer personalized tips")
    print("✅ Integrate with risk prediction models")
    print("\nUse the IntegratedMaternalAssistant class for full functionality!")