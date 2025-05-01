from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import hashlib
import json
from datetime import datetime
import pickle
import os
import openai
from typing import Dict, List, Optional

app = Flask(__name__)

# Configuration
MODEL_FILE = 'fraud_detection_model.pkl'
SCALER_FILE = 'scaler.pkl'
DATA_FILE = 'transaction_data.csv'
OPENAI_API_KEY = 'your_openai_api_key'  # Replace with your actual key

# Initialize AI Enhancements
class AIEnhancements:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.behavior_profiles = {}
        self.fraud_patterns = self._load_known_patterns()
        
    def _load_known_patterns(self) -> Dict:
        try:
            with open('fraud_patterns.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "common_fraud_behaviors": [
                    "rapid checkout after account creation",
                    "multiple failed payment attempts",
                    "unusual high-value purchases",
                    "shipping-billing address mismatch",
                    "unusual device/browser characteristics"
                ],
                "high_risk_countries": ["RU", "CN", "NG", "BR", "VN", "PK"],
                "suspicious_email_patterns": [
                    "temporary email domains",
                    "random character sequences",
                    "disposable email services"
                ]
            }
    
    def analyze_transaction_text(self, transaction_data: Dict) -> Dict:
        prompt = self._create_analysis_prompt(transaction_data)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a fraud detection AI for an e-commerce platform. Analyze the transaction for potential fraud indicators."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            return {
                "analysis": analysis,
                "risk_indicators": self._extract_risk_indicators(analysis),
                "recommendation": self._generate_recommendation(analysis)
            }
            
        except Exception as e:
            print(f"Error in AI analysis: {str(e)}")
            return {
                "error": str(e),
                "analysis": "Unable to complete AI analysis",
                "risk_indicators": [],
                "recommendation": "Review manually"
            }
    
    def _create_analysis_prompt(self, transaction: Dict) -> str:
        prompt = f"""
        Analyze this e-commerce transaction for potential fraud indicators:
        
        Transaction Summary:
        - Amount: ${sum(item['price'] * item.get('quantity', 1) for item in transaction['cart'])}
        - Items: {len(transaction['cart'])} products
        - Customer: {transaction['userInfo'].get('name', 'Unknown')}
        - Email: {transaction['userInfo'].get('email', 'Not provided')}
        - Location: {transaction['userInfo'].get('location', 'Unknown')}
        - Device: {'Known' if transaction.get('deviceFingerprint') != 'unknown' else 'Unknown/Suspicious'}
        - Session Duration: {transaction['userBehavior'].get('sessionDuration', 0)} seconds
        
        Behavioral Patterns:
        - Add to cart rate: {transaction['userBehavior'].get('addToCartRate', 0)} items/minute
        - Page changes: {transaction['userBehavior'].get('pageChanges', 0)} during session
        
        Known Fraud Patterns to consider:
        {json.dumps(self.fraud_patterns, indent=2)}
        
        Provide a detailed analysis of potential fraud indicators, similarity to known fraud patterns,
        and a risk assessment. Format your response with:
        1. Summary of key observations
        2. List of potential risk factors
        3. Overall risk assessment (Low, Medium, High)
        4. Recommended action (Approve, Review, Block)
        """
        return prompt
    
    def _extract_risk_indicators(self, analysis_text: str) -> List[str]:
        indicators = []
        text_lower = analysis_text.lower()
        
        for pattern in self.fraud_patterns["common_fraud_behaviors"]:
            if pattern.lower() in text_lower:
                indicators.append(pattern)
                
        if "high risk" in text_lower:
            indicators.append("AI flagged as high risk")
        if "suspicious" in text_lower:
            indicators.append("AI found suspicious patterns")
            
        return indicators if indicators else ["No strong AI indicators found"]
    
    def _generate_recommendation(self, analysis_text: str) -> str:
        text_lower = analysis_text.lower()
        
        if "high risk" in text_lower or "recommend block" in text_lower:
            return "Block"
        elif "medium risk" in text_lower or "recommend review" in text_lower:
            return "Review"
        else:
            return "Approve"
    
    def generate_user_behavior_profile(self, user_id: str, historical_data: List[Dict]) -> Dict:
        if user_id in self.behavior_profiles:
            return self.behavior_profiles[user_id]
            
        purchase_amounts = [t['transaction_amount'] for t in historical_data]
        session_durations = [t['userBehavior']['sessionDuration'] for t in historical_data]
        cart_rates = [t['userBehavior']['addToCartRate'] for t in historical_data]
        
        profile = {
            "user_id": user_id,
            "first_seen": min(t['timestamp'] for t in historical_data),
            "last_seen": max(t['timestamp'] for t in historical_data),
            "total_transactions": len(historical_data),
            "avg_purchase_amount": np.mean(purchase_amounts) if purchase_amounts else 0,
            "median_purchase_amount": np.median(purchase_amounts) if purchase_amounts else 0,
            "avg_session_duration": np.mean(session_durations) if session_durations else 0,
            "avg_cart_rate": np.mean(cart_rates) if cart_rates else 0,
            "preferred_categories": self._calculate_preferred_categories(historical_data),
            "typical_purchase_times": self._calculate_typical_times(historical_data),
            "risk_score": self._calculate_historical_risk(historical_data)
        }
        
        self.behavior_profiles[user_id] = profile
        return profile
    
    def _calculate_preferred_categories(self, transactions: List[Dict]) -> Dict:
        categories = {}
        for t in transactions:
            for item in t['cart']:
                category = item.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + (item.get('quantity', 1))
        return categories
    
    def _calculate_typical_times(self, transactions: List[Dict]) -> Dict:
        times = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
        for t in transactions:
            hour = datetime.fromisoformat(t['timestamp']).hour
            if 5 <= hour < 12:
                times["morning"] += 1
            elif 12 <= hour < 17:
                times["afternoon"] += 1
            elif 17 <= hour < 22:
                times["evening"] += 1
            else:
                times["night"] += 1
        return times
    
    def _calculate_historical_risk(self, transactions: List[Dict]) -> float:
        if not transactions:
            return 50.0
            
        risk_scores = [t.get('risk_score', 0) for t in transactions]
        return float(np.mean(risk_scores))
    
    def detect_behavior_anomalies(self, current_behavior: Dict, user_profile: Dict) -> Dict:
        anomalies = {}
        threshold = 2.0
        
        current_amount = sum(item['price'] * item.get('quantity', 1) for item in current_behavior['cart'])
        if user_profile['avg_purchase_amount'] > 0:
            z_score = (current_amount - user_profile['avg_purchase_amount']) / max(1, user_profile['avg_purchase_amount'])
            if abs(z_score) > threshold:
                anomalies['purchase_amount'] = {
                    "current": current_amount,
                    "typical": user_profile['avg_purchase_amount'],
                    "deviation": f"{z_score:.1f} standard deviations"
                }
        
        current_duration = current_behavior['userBehavior']['sessionDuration']
        if user_profile['avg_session_duration'] > 0:
            z_score = (current_duration - user_profile['avg_session_duration']) / max(1, user_profile['avg_session_duration'])
            if abs(z_score) > threshold:
                anomalies['session_duration'] = {
                    "current": current_duration,
                    "typical": user_profile['avg_session_duration'],
                    "deviation": f"{z_score:.1f} standard deviations"
                }
        
        current_categories = {item.get('category', 'unknown') for item in current_behavior['cart']}
        preferred_categories = set(user_profile['preferred_categories'].keys())
        unusual_categories = current_categories - preferred_categories
        if unusual_categories and len(preferred_categories) > 0:
            anomalies['unusual_categories'] = list(unusual_categories)
        
        return anomalies
    
    def generate_fraud_explanation(self, transaction: Dict, risk_score: float, risk_factors: Dict) -> str:
        prompt = f"""
        You are a fraud analyst explaining this transaction risk to a customer support agent:
        
        Transaction Details:
        - Amount: ${sum(item['price'] * item.get('quantity', 1) for item in transaction['cart'])}
        - Items: {len(transaction['cart'])} products
        - Customer: {transaction['userInfo'].get('name', 'Unknown')}
        - Email: {transaction['userInfo'].get('email', 'Not provided')}
        - Location: {transaction['userInfo'].get('location', 'Unknown')}
        
        Risk Assessment:
        - Score: {risk_score}/100
        - Key Risk Factors: {', '.join(f"{k}: {v}" for k, v in risk_factors.items())}
        
        Please provide a concise, professional explanation of why this transaction was flagged,
        focusing on the most significant risk factors. Use simple language and suggest any
        verification steps that might help confirm the transaction's legitimacy.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful fraud analyst explaining transaction risks in clear, professional language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return "Unable to generate AI explanation. Please review manually."

# Initialize AI
ai = AIEnhancements(OPENAI_API_KEY)

# Initialize or load model
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
else:
    print("Training new fraud detection model...")
    model = IsolationForest(n_estimators=100, 
                          contamination=0.1, 
                          random_state=42,
                          verbose=1)
    scaler = StandardScaler()
    
    np.random.seed(42)
    n_samples = 1000
    X_train = np.column_stack([
        np.random.exponential(scale=500, size=n_samples),
        np.random.poisson(lam=3, size=n_samples),
        np.random.normal(loc=300, scale=100, size=n_samples),
        np.random.randint(0, 2, size=n_samples),
        np.random.beta(a=2, b=5, size=n_samples) * 100
    ])
    
    anomalies = np.column_stack([
        np.random.exponential(scale=5000, size=100),
        np.random.poisson(lam=15, size=100),
        np.random.normal(loc=30, scale=10, size=100),
        np.ones(100),
        np.random.beta(a=5, b=2, size=100) * 100
    ])
    X_train = np.vstack([X_train, anomalies])
    
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model.fit(X_train_scaled)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Model training complete and saved.")

def get_user_history(email: str) -> List[Dict]:
    """Retrieve user's transaction history from database"""
    if not os.path.exists(DATA_FILE):
        return []
    
    try:
        df = pd.read_csv(DATA_FILE)
        user_data = df[df['user_email'] == email].to_dict('records')
        return user_data
    except Exception as e:
        print(f"Error retrieving user history: {str(e)}")
        return []

def log_transaction(transaction: Dict, features: np.ndarray, risk_score: float, 
                   ai_analysis: Optional[Dict] = None, ai_anomalies: Optional[Dict] = None) -> None:
    """Log transaction data with AI analysis"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': hashlib.md5(json.dumps(transaction, sort_keys=True).encode()).hexdigest(),
            'user_email': transaction['userInfo']['email'],
            'transaction_amount': sum(item['price'] * item.get('quantity', 1) for item in transaction['cart']),
            'item_count': len(transaction['cart']),
            'high_value_items': sum(1 for item in transaction['cart'] if item['price'] > 500),
            'country': transaction['userInfo']['location'],
            'session_duration': transaction['userBehavior']['sessionDuration'],
            'add_to_cart_rate': transaction['userBehavior']['addToCartRate'],
            'features': features.tolist(),
            'risk_score': risk_score,
            'is_approved': risk_score < 70,
            'device_fingerprint': transaction['deviceFingerprint'],
            'ai_analysis': json.dumps(ai_analysis) if ai_analysis else None,
            'ai_anomalies': json.dumps(ai_anomalies) if ai_anomalies else None
        }
        
        df = pd.DataFrame([log_entry])
        
        if not os.path.exists(DATA_FILE):
            df.to_csv(DATA_FILE, index=False)
        else:
            df.to_csv(DATA_FILE, mode='a', header=False, index=False)
            
    except Exception as e:
        print(f"Error logging transaction: {str(e)}")

def calculate_features(transaction: Dict) -> np.ndarray:
    """Convert transaction data to ML features"""
    try:
        transaction_amount = sum(item['price'] * item.get('quantity', 1) for item in transaction['cart'])
        item_count = len(transaction['cart'])
        high_value_items = sum(1 for item in transaction['cart'] if item['price'] > 500)
        session_duration = transaction['userBehavior']['sessionDuration']
        add_to_cart_rate = transaction['userBehavior']['addToCartRate']
        
        high_risk_countries = ['RU', 'CN', 'NG', 'BR', 'VN', 'PK']
        is_high_risk_country = 1 if transaction['userInfo']['location'] in high_risk_countries else 0
        
        device_trust = 0 if transaction['deviceFingerprint'] == 'unknown' else 1
        
        features = np.array([
            transaction_amount,
            item_count,
            high_value_items,
            session_duration,
            add_to_cart_rate,
            is_high_risk_country,
            device_trust
        ]).reshape(1, -1)
        
        return features
        
    except Exception as e:
        print(f"Error calculating features: {str(e)}")
        return np.zeros((1, 7))

def generate_explanations(transaction: Dict, features: np.ndarray, risk_score: float) -> List[str]:
    """Generate human-readable explanations for the risk assessment"""
    explanations = []
    
    transaction_amount = features[0][0]
    item_count = features[0][1]
    high_value_items = features[0][2]
    session_duration = features[0][3]
    add_to_cart_rate = features[0][4]
    is_high_risk_country = features[0][5]
    
    if transaction_amount > 2000:
        explanations.append(f"High transaction amount (${transaction_amount:.2f})")
    elif transaction_amount > 1000:
        explanations.append(f"Medium transaction amount (${transaction_amount:.2f})")
    
    if item_count > 10:
        explanations.append(f"Large number of items ({item_count})")
    elif item_count > 5:
        explanations.append(f"Medium number of items ({item_count})")
    
    if high_value_items > 3:
        explanations.append(f"Multiple high-value items ({high_value_items})")
    elif high_value_items > 1:
        explanations.append(f"Several high-value items ({high_value_items})")
    
    if session_duration < 30:
        explanations.append(f"Very short session duration ({session_duration} seconds)")
    elif session_duration < 120:
        explanations.append(f"Short session duration ({session_duration} seconds)")
    
    if add_to_cart_rate > 10:
        explanations.append(f"High add-to-cart rate ({add_to_cart_rate})")
    
    if is_high_risk_country:
        explanations.append(f"High risk country ({transaction['userInfo']['location']})")
    
    if not explanations and risk_score < 30:
        explanations.append("No significant risk factors detected")
    
    return explanations

def determine_status(risk_score: float, ai_recommendation: str = "Approve") -> tuple:
    """Determine status and action based on risk score and AI recommendation"""
    if risk_score > 85 or ai_recommendation == "Block":
        return "High Risk", "Block"
    elif risk_score > 70 or ai_recommendation == "Review":
        return "Medium Risk", "Review"
    elif risk_score > 30:
        return "Low Risk", "Monitor"
    else:
        return "Very Low Risk", "Approve"

@app.route('/api/fraud-detection', methods=['POST'])
def detect_fraud():
    try:
        transaction = request.json
        
        # Existing ML model analysis
        features = calculate_features(transaction)
        features_scaled = scaler.transform(features)
        anomaly_score = model.decision_function(features_scaled)[0]
        risk_score = min(100, max(0, 50 - (anomaly_score * 100)))
        
        # AI-enhanced analysis
        ai_analysis = ai.analyze_transaction_text(transaction)
        user_history = get_user_history(transaction['userInfo']['email'])
        user_profile = ai.generate_user_behavior_profile(transaction['userInfo']['email'], user_history)
        ai_anomalies = ai.detect_behavior_anomalies(transaction, user_profile)
        
        # Combine explanations
        ml_explanations = generate_explanations(transaction, features, risk_score)
        ai_explanations = ai_analysis.get('risk_indicators', [])
        combined_explanations = ml_explanations + ai_explanations
        
        # Add AI explanation
        ai_explanation = ai.generate_fraud_explanation(transaction, risk_score, {
            "Transaction Amount": features[0][0],
            "Item Count": features[0][1],
            "High Value Items": features[0][2],
            "Session Duration": features[0][3],
            "Country Risk": "High" if features[0][5] else "Low"
        })
        
        # Determine final status
        status, action = determine_status(risk_score, ai_analysis.get('recommendation', 'Approve'))
        
        # Log the transaction with AI data
        log_transaction(transaction, features, risk_score, ai_analysis, ai_anomalies)
        
        # Prepare enhanced response
        response = {
            'success': True,
            'score': float(risk_score),
            'status': status,
            'action': action,
            'explanations': combined_explanations,
            'ai_explanation': ai_explanation,
            'ai_analysis': ai_analysis.get('analysis', 'No AI analysis available'),
            'ai_anomalies': ai_anomalies,
            'user_behavior_profile': user_profile,
            'features': {
                'transaction_amount': float(features[0][0]),
                'item_count': int(features[0][1]),
                'high_value_items': int(features[0][2]),
                'session_duration': float(features[0][3]),
                'add_to_cart_rate': float(features[0][4]),
                'is_high_risk_country': bool(features[0][5])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/retrain-model', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the model with new data"""
    try:
        if not os.path.exists(DATA_FILE):
            return jsonify({'success': False, 'error': 'No transaction data available'}), 400
            
        df = pd.read_csv(DATA_FILE)
        
        if len(df) < 100:
            return jsonify({'success': False, 'error': 'Insufficient data for retraining'}), 400
        
        X = df['features'].apply(eval).tolist()
        X = np.array(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Retraining fraud detection model...")
        model = IsolationForest(n_estimators=100, 
                              contamination=0.1, 
                              random_state=42,
                              verbose=1)
        model.fit(X_scaled)
        
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'training_samples': len(X),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)