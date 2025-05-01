import openai
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json

class AIEnhancements:
    def _init_(self, api_key: str):
        """
        Initialize the AI enhancements module with OpenAI API key
        
        Args:
            api_key: OpenAI API key for accessing GPT models
        """
        openai.api_key = api_key
        self.behavior_profiles = {}
        self.fraud_patterns = self._load_known_patterns()
        
    def _load_known_patterns(self) -> Dict:
        """Load known fraud patterns from a JSON file or return default patterns"""
        try:
            with open('fraud_patterns.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default patterns if file doesn't exist
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
        """
        Use AI to analyze textual transaction data for potential fraud indicators
        
        Args:
            transaction_data: Dictionary containing transaction details
            
        Returns:
            Dictionary with AI analysis results
        """
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
        """Create a detailed prompt for the AI analysis"""
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
        """Extract key risk indicators from the AI analysis text"""
        # This can be enhanced with more sophisticated NLP
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
        """Generate a recommendation based on the AI analysis"""
        text_lower = analysis_text.lower()
        
        if "high risk" in text_lower or "recommend block" in text_lower:
            return "Block"
        elif "medium risk" in text_lower or "recommend review" in text_lower:
            return "Review"
        else:
            return "Approve"
    
    def generate_user_behavior_profile(self, user_id: str, historical_data: List[Dict]) -> Dict:
        """
        Create a comprehensive behavior profile for a user based on historical data
        
        Args:
            user_id: Unique identifier for the user
            historical_data: List of previous transactions and behaviors
            
        Returns:
            Dictionary with comprehensive behavior profile
        """
        if user_id in self.behavior_profiles:
            return self.behavior_profiles[user_id]
            
        # Calculate basic statistics
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
        """Calculate the user's preferred product categories"""
        categories = {}
        for t in transactions:
            for item in t['cart']:
                category = item.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + (item.get('quantity', 1))
        return categories
    
    def _calculate_typical_times(self, transactions: List[Dict]) -> Dict:
        """Calculate when the user typically makes purchases"""
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
        """Calculate a historical risk score based on past behavior"""
        if not transactions:
            return 50.0  # Neutral score for new users
            
        risk_scores = [t.get('risk_score', 0) for t in transactions]
        return float(np.mean(risk_scores))
    
    def detect_behavior_anomalies(self, current_behavior: Dict, user_profile: Dict) -> Dict:
        """
        Detect anomalies in current behavior compared to user's historical profile
        
        Args:
            current_behavior: Current transaction behavior data
            user_profile: User's historical behavior profile
            
        Returns:
            Dictionary with anomaly detection results
        """
        anomalies = {}
        threshold = 2.0  # Standard deviations for anomaly detection
        
        # Check purchase amount
        current_amount = sum(item['price'] * item.get('quantity', 1) for item in current_behavior['cart'])
        if user_profile['avg_purchase_amount'] > 0:
            z_score = (current_amount - user_profile['avg_purchase_amount']) / max(1, user_profile['avg_purchase_amount'])
            if abs(z_score) > threshold:
                anomalies['purchase_amount'] = {
                    "current": current_amount,
                    "typical": user_profile['avg_purchase_amount'],
                    "deviation": f"{z_score:.1f} standard deviations"
                }
        
        # Check session duration
        current_duration = current_behavior['userBehavior']['sessionDuration']
        if user_profile['avg_session_duration'] > 0:
            z_score = (current_duration - user_profile['avg_session_duration']) / max(1, user_profile['avg_session_duration'])
            if abs(z_score) > threshold:
                anomalies['session_duration'] = {
                    "current": current_duration,
                    "typical": user_profile['avg_session_duration'],
                    "deviation": f"{z_score:.1f} standard deviations"
                }
        
        # Check categories (simple version)
        current_categories = {item.get('category', 'unknown') for item in current_behavior['cart']}
        preferred_categories = set(user_profile['preferred_categories'].keys())
        unusual_categories = current_categories - preferred_categories
        if unusual_categories and len(preferred_categories) > 0:
            anomalies['unusual_categories'] = list(unusual_categories)
        
        return anomalies
    
    def generate_fraud_explanation(self, transaction: Dict, risk_score: float, risk_factors: Dict) -> str:
        """
        Generate a human-readable explanation of fraud risk using AI
        
        Args:
            transaction: Transaction data
            risk_score: Calculated risk score
            risk_factors: Dictionary of risk factors
            
        Returns:
            String with natural language explanation
        """
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