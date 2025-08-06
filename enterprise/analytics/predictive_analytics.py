"""
Predictive Analytics Engine for Mental Health Outcomes
Implements machine learning models for treatment prediction and risk assessment
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class OutcomeType(Enum):
    TREATMENT_RESPONSE = "treatment_response"
    DROPOUT_RISK = "dropout_risk"
    CRISIS_RISK = "crisis_risk"
    RECOVERY_TIME = "recovery_time"
    RELAPSE_RISK = "relapse_risk"
    TREATMENT_ADHERENCE = "treatment_adherence"


@dataclass
class PatientFeatures:
    """Patient feature vector for ML models"""
    patient_id: str
    age: int
    gender: str
    diagnosis_primary: str
    diagnosis_secondary: List[str]
    severity_score: float
    previous_episodes: int
    treatment_history: List[str]
    medication_history: List[str]
    social_support_score: float
    employment_status: str
    education_level: str
    substance_use: bool
    comorbidities: List[str]
    session_count: int
    engagement_score: float
    therapeutic_alliance_score: float
    homework_completion_rate: float
    session_attendance_rate: float
    emotional_regulation_score: float
    coping_skills_score: float
    motivation_score: float
    baseline_phq9: Optional[float] = None
    baseline_gad7: Optional[float] = None
    baseline_dass21: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictionResult:
    """Result of a prediction model"""
    patient_id: str
    outcome_type: OutcomeType
    prediction: Union[float, str, bool]
    probability: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    risk_level: RiskLevel
    contributing_factors: List[Tuple[str, float]]
    recommendations: List[str]
    model_version: str
    prediction_date: datetime = field(default_factory=datetime.utcnow)
    explanation: str = ""


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    outcome_type: OutcomeType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    cross_val_score: float
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
    training_size: int
    test_size: int
    last_trained: datetime
    validation_date: datetime = field(default_factory=datetime.utcnow)


class FeatureEngineer:
    """Feature engineering for predictive models"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def engineer_features(self, patient_features: PatientFeatures) -> Dict[str, float]:
        """Convert patient features to model-ready format"""
        features = {}
        
        # Demographic features
        features['age'] = patient_features.age
        features['age_squared'] = patient_features.age ** 2
        features['gender_encoded'] = self._encode_categorical('gender', patient_features.gender)
        
        # Clinical features
        features['severity_score'] = patient_features.severity_score
        features['previous_episodes'] = patient_features.previous_episodes
        features['previous_episodes_log'] = np.log1p(patient_features.previous_episodes)
        
        # Treatment features
        features['treatment_history_count'] = len(patient_features.treatment_history)
        features['medication_history_count'] = len(patient_features.medication_history)
        features['has_previous_treatment'] = float(len(patient_features.treatment_history) > 0)
        features['has_medication_history'] = float(len(patient_features.medication_history) > 0)
        
        # Psychosocial features
        features['social_support_score'] = patient_features.social_support_score
        features['employment_encoded'] = self._encode_categorical('employment', patient_features.employment_status)
        features['education_encoded'] = self._encode_categorical('education', patient_features.education_level)
        features['substance_use'] = float(patient_features.substance_use)
        
        # Clinical scores
        features['session_count'] = patient_features.session_count
        features['engagement_score'] = patient_features.engagement_score
        features['therapeutic_alliance_score'] = patient_features.therapeutic_alliance_score
        features['homework_completion_rate'] = patient_features.homework_completion_rate
        features['session_attendance_rate'] = patient_features.session_attendance_rate
        
        # Psychological features
        features['emotional_regulation_score'] = patient_features.emotional_regulation_score
        features['coping_skills_score'] = patient_features.coping_skills_score
        features['motivation_score'] = patient_features.motivation_score
        
        # Baseline assessments
        if patient_features.baseline_phq9:
            features['baseline_phq9'] = patient_features.baseline_phq9
            features['baseline_phq9_severe'] = float(patient_features.baseline_phq9 >= 20)
            
        if patient_features.baseline_gad7:
            features['baseline_gad7'] = patient_features.baseline_gad7
            features['baseline_gad7_severe'] = float(patient_features.baseline_gad7 >= 15)
            
        if patient_features.baseline_dass21:
            features['baseline_dass21'] = patient_features.baseline_dass21
            
        # Derived features
        features['comorbidity_count'] = len(patient_features.comorbidities)
        features['has_comorbidity'] = float(len(patient_features.comorbidities) > 0)
        
        # Interaction features
        features['engagement_x_alliance'] = features['engagement_score'] * features['therapeutic_alliance_score']
        features['attendance_x_homework'] = features['session_attendance_rate'] * features['homework_completion_rate']
        features['severity_x_support'] = features['severity_score'] * features['social_support_score']
        
        # Risk factors
        features['high_severity'] = float(patient_features.severity_score > 7)
        features['low_support'] = float(patient_features.social_support_score < 3)
        features['poor_engagement'] = float(patient_features.engagement_score < 5)
        
        return features
        
    def _encode_categorical(self, category: str, value: str) -> float:
        """Encode categorical variables"""
        if category not in self.encoders:
            self.encoders[category] = LabelEncoder()
            
        try:
            # For new categories, assign a default value
            if hasattr(self.encoders[category], 'classes_'):
                if value not in self.encoders[category].classes_:
                    return 0.0
                return float(self.encoders[category].transform([value])[0])
            else:
                # First time encoding
                return 0.0
        except:
            return 0.0
            
    def scale_features(self, features: Dict[str, float], fit: bool = False) -> Dict[str, float]:
        """Scale numerical features"""
        if fit or 'main' not in self.scalers:
            self.scalers['main'] = StandardScaler()
            
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        if fit:
            scaled_array = self.scalers['main'].fit_transform(feature_array)
        else:
            scaled_array = self.scalers['main'].transform(feature_array)
            
        scaled_features = dict(zip(features.keys(), scaled_array[0]))
        return scaled_features


class TreatmentResponsePredictor:
    """Predict treatment response probability"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.feature_names = []
        
    async def train(self, training_data: List[Tuple[PatientFeatures, bool]]) -> ModelPerformance:
        """Train the treatment response model"""
        logger.info(f"Training treatment response model with {len(training_data)} samples")
        
        # Engineer features
        X = []
        y = []
        
        for patient_features, outcome in training_data:
            features = self.feature_engineer.engineer_features(patient_features)
            scaled_features = self.feature_engineer.scale_features(features, fit=len(X) == 0)
            
            X.append(list(scaled_features.values()))
            y.append(outcome)
            
            if not self.feature_names:
                self.feature_names = list(scaled_features.keys())
                
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        performance = ModelPerformance(
            model_name="TreatmentResponsePredictor",
            outcome_type=OutcomeType.TREATMENT_RESPONSE,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            cross_val_score=cv_scores.mean(),
            feature_importance=feature_importance,
            confusion_matrix=[[0, 0], [0, 0]],  # Simplified
            training_size=len(X_train),
            test_size=len(X_test),
            last_trained=datetime.utcnow()
        )
        
        logger.info(f"Model trained. Accuracy: {accuracy:.3f}, AUC-ROC: {auc_roc:.3f}")
        return performance
        
    async def predict(self, patient_features: PatientFeatures) -> PredictionResult:
        """Predict treatment response for a patient"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Engineer features
        features = self.feature_engineer.engineer_features(patient_features)
        scaled_features = self.feature_engineer.scale_features(features)
        
        # Prepare for prediction
        X = np.array([list(scaled_features.values())])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]  # Probability of positive response
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = RiskLevel.VERY_LOW
        elif probability >= 0.6:
            risk_level = RiskLevel.LOW
        elif probability >= 0.4:
            risk_level = RiskLevel.MODERATE
        elif probability >= 0.2:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.VERY_HIGH
            
        # Get feature importance for explanation
        feature_importance = list(zip(self.feature_names, self.model.feature_importances_))
        contributing_factors = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:5]
        
        # Generate recommendations
        recommendations = self._generate_treatment_recommendations(patient_features, probability)
        
        return PredictionResult(
            patient_id=patient_features.patient_id,
            outcome_type=OutcomeType.TREATMENT_RESPONSE,
            prediction=bool(prediction),
            probability=probability,
            risk_level=risk_level,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            model_version="1.0",
            explanation=f"Probability of positive treatment response: {probability:.2f}"
        )
        
    def _generate_treatment_recommendations(self, patient_features: PatientFeatures, 
                                          probability: float) -> List[str]:
        """Generate treatment recommendations based on prediction"""
        recommendations = []
        
        if probability < 0.4:  # Low probability of response
            if patient_features.engagement_score < 5:
                recommendations.append("Focus on improving patient engagement and motivation")
            if patient_features.therapeutic_alliance_score < 6:
                recommendations.append("Work on strengthening therapeutic alliance")
            if patient_features.social_support_score < 3:
                recommendations.append("Consider involving family/social support in treatment")
            if patient_features.homework_completion_rate < 0.5:
                recommendations.append("Simplify homework assignments and increase support")
                
        elif probability < 0.6:  # Moderate probability
            recommendations.append("Continue current treatment with regular monitoring")
            if patient_features.session_attendance_rate < 0.8:
                recommendations.append("Address barriers to session attendance")
                
        else:  # High probability of response
            recommendations.append("Current treatment approach is likely to be effective")
            recommendations.append("Consider setting more ambitious treatment goals")
            
        return recommendations


class CrisisRiskPredictor:
    """Predict crisis risk for patients"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.feature_names = []
        
    async def train(self, training_data: List[Tuple[PatientFeatures, float]]) -> ModelPerformance:
        """Train the crisis risk model"""
        logger.info(f"Training crisis risk model with {len(training_data)} samples")
        
        # Engineer features
        X = []
        y = []
        
        for patient_features, risk_score in training_data:
            features = self.feature_engineer.engineer_features(patient_features)
            scaled_features = self.feature_engineer.scale_features(features, fit=len(X) == 0)
            
            X.append(list(scaled_features.values()))
            y.append(risk_score)
            
            if not self.feature_names:
                self.feature_names = list(scaled_features.keys())
                
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics (for regression)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        performance = ModelPerformance(
            model_name="CrisisRiskPredictor",
            outcome_type=OutcomeType.CRISIS_RISK,
            accuracy=r2,  # Using R² as accuracy measure
            precision=mae,  # Using MAE as precision measure
            recall=rmse,   # Using RMSE as recall measure
            f1_score=cv_scores.mean(),
            auc_roc=None,
            cross_val_score=cv_scores.mean(),
            feature_importance=feature_importance,
            confusion_matrix=[[0, 0], [0, 0]],
            training_size=len(X_train),
            test_size=len(X_test),
            last_trained=datetime.utcnow()
        )
        
        logger.info(f"Crisis risk model trained. R²: {r2:.3f}, RMSE: {rmse:.3f}")
        return performance
        
    async def predict(self, patient_features: PatientFeatures) -> PredictionResult:
        """Predict crisis risk for a patient"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Engineer features
        features = self.feature_engineer.engineer_features(patient_features)
        scaled_features = self.feature_engineer.scale_features(features)
        
        # Prepare for prediction
        X = np.array([list(scaled_features.values())])
        
        # Make prediction
        risk_score = self.model.predict(X)[0]
        risk_score = max(0, min(10, risk_score))  # Clamp to 0-10 range
        
        # Determine risk level
        if risk_score >= 8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 6:
            risk_level = RiskLevel.VERY_HIGH
        elif risk_score >= 4:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 2:
            risk_level = RiskLevel.MODERATE
        elif risk_score >= 1:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.VERY_LOW
            
        # Get feature importance for explanation
        feature_importance = list(zip(self.feature_names, self.model.feature_importances_))
        contributing_factors = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:5]
        
        # Generate recommendations
        recommendations = self._generate_crisis_recommendations(patient_features, risk_score)
        
        return PredictionResult(
            patient_id=patient_features.patient_id,
            outcome_type=OutcomeType.CRISIS_RISK,
            prediction=risk_score,
            probability=risk_score / 10.0,  # Normalize to probability
            risk_level=risk_level,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            model_version="1.0",
            explanation=f"Crisis risk score: {risk_score:.2f}/10"
        )
        
    def _generate_crisis_recommendations(self, patient_features: PatientFeatures, 
                                       risk_score: float) -> List[str]:
        """Generate crisis intervention recommendations"""
        recommendations = []
        
        if risk_score >= 6:  # High risk
            recommendations.append("URGENT: Implement immediate safety planning")
            recommendations.append("Consider emergency intervention or hospitalization")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Activate crisis response team")
            
        elif risk_score >= 4:  # Moderate risk
            recommendations.append("Develop comprehensive safety plan")
            recommendations.append("Increase session frequency")
            recommendations.append("Consider medication review")
            recommendations.append("Enhance social support network")
            
        else:  # Lower risk
            recommendations.append("Continue regular monitoring")
            recommendations.append("Maintain crisis prevention strategies")
            
        # Specific recommendations based on features
        if patient_features.substance_use:
            recommendations.append("Address substance use as crisis risk factor")
            
        if patient_features.social_support_score < 3:
            recommendations.append("Strengthen social support network")
            
        if len(patient_features.comorbidities) > 2:
            recommendations.append("Coordinate care for multiple conditions")
            
        return recommendations


class PredictiveAnalyticsEngine:
    """Main predictive analytics engine"""
    
    def __init__(self):
        self.treatment_predictor = TreatmentResponsePredictor()
        self.crisis_predictor = CrisisRiskPredictor()
        self.models = {
            OutcomeType.TREATMENT_RESPONSE: self.treatment_predictor,
            OutcomeType.CRISIS_RISK: self.crisis_predictor
        }
        self.model_performances: Dict[OutcomeType, ModelPerformance] = {}
        
    async def train_models(self, training_data: Dict[OutcomeType, List[Tuple[PatientFeatures, Any]]]) -> Dict[OutcomeType, ModelPerformance]:
        """Train all predictive models"""
        performances = {}
        
        for outcome_type, data in training_data.items():
            if outcome_type in self.models and data:
                try:
                    performance = await self.models[outcome_type].train(data)
                    performances[outcome_type] = performance
                    self.model_performances[outcome_type] = performance
                    logger.info(f"Successfully trained model for {outcome_type.value}")
                except Exception as e:
                    logger.error(f"Failed to train model for {outcome_type.value}: {e}")
                    
        return performances
        
    async def predict_outcomes(self, patient_features: PatientFeatures, 
                             outcome_types: List[OutcomeType] = None) -> List[PredictionResult]:
        """Generate predictions for specified outcome types"""
        if outcome_types is None:
            outcome_types = list(self.models.keys())
            
        predictions = []
        
        for outcome_type in outcome_types:
            if outcome_type in self.models:
                try:
                    prediction = await self.models[outcome_type].predict(patient_features)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Failed to predict {outcome_type.value} for patient {patient_features.patient_id}: {e}")
                    
        return predictions
        
    async def batch_predict(self, patients: List[PatientFeatures], 
                          outcome_types: List[OutcomeType] = None) -> Dict[str, List[PredictionResult]]:
        """Generate predictions for multiple patients"""
        batch_results = {}
        
        for patient in patients:
            predictions = await self.predict_outcomes(patient, outcome_types)
            batch_results[patient.patient_id] = predictions
            
        return batch_results
        
    async def get_model_performance(self, outcome_type: OutcomeType) -> Optional[ModelPerformance]:
        """Get performance metrics for a specific model"""
        return self.model_performances.get(outcome_type)
        
    async def validate_models(self, validation_data: Dict[OutcomeType, List[Tuple[PatientFeatures, Any]]]) -> Dict[OutcomeType, Dict[str, float]]:
        """Validate models on new data"""
        validation_results = {}
        
        for outcome_type, data in validation_data.items():
            if outcome_type in self.models and data:
                try:
                    # Generate predictions
                    predictions = []
                    actuals = []
                    
                    for patient_features, actual_outcome in data:
                        prediction_result = await self.models[outcome_type].predict(patient_features)
                        predictions.append(prediction_result.prediction)
                        actuals.append(actual_outcome)
                        
                    # Calculate validation metrics
                    if outcome_type == OutcomeType.TREATMENT_RESPONSE:
                        # Classification metrics
                        accuracy = accuracy_score(actuals, predictions)
                        precision = precision_score(actuals, predictions)
                        recall = recall_score(actuals, predictions)
                        f1 = f1_score(actuals, predictions)
                        
                        validation_results[outcome_type] = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1
                        }
                    elif outcome_type == OutcomeType.CRISIS_RISK:
                        # Regression metrics
                        mse = np.mean((np.array(actuals) - np.array(predictions)) ** 2)
                        rmse = np.sqrt(mse)
                        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
                        
                        validation_results[outcome_type] = {
                            "mse": mse,
                            "rmse": rmse,
                            "mae": mae
                        }
                        
                except Exception as e:
                    logger.error(f"Failed to validate model for {outcome_type.value}: {e}")
                    
        return validation_results
        
    async def get_population_insights(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Generate population-level insights from predictions"""
        if not predictions:
            return {"error": "No predictions provided"}
            
        # Group by outcome type
        outcome_groups = {}
        for pred in predictions:
            if pred.outcome_type not in outcome_groups:
                outcome_groups[pred.outcome_type] = []
            outcome_groups[pred.outcome_type].append(pred)
            
        insights = {}
        
        for outcome_type, preds in outcome_groups.items():
            if outcome_type == OutcomeType.TREATMENT_RESPONSE:
                positive_responses = sum(1 for p in preds if p.prediction)
                insights[outcome_type.value] = {
                    "total_patients": len(preds),
                    "predicted_positive_response": positive_responses,
                    "positive_response_rate": positive_responses / len(preds),
                    "average_probability": np.mean([p.probability for p in preds if p.probability]),
                    "high_probability_patients": sum(1 for p in preds if p.probability and p.probability > 0.7)
                }
                
            elif outcome_type == OutcomeType.CRISIS_RISK:
                risk_scores = [p.prediction for p in preds if isinstance(p.prediction, (int, float))]
                high_risk_count = sum(1 for p in preds if p.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL])
                
                insights[outcome_type.value] = {
                    "total_patients": len(preds),
                    "average_risk_score": np.mean(risk_scores) if risk_scores else 0,
                    "high_risk_patients": high_risk_count,
                    "critical_risk_patients": sum(1 for p in preds if p.risk_level == RiskLevel.CRITICAL),
                    "risk_distribution": {
                        level.value: sum(1 for p in preds if p.risk_level == level)
                        for level in RiskLevel
                    }
                }
                
        return insights
        
    async def generate_alerts(self, predictions: List[PredictionResult]) -> List[Dict[str, Any]]:
        """Generate alerts based on predictions"""
        alerts = []
        
        for prediction in predictions:
            # Crisis risk alerts
            if (prediction.outcome_type == OutcomeType.CRISIS_RISK and 
                prediction.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]):
                
                alerts.append({
                    "type": "crisis_risk",
                    "patient_id": prediction.patient_id,
                    "risk_level": prediction.risk_level.value,
                    "risk_score": prediction.prediction,
                    "message": f"Patient {prediction.patient_id} has {prediction.risk_level.value} crisis risk",
                    "recommendations": prediction.recommendations,
                    "created_at": datetime.utcnow().isoformat()
                })
                
            # Treatment response alerts
            elif (prediction.outcome_type == OutcomeType.TREATMENT_RESPONSE and 
                  prediction.probability and prediction.probability < 0.3):
                
                alerts.append({
                    "type": "poor_treatment_response",
                    "patient_id": prediction.patient_id,
                    "response_probability": prediction.probability,
                    "message": f"Patient {prediction.patient_id} has low probability of treatment response",
                    "recommendations": prediction.recommendations,
                    "created_at": datetime.utcnow().isoformat()
                })
                
        return alerts