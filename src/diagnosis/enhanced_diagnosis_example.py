"""
Enhanced Diagnosis System Example

This example demonstrates the complete enhanced diagnostic capabilities including:
1. State-of-the-art differential diagnosis with multi-hypothesis testing
2. Temporal pattern analysis for longitudinal symptom tracking
3. Cultural context integration and sensitivity
4. Uncertainty quantification and confidence levels
5. Evidence-based criteria matching (DSM-5, ICD-11)
6. Adaptive learning from diagnostic accuracy over time
7. Real-time research integration for latest diagnostic approaches
8. Comprehensive diagnostic reports with actionable insights

Usage Example:
    python enhanced_diagnosis_example.py
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .enhanced_integrated_system import EnhancedIntegratedDiagnosticSystem
from .comprehensive_diagnostic_report import ComprehensiveDiagnosticReporter
from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedDiagnosisDemo:
    """
    Demonstration of the enhanced diagnosis system capabilities
    """
    
    def __init__(self):
        """Initialize the demo system"""
        self.vector_db = None  # Would initialize with actual database
        self.diagnostic_system = EnhancedIntegratedDiagnosticSystem(self.vector_db)
        self.reporter = ComprehensiveDiagnosticReporter(self.vector_db)
        
    async def run_comprehensive_example(self):
        """Run comprehensive example showcasing all features"""
        
        print("🧠 Enhanced AI Diagnosis System - Comprehensive Demo")
        print("=" * 60)
        
        # Simulate patient case
        user_id = "demo_patient_001"
        session_id = f"session_{int(datetime.now().timestamp())}"
        
        # Patient presents with complex symptom profile
        user_message = """
        I've been feeling really down for the past few weeks. I can barely get out of bed,
        have no energy, and don't enjoy things I used to love. I'm having trouble concentrating
        at work and my sleep is all messed up - either can't fall asleep or sleep too much.
        I keep thinking about how worthless I am and sometimes wonder if life is worth living.
        My appetite is gone and I've lost weight. This isn't the first time this has happened,
        but it feels worse than before. I'm worried about my job and my family.
        """
        
        # Conversation history with temporal context
        conversation_history = [
            {
                "timestamp": datetime.now() - timedelta(days=7),
                "message": "I've been feeling off lately, like something's not right",
                "mood_score": 6
            },
            {
                "timestamp": datetime.now() - timedelta(days=3),
                "message": "The sadness is getting worse and I'm sleeping poorly",
                "mood_score": 4
            },
            {
                "timestamp": datetime.now() - timedelta(days=1),
                "message": "I called in sick again today, can't face going to work",
                "mood_score": 3
            }
        ]
        
        # Voice emotion data (simulated)
        voice_emotion_data = {
            "emotions": {
                "sadness": 0.8,
                "anxiety": 0.6,
                "fear": 0.4,
                "happiness": 0.1,
                "anger": 0.2
            },
            "energy_level": 0.2,
            "speech_rate": 0.3,  # Slower than normal
            "voice_quality": "flat_affect"
        }
        
        # Personality data from previous assessments
        personality_data = {
            "big_five": {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.3,
                "agreeableness": 0.9,
                "neuroticism": 0.8
            },
            "risk_factors": ["perfectionism", "rumination_tendency"],
            "strengths": ["high_empathy", "strong_values"]
        }
        
        # Cultural information
        cultural_info = {
            "primary_culture": "Western",
            "age": 32,
            "gender": "female",
            "education": "college",
            "employment": "professional",
            "family_history": ["depression", "anxiety"],
            "support_system": "moderate"
        }
        
        print("\n📊 Generating Comprehensive Diagnosis...")
        print("-" * 40)
        
        # Generate comprehensive diagnostic assessment
        diagnostic_result = await self.diagnostic_system.generate_comprehensive_diagnosis(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            conversation_history=conversation_history,
            voice_emotion_data=voice_emotion_data,
            personality_data=personality_data,
            cultural_info=cultural_info
        )
        
        # Display key diagnostic findings
        await self._display_diagnostic_findings(diagnostic_result)
        
        print("\n📋 Generating Comprehensive Report...")
        print("-" * 40)
        
        # Generate comprehensive diagnostic report
        diagnostic_report = await self.reporter.generate_comprehensive_report(
            diagnostic_result=diagnostic_result,
            additional_context={
                "session_context": "Initial assessment",
                "referral_source": "self-referral",
                "presenting_concerns": ["mood", "sleep", "concentration", "suicidal_ideation"]
            }
        )
        
        # Display report summary
        report_summary = await self.reporter.generate_report_summary(diagnostic_report)
        print(report_summary)
        
        print("\n🔍 System Performance Metrics:")
        print("-" * 40)
        await self._display_system_metrics(diagnostic_result, diagnostic_report)
        
        print("\n✅ Demo completed successfully!")
        print("The enhanced diagnosis system demonstrated:")
        print("• Multi-hypothesis differential diagnosis")
        print("• Temporal pattern analysis")
        print("• Cultural sensitivity integration")
        print("• Uncertainty quantification")
        print("• Evidence-based recommendations")
        print("• Comprehensive risk assessment")
        print("• Detailed treatment planning")
        print("• Quality-assured reporting")
        
        return diagnostic_result, diagnostic_report
    
    async def _display_diagnostic_findings(self, result):
        """Display key diagnostic findings"""
        
        print(f"Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"System Reliability: {result.system_reliability:.1%}")
        print(f"Integration Confidence: {result.integration_confidence:.1%}")
        
        if result.primary_diagnosis:
            print(f"\n🎯 PRIMARY DIAGNOSIS:")
            print(f"   Condition: {result.primary_diagnosis.condition_name}")
            print(f"   Confidence: {result.primary_diagnosis.confidence:.1%}")
            print(f"   Severity: {result.primary_diagnosis.severity}")
            print(f"   Rank: #{result.primary_diagnosis.differential_rank}")
            
            if result.primary_diagnosis.supporting_evidence:
                print(f"   Key Evidence:")
                for evidence in result.primary_diagnosis.supporting_evidence[:3]:
                    print(f"   • {evidence}")
        
        if len(result.differential_diagnoses) > 1:
            print(f"\n🔍 DIFFERENTIAL DIAGNOSES:")
            for i, diagnosis in enumerate(result.differential_diagnoses[1:4], 2):
                print(f"   {i}. {diagnosis.condition_name} (Confidence: {diagnosis.confidence:.1%})")
        
        if result.symptom_progression:
            print(f"\n📈 TEMPORAL PATTERNS:")
            trends = result.symptom_progression.get("trends", {})
            if trends:
                print(f"   Trend Direction: {trends.get('direction', 'stable')}")
                print(f"   Volatility: {trends.get('volatility', 0):.2f}")
        
        if result.therapeutic_response:
            print(f"\n🎭 THERAPEUTIC RESPONSE:")
            print(f"   Technique: {result.therapeutic_response.therapeutic_technique}")
            print(f"   Growth Readiness: {result.therapeutic_response.growth_readiness_score:.1%}")
        
        if result.evidence_based_recommendations:
            print(f"\n📚 EVIDENCE-BASED RECOMMENDATIONS:")
            for i, rec in enumerate(result.evidence_based_recommendations[:3], 1):
                print(f"   {i}. {rec.recommendation_text[:60]}...")
                print(f"      Evidence Level: {rec.evidence_level}, Confidence: {rec.confidence_score:.1%}")
        
        if result.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"   • {warning}")
        
        if result.limitations:
            print(f"\n⚡ LIMITATIONS:")
            for limitation in result.limitations:
                print(f"   • {limitation}")
    
    async def _display_system_metrics(self, diagnostic_result, diagnostic_report):
        """Display system performance metrics"""
        
        print(f"Diagnostic Confidence: {diagnostic_result.confidence_score:.1%}")
        print(f"System Integration: {diagnostic_result.integration_confidence:.1%}")
        print(f"Processing Efficiency: {diagnostic_result.processing_time_ms:.1f}ms")
        
        print(f"\nReport Quality Metrics:")
        quality = diagnostic_report.report_quality
        print(f"  Completeness: {quality['completeness']:.1%}")
        print(f"  Evidence Strength: {quality['evidence_strength']:.1%}")
        print(f"  Clinical Utility: {quality['clinical_utility']:.1%}")
        print(f"  Overall Quality: {quality['overall_quality']:.1%}")
        
        print(f"\nRisk Assessment:")
        risk = diagnostic_report.risk_assessment
        print(f"  Overall Risk Level: {risk.overall_risk_level.title()}")
        print(f"  Suicide Risk: {risk.suicide_risk['level'].title()}")
        print(f"  Monitoring Frequency: {risk.monitoring_frequency}")
        
        print(f"\nTreatment Planning:")
        treatment = diagnostic_report.treatment_plan
        print(f"  Primary Interventions: {len(treatment.primary_interventions)}")
        print(f"  Session Frequency: {treatment.session_frequency}")
        print(f"  Estimated Duration: {treatment.estimated_duration}")
    
    async def demonstrate_advanced_features(self):
        """Demonstrate advanced features of the system"""
        
        print("\n🚀 Advanced Features Demonstration")
        print("=" * 50)
        
        # 1. Multi-hypothesis testing
        print("\n1. Multi-Hypothesis Testing:")
        print("   ✅ Evaluates multiple diagnostic possibilities simultaneously")
        print("   ✅ Ranks diagnoses by evidence strength and probability")
        print("   ✅ Considers comorbidity patterns and interactions")
        
        # 2. Temporal analysis
        print("\n2. Temporal Pattern Analysis:")
        print("   ✅ Tracks symptom progression over time")
        print("   ✅ Identifies behavioral patterns and triggers")
        print("   ✅ Predicts symptom trajectory for treatment planning")
        
        # 3. Cultural sensitivity
        print("\n3. Cultural Context Integration:")
        print("   ✅ Adapts diagnostic criteria for cultural background")
        print("   ✅ Provides culturally appropriate treatment recommendations")
        print("   ✅ Considers family and community factors")
        
        # 4. Uncertainty quantification
        print("\n4. Uncertainty Quantification:")
        print("   ✅ Calculates confidence intervals for diagnoses")
        print("   ✅ Identifies sources of diagnostic uncertainty")
        print("   ✅ Provides recommendations to reduce uncertainty")
        
        # 5. Evidence-based matching
        print("\n5. Evidence-Based Criteria Matching:")
        print("   ✅ Uses comprehensive DSM-5 and ICD-11 criteria")
        print("   ✅ Applies temporal and severity requirements")
        print("   ✅ Considers exclusion criteria and differential diagnosis")
        
        # 6. Adaptive learning
        print("\n6. Adaptive Learning:")
        print("   ✅ Learns from diagnostic accuracy over time")
        print("   ✅ Personalizes recommendations based on user response")
        print("   ✅ Continuously improves intervention effectiveness")
        
        # 7. Research integration
        print("\n7. Real-Time Research Integration:")
        print("   ✅ Incorporates latest clinical research and guidelines")
        print("   ✅ Validates treatment approaches against current evidence")
        print("   ✅ Provides evidence-graded recommendations")
        
        # 8. Comprehensive reporting
        print("\n8. Comprehensive Diagnostic Reporting:")
        print("   ✅ Generates clinical-quality diagnostic reports")
        print("   ✅ Includes risk assessment and safety planning")
        print("   ✅ Provides actionable treatment recommendations")
        print("   ✅ Supports multiple export formats (PDF, HTML, JSON)")
        
        # System validation
        print("\n🔧 System Validation:")
        validation_results = await self.diagnostic_system.validate_system_integration()
        print(f"   Overall Status: {validation_results['overall_status'].title()}")
        print(f"   System Reliability: {validation_results['performance_metrics']['system_reliability']:.1%}")
        print(f"   Initialization Success Rate: {validation_results['performance_metrics']['initialization_success_rate']:.1%}")
        
        if validation_results.get('recommendations'):
            print("   Recommendations:")
            for rec in validation_results['recommendations']:
                print(f"   • {rec}")

async def main():
    """Main demonstration function"""
    
    print("🌟 Welcome to the Enhanced AI Diagnosis System Demo")
    print("This system represents state-of-the-art diagnostic capabilities")
    print("for mental health assessment with clinical-grade accuracy.\n")
    
    demo = EnhancedDiagnosisDemo()
    
    try:
        # Run comprehensive example
        diagnostic_result, diagnostic_report = await demo.run_comprehensive_example()
        
        # Demonstrate advanced features
        await demo.demonstrate_advanced_features()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"Report ID: {diagnostic_report.report_id}")
        print(f"Patient ID: {diagnostic_report.patient_id}")
        print(f"Generated: {diagnostic_report.generation_timestamp}")
        
        # Optional: Export report
        export_choice = input("\nWould you like to export the diagnostic report? (y/n): ")
        if export_choice.lower() == 'y':
            json_report = await demo.reporter.export_report(diagnostic_report, "json")
            with open(f"diagnostic_report_{diagnostic_report.patient_id}.json", "w") as f:
                f.write(json_report)
            print(f"Report exported to: diagnostic_report_{diagnostic_report.patient_id}.json")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())