"""
Enterprise Multi-Modal Diagnostic Pipeline - Usage Examples and Integration Guide

This file demonstrates how to use the enterprise diagnostic pipeline with various
input types and configurations. It provides comprehensive examples for:

1. Basic usage with text input
2. Multi-modal input processing
3. Real-time streaming analysis
4. Batch processing for multiple users
5. Integration with existing Solace-AI components
6. Monitoring and A/B testing
7. Clinical workflow integration

Author: Solace-AI Development Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

# Import the enterprise pipeline
from .enterprise_multimodal_pipeline import (
    EnterpriseMultiModalDiagnosticPipeline,
    IntegratedDiagnosticSystem,
    create_enterprise_pipeline,
    ModalityType,
    ClinicalSeverity
)

# Import existing Solace-AI components for integration
from ..components.voice_component import VoiceComponent
from ..analysis.conversation_analysis import ConversationAnalyzer
from ..personality.big_five import BigFiveAssessment
from ..utils.logger import get_logger

logger = get_logger(__name__)

class EnterpriseUsageExamples:
    """
    Comprehensive examples of using the Enterprise Multi-Modal Diagnostic Pipeline
    """
    
    def __init__(self):
        self.pipeline = None
        self.integrated_system = None
        
    async def initialize_systems(self):
        """Initialize the diagnostic systems"""
        # Initialize enterprise pipeline with custom configuration
        config = {
            "model": {
                "fusion_dim": 512,
                "attention_heads": 8,
                "dropout": 0.1,
                "uncertainty_samples": 50  # Reduced for faster processing in examples
            },
            "clinical": {
                "confidence_threshold": 0.6,
                "dsm5_compliance": True,
                "icd11_compliance": True
            },
            "privacy": {
                "enable_encryption": True,
                "audit_logging": True,
                "data_retention_days": 90
            }
        }
        
        self.pipeline = create_enterprise_pipeline(config)
        self.integrated_system = IntegratedDiagnosticSystem(use_enterprise=True, config=config)
        
        logger.info("Enterprise diagnostic systems initialized")

    async def example_1_basic_text_analysis(self):
        """
        Example 1: Basic text-only diagnostic analysis
        """
        print("\n=== Example 1: Basic Text Analysis ===")
        
        # Sample conversation data
        input_data = {
            "text": {
                "content": "I've been feeling really down lately. I can't sleep well, "
                          "I've lost interest in things I used to enjoy, and I feel "
                          "worthless most of the time. It's been going on for about "
                          "three weeks now. I just don't have energy for anything.",
                "metadata": {
                    "message_count": 1,
                    "conversation_length": 87,
                    "emotional_tone": "negative"
                }
            },
            "contextual": {
                "timestamp": datetime.now().isoformat(),
                "environment": "home",
                "social_context": {"alone": True}
            }
        }
        
        # Process the input
        result = await self.pipeline.process_multimodal_input(
            input_data=input_data,
            user_id="user_001",
            session_id="session_001"
        )
        
        # Display results
        self._display_results("Basic Text Analysis", result)
        
        return result

    async def example_2_multimodal_analysis(self):
        """
        Example 2: Multi-modal analysis with text, voice, and behavioral data
        """
        print("\n=== Example 2: Multi-Modal Analysis ===")
        
        # Comprehensive multi-modal input
        input_data = {
            "text": {
                "content": "I keep having these panic attacks at work. My heart races, "
                          "I start sweating, and I feel like I can't breathe. It's "
                          "happening more frequently now.",
                "history": [
                    {"text": "Work has been really stressful lately", "timestamp": "2024-01-15T10:00:00"},
                    {"text": "I avoid meetings now because of anxiety", "timestamp": "2024-01-15T14:30:00"}
                ]
            },
            "voice": {
                "acoustic_features": {
                    "pitch_mean": 180.5,
                    "pitch_variance": 45.2,
                    "speech_rate": 4.2,  # words per second
                    "pause_frequency": 0.8,
                    "voice_tremor": 0.3
                },
                "emotions": {
                    "anxiety": 0.7,
                    "fear": 0.6,
                    "sadness": 0.3
                }
            },
            "behavioral": {
                "activities": {
                    "social_events_avoided": 5,
                    "work_meetings_skipped": 3,
                    "sleep_hours": 5.2,
                    "exercise_frequency": 0  # times per week
                },
                "physiological": {
                    "heart_rate_avg": 85,
                    "heart_rate_variance": 15
                }
            },
            "temporal": {
                "symptom_onset": "2024-01-01T00:00:00",
                "symptom_progression": "increasing",
                "episode_frequency": "3-4 times per week"
            },
            "contextual": {
                "timestamp": datetime.now().isoformat(),
                "environment": "work",
                "social_context": {"with_colleagues": True},
                "trigger_context": "presentation meeting"
            }
        }
        
        # Process with personalized adaptation enabled
        result = await self.pipeline.process_multimodal_input(
            input_data=input_data,
            user_id="user_002",
            session_id="session_002",
            enable_adaptation=True
        )
        
        self._display_results("Multi-Modal Analysis", result)
        return result

    async def example_3_temporal_pattern_analysis(self):
        """
        Example 3: Temporal pattern analysis with historical data
        """
        print("\n=== Example 3: Temporal Pattern Analysis ===")
        
        # Historical temporal data showing symptom progression
        temporal_data = [
            {
                "timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
                "data": {
                    "symptoms": {
                        "depressed_mood": 0.3,
                        "fatigue": 0.4,
                        "sleep_problems": 0.2
                    },
                    "severity": "mild"
                }
            },
            {
                "timestamp": (datetime.now() - timedelta(days=20)).isoformat(),
                "data": {
                    "symptoms": {
                        "depressed_mood": 0.5,
                        "fatigue": 0.6,
                        "sleep_problems": 0.5,
                        "loss_of_interest": 0.3
                    },
                    "severity": "moderate"
                }
            },
            {
                "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
                "data": {
                    "symptoms": {
                        "depressed_mood": 0.7,
                        "fatigue": 0.8,
                        "sleep_problems": 0.7,
                        "loss_of_interest": 0.6,
                        "worthlessness": 0.4
                    },
                    "severity": "moderate"
                }
            },
            {
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "symptoms": {
                        "depressed_mood": 0.8,
                        "fatigue": 0.9,
                        "sleep_problems": 0.8,
                        "loss_of_interest": 0.7,
                        "worthlessness": 0.6,
                        "concentration_problems": 0.5
                    },
                    "severity": "severe"
                }
            }
        ]
        
        input_data = {
            "text": {
                "content": "My depression seems to be getting worse over the past month. "
                          "I'm sleeping less, feeling more tired, and losing interest "
                          "in everything. I'm worried about how I'm declining."
            },
            "temporal_data": temporal_data,
            "contextual": {
                "timestamp": datetime.now().isoformat(),
                "environment": "home"
            }
        }
        
        result = await self.pipeline.process_multimodal_input(
            input_data=input_data,
            user_id="user_003",
            session_id="session_003"
        )
        
        self._display_results("Temporal Pattern Analysis", result)
        
        # Display temporal patterns if available
        if result.get('temporal_patterns'):
            print("\n--- Temporal Patterns Detected ---")
            for pattern in result['temporal_patterns']:
                print(f"Symptom: {pattern.symptom}")
                print(f"Pattern Type: {pattern.pattern_type}")
                print(f"Trend Score: {pattern.trend_score:.3f}")
                print(f"Time Window: {pattern.time_window} days")
                print(f"Significance: {pattern.significance:.3f}")
                print()
        
        return result

    async def example_4_integration_with_existing_components(self):
        """
        Example 4: Integration with existing Solace-AI components
        """
        print("\n=== Example 4: Integration with Existing Components ===")
        
        # Simulate data from existing Solace-AI components
        conversation_text = "I feel overwhelmed by work and personal life. " \
                          "Nothing seems to go right lately."
        
        # Use integrated system for backward compatibility
        legacy_input = {
            "conversation_data": {
                "text": conversation_text,
                "extracted_symptoms": ["overwhelmed", "negative outlook"],
                "sentiment": -0.6
            },
            "voice_emotion_data": {
                "emotions": {"stress": 0.7, "sadness": 0.5},
                "characteristics": {"monotone": 0.6, "low_energy": 0.7}
            },
            "personality_data": {
                "big_five": {
                    "neuroticism": 0.75,
                    "extraversion": 0.3,
                    "openness": 0.6,
                    "agreeableness": 0.8,
                    "conscientiousness": 0.4
                }
            },
            "user_id": "user_004",
            "session_id": "session_004"
        }
        
        # Process using integrated system (automatically uses enterprise with legacy fallback)
        result = await self.integrated_system.generate_diagnosis(**legacy_input)
        
        self._display_results("Integration with Existing Components", result)
        return result

    async def example_5_batch_processing(self):
        """
        Example 5: Batch processing for multiple users
        """
        print("\n=== Example 5: Batch Processing ===")
        
        # Sample batch of user inputs
        batch_inputs = [
            {
                "user_id": "batch_user_001",
                "session_id": "batch_session_001",
                "data": {
                    "text": {"content": "I feel anxious about my upcoming presentation"},
                    "contextual": {"environment": "work", "timestamp": datetime.now().isoformat()}
                }
            },
            {
                "user_id": "batch_user_002", 
                "session_id": "batch_session_002",
                "data": {
                    "text": {"content": "I can't stop thinking about past mistakes"},
                    "contextual": {"environment": "home", "timestamp": datetime.now().isoformat()}
                }
            },
            {
                "user_id": "batch_user_003",
                "session_id": "batch_session_003", 
                "data": {
                    "text": {"content": "I'm having trouble sleeping and feel tired all day"},
                    "contextual": {"environment": "home", "timestamp": datetime.now().isoformat()}
                }
            }
        ]
        
        # Process batch concurrently
        batch_results = await asyncio.gather(*[
            self.pipeline.process_multimodal_input(
                input_data=item["data"],
                user_id=item["user_id"],
                session_id=item["session_id"]
            )
            for item in batch_inputs
        ])
        
        # Display batch results summary
        print("Batch Processing Results:")
        for i, result in enumerate(batch_results):
            user_id = batch_inputs[i]["user_id"]
            success = result.get("success", False)
            primary_condition = None
            if result.get("diagnostic_results", {}).get("conditions"):
                primary_condition = result["diagnostic_results"]["conditions"][0]["name"]
            
            print(f"User {user_id}: Success={success}, "
                  f"Primary Condition={primary_condition}, "
                  f"Processing Time={result.get('processing_time', 0):.3f}s")
        
        return batch_results

    async def example_6_monitoring_and_ab_testing(self):
        """
        Example 6: Monitoring and A/B testing capabilities
        """
        print("\n=== Example 6: Monitoring and A/B Testing ===")
        
        # Setup A/B test
        test_variants = {
            "variant_a": {"confidence_threshold": 0.6, "fusion_dim": 512},
            "variant_b": {"confidence_threshold": 0.7, "fusion_dim": 768}
        }
        
        self.pipeline.setup_ab_test("confidence_threshold_test", test_variants)
        
        # Simulate processing with different variants
        test_input = {
            "text": {"content": "I feel worried about my health lately"},
            "contextual": {"timestamp": datetime.now().isoformat()}
        }
        
        # Process with variant A
        result_a = await self.pipeline.process_multimodal_input(
            input_data=test_input,
            user_id="ab_test_user_a",
            session_id="ab_test_session_a"
        )
        
        # Record A/B test result
        self.pipeline.record_ab_test_result(
            "confidence_threshold_test", 
            "variant_a", 
            {
                "success": result_a.get("success", False),
                "processing_time": result_a.get("processing_time", 0),
                "confidence_level": result_a.get("confidence_level", "unknown")
            }
        )
        
        # Get performance summary
        performance_summary = self.pipeline.get_performance_summary()
        print("Performance Summary:")
        print(json.dumps(performance_summary, indent=2))
        
        # Analyze A/B test results
        ab_analysis = self.pipeline.analyze_ab_test_results("confidence_threshold_test")
        print("\nA/B Test Analysis:")
        print(json.dumps(ab_analysis, indent=2))
        
        return {"performance": performance_summary, "ab_test": ab_analysis}

    async def example_7_clinical_workflow_integration(self):
        """
        Example 7: Clinical workflow integration with compliance reporting
        """
        print("\n=== Example 7: Clinical Workflow Integration ===")
        
        # Clinical assessment scenario
        clinical_input = {
            "text": {
                "content": "Patient reports persistent sadness, loss of appetite, "
                          "difficulty sleeping, and thoughts of self-harm for the past 6 weeks. "
                          "Significant functional impairment noted.",
                "clinical_context": {
                    "setting": "clinical_interview",
                    "clinician_id": "clinician_001",
                    "assessment_type": "initial_evaluation"
                }
            },
            "behavioral": {
                "functional_assessment": {
                    "work_impairment": 0.8,
                    "social_impairment": 0.7,
                    "self_care_impairment": 0.6
                },
                "risk_factors": {
                    "suicidal_ideation": True,
                    "substance_use": False,
                    "family_history": True
                }
            },
            "contextual": {
                "timestamp": datetime.now().isoformat(),
                "environment": "healthcare",
                "urgency_level": "high"
            }
        }
        
        result = await self.pipeline.process_multimodal_input(
            input_data=clinical_input,
            user_id="clinical_patient_001",
            session_id="clinical_session_001"
        )
        
        # Display clinical results with emphasis on safety
        self._display_clinical_results(result)
        
        # Generate compliance report
        compliance_report = self.pipeline.get_compliance_report()
        print("\nCompliance Report:")
        print(json.dumps(compliance_report, indent=2))
        
        return result

    def _display_results(self, example_title: str, result: Dict[str, Any]):
        """Display diagnostic results in a formatted way"""
        print(f"\n--- {example_title} Results ---")
        
        if not result.get("success"):
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return
        
        print(f"✅ Processing successful")
        print(f"Processing time: {result.get('processing_time', 0):.3f} seconds")
        print(f"Modalities processed: {', '.join(result.get('modalities_processed', []))}")
        print(f"Overall confidence: {result.get('confidence_level', 'unknown')}")
        
        # Display diagnostic results
        diagnostic_results = result.get("diagnostic_results", {})
        if diagnostic_results.get("conditions"):
            print("\n🔍 Diagnostic Results:")
            for i, condition in enumerate(diagnostic_results["conditions"][:3]):  # Top 3
                print(f"  {i+1}. {condition['name']}: {condition['probability']:.3f} "
                      f"(±{condition.get('uncertainty', 0):.3f})")
        
        # Display severity assessment
        severity = diagnostic_results.get("severity", {})
        if severity:
            print(f"\n📊 Severity: {severity.get('predicted', 'unknown')} "
                  f"(confidence: {severity.get('probability', 0):.3f})")
        
        # Display key recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3]):  # Top 3
                print(f"  {i+1}. {rec}")
        
        # Display uncertainty analysis
        uncertainty = result.get("uncertainty_analysis", {})
        if uncertainty.get("reliability_score"):
            print(f"\n🎯 Reliability Score: {uncertainty['reliability_score']:.3f}")
        
        print("-" * 50)

    def _display_clinical_results(self, result: Dict[str, Any]):
        """Display clinical results with emphasis on safety and compliance"""
        print("\n--- Clinical Assessment Results ---")
        
        if not result.get("success"):
            print(f"❌ Assessment failed: {result.get('error', 'Unknown error')}")
            return
        
        # Check for high-priority alerts
        clinical_assessment = result.get("clinical_assessment", {})
        suicide_risk = clinical_assessment.get("suicide_risk", {})
        
        if suicide_risk.get("requires_immediate_attention"):
            print("🚨 URGENT ALERT: Immediate mental health evaluation required")
            print(f"   Risk Level: {suicide_risk.get('risk_level', 'unknown')}")
            print(f"   Risk Indicators: {', '.join(suicide_risk.get('indicators', []))}")
        
        # Display DSM-5/ICD-11 compliance
        dsm5_compliance = clinical_assessment.get("dsm5_compliance", {})
        icd11_compliance = clinical_assessment.get("icd11_compliance", {})
        
        print(f"\n📋 Clinical Standards Compliance:")
        print(f"   DSM-5 Compliant: {'✅' if dsm5_compliance.get('compliant') else '❌'}")
        print(f"   ICD-11 Codes: {', '.join([c['code'] for c in icd11_compliance.get('diagnostic_codes', [])])}")
        
        # Display diagnostic results
        self._display_results("Clinical Assessment", result)

    async def run_all_examples(self):
        """Run all usage examples"""
        print("🚀 Starting Enterprise Multi-Modal Diagnostic Pipeline Examples")
        print("=" * 70)
        
        await self.initialize_systems()
        
        examples = [
            self.example_1_basic_text_analysis,
            self.example_2_multimodal_analysis,
            self.example_3_temporal_pattern_analysis,
            self.example_4_integration_with_existing_components,
            self.example_5_batch_processing,
            self.example_6_monitoring_and_ab_testing,
            self.example_7_clinical_workflow_integration
        ]
        
        results = {}
        for i, example in enumerate(examples, 1):
            try:
                print(f"\n{'='*20} Running Example {i} {'='*20}")
                result = await example()
                results[f"example_{i}"] = result
            except Exception as e:
                print(f"❌ Example {i} failed: {str(e)}")
                results[f"example_{i}"] = {"error": str(e)}
        
        print("\n🎉 All examples completed!")
        return results

# Integration helper functions
class SolaceAIIntegration:
    """
    Helper class for integrating enterprise pipeline with existing Solace-AI components
    """
    
    @staticmethod
    async def integrate_with_voice_component(voice_component: VoiceComponent, text_input: str):
        """
        Example integration with existing voice component
        """
        # This would be called when voice analysis is available
        try:
            # Get voice analysis (mock implementation)
            voice_analysis = {
                "emotions": {"anxiety": 0.6, "sadness": 0.4},
                "acoustic_features": {
                    "pitch_mean": 165.0,
                    "speech_rate": 3.5,
                    "pause_frequency": 0.6
                }
            }
            
            # Prepare enterprise input format
            enterprise_input = {
                "text": {"content": text_input},
                "voice": voice_analysis,
                "contextual": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "voice_component_integration"
                }
            }
            
            # Create and use enterprise pipeline
            pipeline = create_enterprise_pipeline()
            result = await pipeline.process_multimodal_input(
                enterprise_input, "integrated_user", "integrated_session"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Voice component integration error: {str(e)}")
            return None

    @staticmethod
    async def integrate_with_conversation_analyzer(analyzer: ConversationAnalyzer, conversation_history: List[str]):
        """
        Example integration with conversation analyzer
        """
        try:
            # Analyze conversation (mock implementation)
            conversation_analysis = {
                "sentiment_trend": -0.3,
                "topic_shifts": 2,
                "emotional_progression": ["neutral", "concerned", "distressed"]
            }
            
            # Combine conversation history
            combined_text = " ".join(conversation_history)
            
            enterprise_input = {
                "text": {
                    "content": combined_text,
                    "analysis": conversation_analysis,
                    "history": [{"text": msg, "timestamp": datetime.now().isoformat()} 
                              for msg in conversation_history]
                },
                "behavioral": {
                    "conversation_patterns": {
                        "message_frequency": len(conversation_history),
                        "avg_message_length": np.mean([len(msg) for msg in conversation_history]),
                        "sentiment_variance": 0.4
                    }
                },
                "contextual": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "conversation_analyzer_integration"
                }
            }
            
            pipeline = create_enterprise_pipeline()
            result = await pipeline.process_multimodal_input(
                enterprise_input, "conversation_user", "conversation_session"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Conversation analyzer integration error: {str(e)}")
            return None

    @staticmethod
    async def integrate_with_personality_assessment(big_five_scores: Dict[str, float], user_input: str):
        """
        Example integration with personality assessment
        """
        try:
            enterprise_input = {
                "text": {"content": user_input},
                "behavioral": {
                    "personality_profile": {
                        "big_five": big_five_scores,
                        "assessment_date": datetime.now().isoformat(),
                        "assessment_confidence": 0.85
                    }
                },
                "contextual": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "personality_assessment_integration"
                }
            }
            
            pipeline = create_enterprise_pipeline()
            result = await pipeline.process_multimodal_input(
                enterprise_input, "personality_user", "personality_session"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Personality assessment integration error: {str(e)}")
            return None

# Example usage script
async def main():
    """
    Main function to run all examples
    """
    examples = EnterpriseUsageExamples()
    results = await examples.run_all_examples()
    
    print("\n📈 Summary of Results:")
    successful_examples = sum(1 for result in results.values() 
                            if isinstance(result, dict) and result.get("success", False))
    print(f"Successful examples: {successful_examples}/{len(results)}")
    
    return results

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())