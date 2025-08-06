"""
Example usage of TherapeuticFrictionAgent demonstrating growth-oriented therapeutic responses.

This script shows how the agent assesses user readiness, applies appropriate challenges,
and adapts its approach based on therapeutic relationship dynamics.
"""

import asyncio
from unittest.mock import Mock
from datetime import datetime

# Import the agent (in real usage, these would be proper imports)
try:
    from src.agents.therapeutic_friction_agent import TherapeuticFrictionAgent, UserReadinessIndicator, ChallengeLevel
except ImportError:
    print("Note: This is a demonstration example. In actual usage, ensure proper imports are available.")
    exit()


class MockModelProvider:
    """Mock model provider for demonstration purposes."""
    
    def generate(self, prompt):
        return "Generated therapeutic response"
    
    async def agenerate(self, prompt):
        return "Generated therapeutic response"


def demonstrate_readiness_assessment():
    """Demonstrate how the agent assesses user readiness for challenges."""
    print("=== User Readiness Assessment Demo ===\n")
    
    # Create agent instance
    agent = TherapeuticFrictionAgent(MockModelProvider())
    
    # Test scenarios with different user inputs
    scenarios = [
        {
            "input": "This won't work. I've tried everything and nothing helps.",
            "context": {"emotion_analysis": {"primary_emotion": "frustration"}},
            "expected": "Resistant - needs validation and trust building"
        },
        {
            "input": "I'm not sure if this will help, but maybe I could try something different.",
            "context": {"emotion_analysis": {"primary_emotion": "uncertainty"}},
            "expected": "Ambivalent - gentle exploration appropriate"
        },
        {
            "input": "I'm ready to do whatever it takes to change. What should I do?",
            "context": {"emotion_analysis": {"primary_emotion": "determination"}},
            "expected": "Motivated - strong challenges appropriate"
        },
        {
            "input": "Oh wow, I never realized this pattern before! This makes so much sense now!",
            "context": {"emotion_analysis": {"primary_emotion": "clarity"}},
            "expected": "Breakthrough ready - maximum therapeutic leverage"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}:")
        print(f"User: \"{scenario['input']}\"")
        
        result = agent.process(scenario['input'], scenario['context'])
        
        print(f"Assessed Readiness: {result['user_readiness']}")
        print(f"Challenge Level: {result['challenge_level']}")
        print(f"Intervention Type: {result['intervention_type']}")
        print(f"Expected: {scenario['expected']}")
        print(f"Breakthrough Detected: {result['breakthrough_detected']}")
        print("-" * 50)


def demonstrate_progressive_sessions():
    """Demonstrate how the agent adapts over multiple sessions."""
    print("\n=== Progressive Session Demo ===\n")
    
    agent = TherapeuticFrictionAgent(MockModelProvider())
    
    # Simulate a therapeutic journey
    sessions = [
        {
            "session": 1,
            "input": "I don't think therapy can help me. I'm too broken.",
            "context": {"emotion_analysis": {"primary_emotion": "hopelessness"}},
            "description": "Initial resistance"
        },
        {
            "session": 3,
            "input": "Well, maybe there are some things I haven't looked at closely.",
            "context": {"emotion_analysis": {"primary_emotion": "curiosity"}},
            "description": "Growing openness"
        },
        {
            "session": 6,
            "input": "I'm starting to see some patterns in my relationships. What questions should I be asking myself?",
            "context": {"emotion_analysis": {"primary_emotion": "insight"}},
            "description": "Active engagement"
        },
        {
            "session": 10,
            "input": "I get it now! I've been recreating my childhood dynamics. I can choose differently!",
            "context": {"emotion_analysis": {"primary_emotion": "empowerment"}},
            "description": "Breakthrough moment"
        }
    ]
    
    for session_data in sessions:
        print(f"Session {session_data['session']}: {session_data['description']}")
        print(f"User: \"{session_data['input']}\"")
        
        result = agent.process(session_data['input'], session_data['context'])
        
        print(f"Readiness: {result['user_readiness']}")
        print(f"Challenge Level: {result['challenge_level']}")
        print(f"Therapeutic Bond: {result['therapeutic_relationship']['therapeutic_bond_strength']:.2f}")
        print(f"Growth Trajectory: {result['progress_metrics']['growth_trajectory']}")
        print(f"Session Count: {result['progress_metrics']['session_count']}")
        
        if result['breakthrough_detected']:
            print("🎉 BREAKTHROUGH DETECTED!")
        
        print("-" * 50)


def demonstrate_intervention_types():
    """Demonstrate different intervention types based on user content."""
    print("\n=== Intervention Types Demo ===\n")
    
    agent = TherapeuticFrictionAgent(MockModelProvider())
    
    interventions = [
        {
            "input": "I'm scared to apply for new jobs because I might get rejected.",
            "context": {"emotion_analysis": {"primary_emotion": "anxiety"}},
            "expected_intervention": "Exposure Challenge"
        },
        {
            "input": "I always mess things up. I should be perfect at everything I do.",
            "context": {"emotion_analysis": {"primary_emotion": "self-criticism"}},
            "expected_intervention": "Cognitive Reframing"
        },
        {
            "input": "I don't know what really matters to me anymore. Everything feels meaningless.",
            "context": {"emotion_analysis": {"primary_emotion": "emptiness"}},
            "expected_intervention": "Values Clarification"
        },
        {
            "input": "What's the point? Nothing I do makes a difference anyway.",
            "context": {"emotion_analysis": {"primary_emotion": "frustration"}},
            "expected_intervention": "Strategic Resistance (Paradoxical)"
        }
    ]
    
    for intervention in interventions:
        print(f"User: \"{intervention['input']}\"")
        
        result = agent.process(intervention['input'], intervention['context'])
        
        print(f"Intervention Type: {result['intervention_type']}")
        print(f"Expected: {intervention['expected_intervention']}")
        
        # Show strategy components
        strategy = result['response_strategy']
        if strategy.get('growth_questions'):
            print(f"Growth Questions: {len(strategy['growth_questions'])} generated")
        if strategy.get('behavioral_experiments'):
            print(f"Behavioral Experiments: {len(strategy['behavioral_experiments'])} suggested")
        if strategy.get('strategic_challenges'):
            print(f"Strategic Challenges: {len(strategy['strategic_challenges'])} crafted")
        
        print("-" * 50)


def demonstrate_response_enhancement():
    """Demonstrate how the agent enhances responses with therapeutic friction."""
    print("\n=== Response Enhancement Demo ===\n")
    
    agent = TherapeuticFrictionAgent(MockModelProvider())
    
    # Simulate a scenario
    user_input = "I keep making the same mistakes in relationships. I guess I'm just not good at this."
    context = {"emotion_analysis": {"primary_emotion": "self-doubt"}}
    
    # Process with friction agent
    friction_result = agent.process(user_input, context)
    
    # Show original vs enhanced response
    original_response = "I understand that relationships can be challenging and it's frustrating when patterns repeat."
    
    enhanced_response = agent.enhance_response(original_response, friction_result)
    
    print("Original Response:")
    print(f'"{original_response}"')
    print("\nEnhanced Response with Therapeutic Friction:")
    print(f'"{enhanced_response}"')
    
    print(f"\nFriction Level: {friction_result['challenge_level']}")
    print(f"Intervention Focus: {friction_result['intervention_type']}")
    print(f"Friction Recommendation: {friction_result['friction_recommendation']}")


def demonstrate_therapeutic_relationship_tracking():
    """Demonstrate therapeutic relationship monitoring."""
    print("\n=== Therapeutic Relationship Tracking Demo ===\n")
    
    agent = TherapeuticFrictionAgent(MockModelProvider())
    
    # Simulate interactions that affect relationship
    interactions = [
        {
            "input": "I don't want to talk about this anymore.",
            "expected_effect": "Trust decrease, engagement decrease"
        },
        {
            "input": "Thank you for helping me understand this better. I appreciate your patience.",
            "expected_effect": "Trust increase, bond strengthening"
        },
        {
            "input": "I've been thinking about what we discussed, and I tried that technique you suggested.",
            "expected_effect": "Engagement increase, receptivity to challenge increase"
        }
    ]
    
    print("Initial Relationship Metrics:")
    rel = agent.therapeutic_relationship
    print(f"Trust Level: {rel.trust_level:.2f}")
    print(f"Engagement: {rel.engagement_score:.2f}")
    print(f"Challenge Receptivity: {rel.receptivity_to_challenge:.2f}")
    print(f"Bond Strength: {rel.therapeutic_bond_strength:.2f}")
    print()
    
    for i, interaction in enumerate(interactions, 1):
        print(f"Interaction {i}: \"{interaction['input']}\"")
        
        result = agent.process(interaction['input'], {"emotion_analysis": {"primary_emotion": "neutral"}})
        
        rel = agent.therapeutic_relationship
        print(f"Trust Level: {rel.trust_level:.2f}")
        print(f"Engagement: {rel.engagement_score:.2f}")
        print(f"Challenge Receptivity: {rel.receptivity_to_challenge:.2f}")
        print(f"Bond Strength: {rel.therapeutic_bond_strength:.2f}")
        print(f"Expected Effect: {interaction['expected_effect']}")
        print("-" * 40)


def demonstrate_comprehensive_assessment():
    """Demonstrate comprehensive assessment after multiple sessions."""
    print("\n=== Comprehensive Assessment Demo ===\n")
    
    agent = TherapeuticFrictionAgent(MockModelProvider())
    
    # Simulate several sessions
    for i in range(8):
        inputs = [
            "I don't think this will work",
            "Maybe I should try to understand this better",
            "I'm starting to see some patterns",
            "What questions should I be asking myself?",
            "I tried that exercise and it was helpful",
            "I'm noticing when I do this behavior now",
            "I understand why I react this way",
            "I feel like I can handle challenges better now"
        ]
        
        agent.process(inputs[i], {"emotion_analysis": {"primary_emotion": "neutral"}})
    
    # Get comprehensive assessment
    assessment = agent.get_comprehensive_assessment()
    
    print("COMPREHENSIVE THERAPEUTIC ASSESSMENT")
    print("=" * 50)
    
    print(f"Total Sessions: {assessment['session_summary']['total_sessions']}")
    print(f"Current Therapy Phase: {assessment['session_summary']['current_phase']}")
    print(f"Growth Trajectory: {assessment['user_progress']['growth_trajectory']}")
    print(f"Breakthrough Moments: {len(assessment['user_progress']['breakthrough_moments'])}")
    print(f"Challenge Acceptance Rate: {assessment['user_progress']['challenge_acceptance_rate']:.2f}")
    print(f"Therapeutic Bond Strength: {assessment['therapeutic_relationship']['therapeutic_bond_strength']:.2f}")
    print(f"Breakthrough Potential: {assessment['breakthrough_potential']:.2f}")
    
    print("\nRECOMMENDATIONS:")
    for rec in assessment['recommendations']:
        print(f"• {rec}")
    
    print("\nNEXT STEPS:")
    for step in assessment['session_summary']['next_steps']:
        print(f"• {step}")


def main():
    """Run all demonstrations."""
    print("THERAPEUTIC FRICTION AGENT DEMONSTRATION")
    print("=" * 60)
    
    try:
        demonstrate_readiness_assessment()
        demonstrate_progressive_sessions()
        demonstrate_intervention_types()
        demonstrate_response_enhancement()
        demonstrate_therapeutic_relationship_tracking()
        demonstrate_comprehensive_assessment()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("\nKey Features Demonstrated:")
        print("✓ User readiness assessment with NLP analysis")
        print("✓ Adaptive challenge level determination")
        print("✓ Multiple intervention types (Socratic, Behavioral, Cognitive)")
        print("✓ Progressive session tracking and relationship monitoring")
        print("✓ Breakthrough detection and growth trajectory analysis")
        print("✓ Comprehensive therapeutic assessment and recommendations")
        print("✓ Strategic friction application with therapeutic boundaries")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Note: This demonstration requires the TherapeuticFrictionAgent to be properly installed.")


if __name__ == "__main__":
    main()