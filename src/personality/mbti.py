from typing import Dict, Any, List, Optional
import logging
import json
import os
import random

logger = logging.getLogger(__name__)

class MBTIAssessment:
    """
    Implementation of the Myers-Briggs Type Indicator (MBTI) personality assessment.
    """
    
    def __init__(self):
        """Initialize the MBTI assessment"""
        self.questions = self._load_questions()
        self.type_descriptions = self._load_type_descriptions()
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load the MBTI questions from the data file"""
        try:
            # Try to load questions from the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'personality')
            os.makedirs(data_dir, exist_ok=True)
            
            questions_path = os.path.join(data_dir, 'mbti_questions.json')
            
            # If the file doesn't exist, create it with default questions
            if not os.path.exists(questions_path):
                self._create_default_questions(questions_path)
            
            with open(questions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading MBTI questions: {str(e)}")
            # Return a minimal set of questions as fallback
            return self._get_fallback_questions()
    
    def _create_default_questions(self, file_path: str) -> None:
        """Create a default questions file"""
        default_questions = [
            {
                "id": 1,
                "text": "At a party, you:",
                "options": [
                    {"key": "A", "text": "Interact with many, including strangers", "dimension": "E"},
                    {"key": "B", "text": "Interact with a few, known to you", "dimension": "I"}
                ]
            },
            {
                "id": 2,
                "text": "You are more:",
                "options": [
                    {"key": "A", "text": "Realistic than speculative", "dimension": "S"},
                    {"key": "B", "text": "Speculative than realistic", "dimension": "N"}
                ]
            },
            {
                "id": 3,
                "text": "Is it worse to:",
                "options": [
                    {"key": "A", "text": "Have your head in the clouds", "dimension": "S"},
                    {"key": "B", "text": "Be in a rut", "dimension": "N"}
                ]
            },
            {
                "id": 4,
                "text": "You are more impressed by:",
                "options": [
                    {"key": "A", "text": "Principles", "dimension": "T"},
                    {"key": "B", "text": "Emotions", "dimension": "F"}
                ]
            },
            {
                "id": 5,
                "text": "You are drawn more to:",
                "options": [
                    {"key": "A", "text": "Convincing reason", "dimension": "T"},
                    {"key": "B", "text": "Touching hearts", "dimension": "F"}
                ]
            },
            {
                "id": 6,
                "text": "You prefer to work:",
                "options": [
                    {"key": "A", "text": "To deadlines", "dimension": "J"},
                    {"key": "B", "text": "Just whenever", "dimension": "P"}
                ]
            },
            {
                "id": 7,
                "text": "You tend to choose:",
                "options": [
                    {"key": "A", "text": "Rather carefully", "dimension": "J"},
                    {"key": "B", "text": "Somewhat impulsively", "dimension": "P"}
                ]
            },
            {
                "id": 8,
                "text": "At parties, you:",
                "options": [
                    {"key": "A", "text": "Stay late, with increasing energy", "dimension": "E"},
                    {"key": "B", "text": "Leave early, with decreased energy", "dimension": "I"}
                ]
            },
            {
                "id": 9,
                "text": "You are more attracted to:",
                "options": [
                    {"key": "A", "text": "Sensible people", "dimension": "S"},
                    {"key": "B", "text": "Imaginative people", "dimension": "N"}
                ]
            },
            {
                "id": 10,
                "text": "You are more interested in:",
                "options": [
                    {"key": "A", "text": "What is actual", "dimension": "S"},
                    {"key": "B", "text": "What is possible", "dimension": "N"}
                ]
            },
            {
                "id": 11,
                "text": "In judging others, you are more swayed by:",
                "options": [
                    {"key": "A", "text": "Laws than circumstances", "dimension": "T"},
                    {"key": "B", "text": "Circumstances than laws", "dimension": "F"}
                ]
            },
            {
                "id": 12,
                "text": "In approaching others, you are usually more:",
                "options": [
                    {"key": "A", "text": "Objective", "dimension": "T"},
                    {"key": "B", "text": "Personal", "dimension": "F"}
                ]
            },
            {
                "id": 13,
                "text": "You are more:",
                "options": [
                    {"key": "A", "text": "Punctual", "dimension": "J"},
                    {"key": "B", "text": "Leisurely", "dimension": "P"}
                ]
            },
            {
                "id": 14,
                "text": "It bothers you more having things:",
                "options": [
                    {"key": "A", "text": "Incomplete", "dimension": "J"},
                    {"key": "B", "text": "Completed", "dimension": "P"}
                ]
            },
            {
                "id": 15,
                "text": "In your social groups, you:",
                "options": [
                    {"key": "A", "text": "Keep abreast of others' happenings", "dimension": "E"},
                    {"key": "B", "text": "Get behind on the news", "dimension": "I"}
                ]
            },
            {
                "id": 16,
                "text": "In doing ordinary things, you are more likely to:",
                "options": [
                    {"key": "A", "text": "Do it the usual way", "dimension": "S"},
                    {"key": "B", "text": "Do it your own way", "dimension": "N"}
                ]
            },
            {
                "id": 17,
                "text": "Writers should:",
                "options": [
                    {"key": "A", "text": "Say what they mean and mean what they say", "dimension": "S"},
                    {"key": "B", "text": "Express things more by use of analogy", "dimension": "N"}
                ]
            },
            {
                "id": 18,
                "text": "You are more drawn to:",
                "options": [
                    {"key": "A", "text": "Consistency of thought", "dimension": "T"},
                    {"key": "B", "text": "Harmonious human relationships", "dimension": "F"}
                ]
            },
            {
                "id": 19,
                "text": "You are more comfortable making:",
                "options": [
                    {"key": "A", "text": "Logical judgments", "dimension": "T"},
                    {"key": "B", "text": "Value judgments", "dimension": "F"}
                ]
            },
            {
                "id": 20,
                "text": "You want things:",
                "options": [
                    {"key": "A", "text": "Settled and decided", "dimension": "J"},
                    {"key": "B", "text": "Unsettled and undecided", "dimension": "P"}
                ]
            }
        ]
        
        try:
            with open(file_path, 'w') as f:
                json.dump(default_questions, f, indent=2)
        except Exception as e:
            logger.error(f"Error creating default questions file: {str(e)}")
    
    def _load_type_descriptions(self) -> Dict[str, Any]:
        """Load the MBTI type descriptions"""
        try:
            # Try to load descriptions from the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'personality')
            os.makedirs(data_dir, exist_ok=True)
            
            descriptions_path = os.path.join(data_dir, 'mbti_descriptions.json')
            
            # If the file doesn't exist, create it with default descriptions
            if not os.path.exists(descriptions_path):
                self._create_default_descriptions(descriptions_path)
            
            with open(descriptions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading MBTI descriptions: {str(e)}")
            # Return a minimal set of descriptions as fallback
            return self._get_fallback_descriptions()
    
    def _create_default_descriptions(self, file_path: str) -> None:
        """Create a default descriptions file"""
        default_descriptions = {
            "ISTJ": {
                "name": "The Inspector",
                "description": "Practical, fact-minded, reliable, and responsible. Decide logically what should be done and work toward it steadily, regardless of distractions.",
                "strengths": ["Organized", "Honest and direct", "Dedicated", "Strong-willed", "Responsible"],
                "weaknesses": ["Stubborn", "Insensitive", "Always by the book", "Judgmental", "Often unreasonably blame themselves"]
            },
            "ISFJ": {
                "name": "The Protector",
                "description": "Quiet, friendly, responsible, and conscientious. Committed and steady in meeting their obligations. Thorough, painstaking, and accurate.",
                "strengths": ["Supportive", "Reliable", "Patient", "Observant", "Enthusiastic"],
                "weaknesses": ["Overload themselves", "Shy and reserved", "Reluctant to change", "Too selfless", "Take things too personally"]
            },
            "INFJ": {
                "name": "The Counselor",
                "description": "Seek meaning and connection in ideas, relationships, and material possessions. Want to understand what motivates people and are insightful about others.",
                "strengths": ["Creative", "Insightful", "Principled", "Passionate", "Altruistic"],
                "weaknesses": ["Sensitive to criticism", "Reluctant to open up", "Perfectionist", "Avoiding confrontation", "Burnout-prone"]
            },
            "INTJ": {
                "name": "The Mastermind",
                "description": "Have original minds and great drive for implementing their ideas and achieving their goals. Quickly see patterns in external events and develop long-range explanatory perspectives.",
                "strengths": ["Strategic thinkers", "Independent", "Hard-working", "Open-minded", "Curious"],
                "weaknesses": ["Arrogant", "Dismissive of emotions", "Overly critical", "Combative", "Romantically clueless"]
            },
            "ISTP": {
                "name": "The Craftsman",
                "description": "Tolerant and flexible, quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily get through large amounts of data to isolate the core of practical problems.",
                "strengths": ["Optimistic and energetic", "Creative and practical", "Spontaneous", "Independent", "Rational and logical"],
                "weaknesses": ["Stubborn", "Insensitive", "Private and reserved", "Easily bored", "Dislike commitment"]
            },
            "ISFP": {
                "name": "The Composer",
                "description": "Quiet, friendly, sensitive, and kind. Enjoy the present moment, what's going on around them. Like to have their own space and to work within their own time frame. Loyal and committed to their values and to people who are important to them.",
                "strengths": ["Charming", "Sensitive to others", "Imaginative", "Passionate", "Curious"],
                "weaknesses": ["Fiercely independent", "Unpredictable", "Easily stressed", "Overly competitive", "Fluctuating self-esteem"]
            },
            "INFP": {
                "name": "The Healer",
                "description": "Idealistic, loyal to their values and to people who are important to them. Want an external life that is congruent with their values. Curious, quick to see possibilities, can be catalysts for implementing ideas.",
                "strengths": ["Empathetic", "Generous", "Open-minded", "Creative", "Passionate"],
                "weaknesses": ["Unrealistic", "Self-isolating", "Unfocused", "Emotionally vulnerable", "Self-critical"]
            },
            "INTP": {
                "name": "The Architect",
                "description": "Seek to develop logical explanations for everything that interests them. Theoretical and abstract, interested more in ideas than in social interaction. Quiet, contained, flexible, and adaptable.",
                "strengths": ["Analytical", "Original", "Open-minded", "Curious", "Objective"],
                "weaknesses": ["Very private", "Insensitive", "Absent-minded", "Condescending", "Loathe rules and guidelines"]
            },
            "ESTP": {
                "name": "The Dynamo",
                "description": "Flexible and tolerant, they take a pragmatic approach focused on immediate results. Theories and conceptual explanations bore them - they want to act energetically to solve the problem.",
                "strengths": ["Bold", "Rational and practical", "Original", "Perceptive", "Direct"],
                "weaknesses": ["Insensitive", "Impatient", "Risk-prone", "Unstructured", "May miss the bigger picture"]
            },
            "ESFP": {
                "name": "The Performer",
                "description": "Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Bring common sense and a realistic approach to their work, and make work fun.",
                "strengths": ["Bold", "Original", "Practical", "Observant", "Excellent people skills"],
                "weaknesses": ["Sensitive", "Conflict-averse", "Easily bored", "Poor long-term planners", "Unfocused"]
            },
            "ENFP": {
                "name": "The Champion",
                "description": "Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information very quickly, and confidently proceed based on the patterns they see.",
                "strengths": ["Curious", "Observant", "Energetic", "Excellent communicators", "Know how to relax"],
                "weaknesses": ["Poor practical skills", "Find it difficult to focus", "Overthink things", "Get stressed easily", "Highly emotional"]
            },
            "ENTP": {
                "name": "The Visionary",
                "description": "Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Adept at generating conceptual possibilities and then analyzing them strategically.",
                "strengths": ["Knowledgeable", "Quick thinkers", "Original", "Charismatic", "Energetic"],
                "weaknesses": ["Very argumentative", "Insensitive", "Intolerant", "Poor practical skills", "Difficulty focusing"]
            },
            "ESTJ": {
                "name": "The Supervisor",
                "description": "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible.",
                "strengths": ["Dedicated", "Strong-willed", "Direct and honest", "Loyal, patient and reliable", "Enjoy creating order"],
                "weaknesses": ["Inflexible", "Stubborn", "Judgmental", "Difficult relaxing", "Difficulty expressing emotion"]
            },
            "ESFJ": {
                "name": "The Provider",
                "description": "Warmhearted, conscientious, and cooperative. Want harmony in their environment, work with determination to establish it. Like to work with others to complete tasks accurately and on time.",
                "strengths": ["Strong practical skills", "Strong sense of duty", "Very loyal", "Sensitive and warm", "Good at connecting with others"],
                "weaknesses": ["Worried about their social status", "Inflexible", "Reluctant to innovate", "Vulnerable to criticism", "Often too selfless"]
            },
            "ENFJ": {
                "name": "The Teacher",
                "description": "Warm, empathetic, responsive, and responsible. Highly attuned to the emotions, needs, and motivations of others. Find potential in everyone, want to help others fulfill their potential.",
                "strengths": ["Tolerant", "Reliable", "Charismatic", "Altruistic", "Natural leaders"],
                "weaknesses": ["Overly idealistic", "Too selfless", "Overly sensitive", "Fluctuating self-esteem", "Struggle with making tough decisions"]
            },
            "ENTJ": {
                "name": "The Commander",
                "description": "Frank, decisive, assume leadership readily. Quickly see illogical and inefficient procedures and policies, develop and implement comprehensive systems to solve organizational problems.",
                "strengths": ["Efficient", "Energetic", "Self-confident", "Strong-willed", "Strategic thinkers"],
                "weaknesses": ["Stubborn", "Intolerant", "Impatient", "Arrogant", "Poor handling of emotions"]
            }
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(default_descriptions, f, indent=2)
        except Exception as e:
            logger.error(f"Error creating default descriptions file: {str(e)}")
    
    def _get_fallback_questions(self) -> List[Dict[str, Any]]:
        """Return a minimal set of questions as fallback"""
        return [
            {
                "id": 1,
                "text": "You prefer to:",
                "options": [
                    {"key": "A", "text": "Interact with many people", "dimension": "E"},
                    {"key": "B", "text": "Interact with a few people", "dimension": "I"}
                ]
            },
            {
                "id": 2,
                "text": "You are more:",
                "options": [
                    {"key": "A", "text": "Practical", "dimension": "S"},
                    {"key": "B", "text": "Imaginative", "dimension": "N"}
                ]
            },
            {
                "id": 3,
                "text": "You make decisions based on:",
                "options": [
                    {"key": "A", "text": "Logic", "dimension": "T"},
                    {"key": "B", "text": "Feelings", "dimension": "F"}
                ]
            },
            {
                "id": 4,
                "text": "You prefer things to be:",
                "options": [
                    {"key": "A", "text": "Planned and organized", "dimension": "J"},
                    {"key": "B", "text": "Flexible and spontaneous", "dimension": "P"}
                ]
            }
        ]
    
    def _get_fallback_descriptions(self) -> Dict[str, Any]:
        """Return a minimal set of descriptions as fallback"""
        return {
            "ISTJ": {"name": "The Inspector", "description": "Practical and fact-minded individual."},
            "ISFJ": {"name": "The Protector", "description": "Quiet, friendly, and responsible."},
            "INFJ": {"name": "The Counselor", "description": "Seeks meaning and connection."},
            "INTJ": {"name": "The Mastermind", "description": "Strategic planner with original ideas."},
            "ISTP": {"name": "The Craftsman", "description": "Tolerant and flexible problem-solver."},
            "ISFP": {"name": "The Composer", "description": "Quiet, friendly, sensitive, and kind."},
            "INFP": {"name": "The Healer", "description": "Idealistic and loyal to values."},
            "INTP": {"name": "The Architect", "description": "Logical and theoretical thinker."},
            "ESTP": {"name": "The Dynamo", "description": "Energetic problem-solver focused on results."},
            "ESFP": {"name": "The Performer", "description": "Outgoing, friendly, and accepting."},
            "ENFP": {"name": "The Champion", "description": "Enthusiastic, imaginative, and sees possibilities."},
            "ENTP": {"name": "The Visionary", "description": "Quick, ingenious, and outspoken."},
            "ESTJ": {"name": "The Supervisor", "description": "Practical, decisive, and organized."},
            "ESFJ": {"name": "The Provider", "description": "Warmhearted, conscientious, and cooperative."},
            "ENFJ": {"name": "The Teacher", "description": "Warm, empathetic, and responsive."},
            "ENTJ": {"name": "The Commander", "description": "Frank, decisive, and assumes leadership."}
        }
    
    def get_questions(self, num_questions: int = 20) -> List[Dict[str, Any]]:
        """
        Get a specified number of MBTI questions
        
        Args:
            num_questions: Number of questions to return (default: 20)
            
        Returns:
            List of question dictionaries
        """
        # Ensure we don't request more questions than available
        num_questions = min(num_questions, len(self.questions))
        
        # Ensure we have a balanced set of questions for each dimension
        dimensions = ["E/I", "S/N", "T/F", "J/P"]
        questions_per_dimension = num_questions // 4
        remaining = num_questions % 4
        
        selected_questions = []
        
        # Group questions by dimension
        dimension_questions = {dim: [] for dim in dimensions}
        
        for question in self.questions:
            # Determine the dimension based on the first option
            if len(question["options"]) >= 1:
                option_dimension = question["options"][0]["dimension"]
                if option_dimension in "EI":
                    dimension_questions["E/I"].append(question)
                elif option_dimension in "SN":
                    dimension_questions["S/N"].append(question)
                elif option_dimension in "TF":
                    dimension_questions["T/F"].append(question)
                elif option_dimension in "JP":
                    dimension_questions["J/P"].append(question)
        
        # Select questions for each dimension
        for dimension in dimensions:
            available = dimension_questions[dimension]
            # If we don't have enough questions for this dimension, use what we have
            count = min(questions_per_dimension, len(available))
            # Randomly select questions to avoid always using the same ones
            selected = random.sample(available, count) if count > 0 else []
            selected_questions.extend(selected)
        
        # Add remaining questions if needed
        if remaining > 0:
            remaining_pool = []
            for dimension in dimensions:
                available = dimension_questions[dimension]
                used = [q for q in selected_questions if q in available]
                remaining_pool.extend([q for q in available if q not in used])
            
            if remaining_pool:
                additional = random.sample(remaining_pool, min(remaining, len(remaining_pool)))
                selected_questions.extend(additional)
        
        return selected_questions
    
    def compute_results(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute MBTI personality assessment results
        
        Args:
            responses: Dictionary containing user responses to assessment questions
            
        Returns:
            Dictionary containing assessment results
        """
        try:
            # Initialize counters for each dimension
            dimension_counts = {
                "E": 0, "I": 0,
                "S": 0, "N": 0,
                "T": 0, "F": 0,
                "J": 0, "P": 0
            }
            
            # Process each response
            for question_id, response_key in responses.items():
                # Find the corresponding question
                question = next((q for q in self.questions if str(q["id"]) == str(question_id)), None)
                
                if question:
                    # Find the selected option
                    selected_option = next((opt for opt in question["options"] if opt["key"] == response_key), None)
                    
                    if selected_option and "dimension" in selected_option:
                        dimension = selected_option["dimension"]
                        dimension_counts[dimension] += 1
            
            # Determine the personality type
            personality_type = ""
            personality_type += "E" if dimension_counts["E"] >= dimension_counts["I"] else "I"
            personality_type += "S" if dimension_counts["S"] >= dimension_counts["N"] else "N"
            personality_type += "T" if dimension_counts["T"] >= dimension_counts["F"] else "F"
            personality_type += "J" if dimension_counts["J"] >= dimension_counts["P"] else "P"
            
            # Calculate percentages for each dimension
            total_ei = dimension_counts["E"] + dimension_counts["I"]
            total_sn = dimension_counts["S"] + dimension_counts["N"]
            total_tf = dimension_counts["T"] + dimension_counts["F"]
            total_jp = dimension_counts["J"] + dimension_counts["P"]
            
            percentages = {
                "E": (dimension_counts["E"] / total_ei * 100) if total_ei > 0 else 50,
                "I": (dimension_counts["I"] / total_ei * 100) if total_ei > 0 else 50,
                "S": (dimension_counts["S"] / total_sn * 100) if total_sn > 0 else 50,
                "N": (dimension_counts["N"] / total_sn * 100) if total_sn > 0 else 50,
                "T": (dimension_counts["T"] / total_tf * 100) if total_tf > 0 else 50,
                "F": (dimension_counts["F"] / total_tf * 100) if total_tf > 0 else 50,
                "J": (dimension_counts["J"] / total_jp * 100) if total_jp > 0 else 50,
                "P": (dimension_counts["P"] / total_jp * 100) if total_jp > 0 else 50
            }
            
            # Get the type description
            type_info = self.type_descriptions.get(personality_type, {
                "name": f"Type {personality_type}",
                "description": "No detailed description available for this type."
            })
            
            # Prepare the results
            results = {
                "model": "MBTI",
                "type": personality_type,
                "type_name": type_info.get("name", f"Type {personality_type}"),
                "description": type_info.get("description", ""),
                "strengths": type_info.get("strengths", []),
                "weaknesses": type_info.get("weaknesses", []),
                "dimensions": {
                    "E/I": {
                        "preference": "E" if dimension_counts["E"] >= dimension_counts["I"] else "I",
                        "scores": {
                            "E": dimension_counts["E"],
                            "I": dimension_counts["I"]
                        },
                        "percentages": {
                            "E": percentages["E"],
                            "I": percentages["I"]
                        }
                    },
                    "S/N": {
                        "preference": "S" if dimension_counts["S"] >= dimension_counts["N"] else "N",
                        "scores": {
                            "S": dimension_counts["S"],
                            "N": dimension_counts["N"]
                        },
                        "percentages": {
                            "S": percentages["S"],
                            "N": percentages["N"]
                        }
                    },
                    "T/F": {
                        "preference": "T" if dimension_counts["T"] >= dimension_counts["F"] else "F",
                        "scores": {
                            "T": dimension_counts["T"],
                            "F": dimension_counts["F"]
                        },
                        "percentages": {
                            "T": percentages["T"],
                            "F": percentages["F"]
                        }
                    },
                    "J/P": {
                        "preference": "J" if dimension_counts["J"] >= dimension_counts["P"] else "P",
                        "scores": {
                            "J": dimension_counts["J"],
                            "P": dimension_counts["P"]
                        },
                        "percentages": {
                            "J": percentages["J"],
                            "P": percentages["P"]
                        }
                    }
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing MBTI results: {str(e)}")
            return {
                "error": str(e),
                "message": "Failed to compute personality assessment results"
            }
