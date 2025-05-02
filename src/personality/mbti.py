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
        """Create a default questions file based on 16personalities.com format"""
        default_questions = [
            # Extraversion vs. Introversion (E/I) questions
            {
                "id": 1,
                "text": "At social events, you:",
                "options": [
                    {"key": "A", "text": "Interact with many people, including strangers", "dimension": "E"},
                    {"key": "B", "text": "Interact with a few people you know well", "dimension": "I"}
                ],
                "category": "E/I"
            },
            {
                "id": 2,
                "text": "You tend to:",
                "options": [
                    {"key": "A", "text": "Think out loud and get energized by conversation", "dimension": "E"},
                    {"key": "B", "text": "Think things through inside your head first", "dimension": "I"}
                ],
                "category": "E/I"
            },
            {
                "id": 3,
                "text": "When you're in a group, you usually prefer to:",
                "options": [
                    {"key": "A", "text": "Join in and contribute to the conversation", "dimension": "E"},
                    {"key": "B", "text": "Listen and observe more than speak", "dimension": "I"}
                ],
                "category": "E/I"
            },
            {
                "id": 4,
                "text": "After spending time with a large group of people, you typically feel:",
                "options": [
                    {"key": "A", "text": "Energized and wanting more social interaction", "dimension": "E"},
                    {"key": "B", "text": "Drained and needing alone time to recharge", "dimension": "I"}
                ],
                "category": "E/I"
            },
            {
                "id": 5,
                "text": "You consider yourself more of a:",
                "options": [
                    {"key": "A", "text": "Talker who thinks while speaking", "dimension": "E"},
                    {"key": "B", "text": "Listener who thinks before speaking", "dimension": "I"}
                ],
                "category": "E/I"
            },
            {
                "id": 6,
                "text": "Your ideal weekend would involve:",
                "options": [
                    {"key": "A", "text": "Going out with friends or to social events", "dimension": "E"},
                    {"key": "B", "text": "Spending quiet time at home or with a few close friends", "dimension": "I"}
                ],
                "category": "E/I"
            },
            {
                "id": 7,
                "text": "When facing a challenge, you prefer to:",
                "options": [
                    {"key": "A", "text": "Discuss it with others to work through solutions", "dimension": "E"},
                    {"key": "B", "text": "Think it through on your own first", "dimension": "I"}
                ],
                "category": "E/I"
            },

            # Sensing vs. Intuition (S/N) questions
            {
                "id": 8,
                "text": "You are more interested in:",
                "options": [
                    {"key": "A", "text": "What is real and concrete in the present", "dimension": "S"},
                    {"key": "B", "text": "What could be possible in the future", "dimension": "N"}
                ],
                "category": "S/N"
            },
            {
                "id": 9,
                "text": "You prefer explanations that are:",
                "options": [
                    {"key": "A", "text": "Practical and focused on facts", "dimension": "S"},
                    {"key": "B", "text": "Theoretical and focused on ideas", "dimension": "N"}
                ],
                "category": "S/N"
            },
            {
                "id": 10,
                "text": "When learning something new, you prefer to:",
                "options": [
                    {"key": "A", "text": "Master practical skills through hands-on experience", "dimension": "S"},
                    {"key": "B", "text": "Understand the underlying concepts and theories", "dimension": "N"}
                ],
                "category": "S/N"
            },
            {
                "id": 11,
                "text": "You tend to focus more on:",
                "options": [
                    {"key": "A", "text": "Details and specific facts", "dimension": "S"},
                    {"key": "B", "text": "The big picture and patterns", "dimension": "N"}
                ],
                "category": "S/N"
            },
            {
                "id": 12,
                "text": "You would describe yourself as more:",
                "options": [
                    {"key": "A", "text": "Practical and grounded", "dimension": "S"},
                    {"key": "B", "text": "Imaginative and innovative", "dimension": "N"}
                ],
                "category": "S/N"
            },
            {
                "id": 13,
                "text": "You are more likely to trust:",
                "options": [
                    {"key": "A", "text": "Your direct experience and observations", "dimension": "S"},
                    {"key": "B", "text": "Your intuition and hunches", "dimension": "N"}
                ],
                "category": "S/N"
            },
            {
                "id": 14,
                "text": "When solving problems, you prefer to:",
                "options": [
                    {"key": "A", "text": "Use proven methods and established solutions", "dimension": "S"},
                    {"key": "B", "text": "Find new approaches and innovative solutions", "dimension": "N"}
                ],
                "category": "S/N"
            },

            # Thinking vs. Feeling (T/F) questions
            {
                "id": 15,
                "text": "When making decisions, you typically prioritize:",
                "options": [
                    {"key": "A", "text": "Logic, consistency, and objective analysis", "dimension": "T"},
                    {"key": "B", "text": "People's feelings and maintaining harmony", "dimension": "F"}
                ],
                "category": "T/F"
            },
            {
                "id": 16,
                "text": "You are more convinced by:",
                "options": [
                    {"key": "A", "text": "A strong logical argument", "dimension": "T"},
                    {"key": "B", "text": "A compelling emotional appeal", "dimension": "F"}
                ],
                "category": "T/F"
            },
            {
                "id": 17,
                "text": "When giving feedback, you tend to be:",
                "options": [
                    {"key": "A", "text": "Direct and straightforward about what needs improvement", "dimension": "T"},
                    {"key": "B", "text": "Tactful and encouraging, focusing on the positive", "dimension": "F"}
                ],
                "category": "T/F"
            },
            {
                "id": 18,
                "text": "In conflicts, you are more concerned with:",
                "options": [
                    {"key": "A", "text": "Finding the objectively fair solution", "dimension": "T"},
                    {"key": "B", "text": "Making sure everyone's feelings are considered", "dimension": "F"}
                ],
                "category": "T/F"
            },
            {
                "id": 19,
                "text": "You value more highly:",
                "options": [
                    {"key": "A", "text": "Truth, even if it might hurt someone's feelings", "dimension": "T"},
                    {"key": "B", "text": "Compassion, even if it means bending the truth", "dimension": "F"}
                ],
                "category": "T/F"
            },
            {
                "id": 20,
                "text": "You consider yourself more:",
                "options": [
                    {"key": "A", "text": "Analytical and objective", "dimension": "T"},
                    {"key": "B", "text": "Empathetic and understanding", "dimension": "F"}
                ],
                "category": "T/F"
            },
            {
                "id": 21,
                "text": "When someone comes to you with a problem, you're more likely to:",
                "options": [
                    {"key": "A", "text": "Help them analyze the situation and find a solution", "dimension": "T"},
                    {"key": "B", "text": "Listen and provide emotional support", "dimension": "F"}
                ],
                "category": "T/F"
            },

            # Judging vs. Perceiving (J/P) questions
            {
                "id": 22,
                "text": "You prefer:",
                "options": [
                    {"key": "A", "text": "Having a detailed plan and sticking to it", "dimension": "J"},
                    {"key": "B", "text": "Going with the flow and adapting as you go", "dimension": "P"}
                ],
                "category": "J/P"
            },
            {
                "id": 23,
                "text": "You find it more satisfying to:",
                "options": [
                    {"key": "A", "text": "Complete a project and check it off your list", "dimension": "J"},
                    {"key": "B", "text": "Start a new project with fresh possibilities", "dimension": "P"}
                ],
                "category": "J/P"
            },
            {
                "id": 24,
                "text": "Your workspace is usually:",
                "options": [
                    {"key": "A", "text": "Organized and structured", "dimension": "J"},
                    {"key": "B", "text": "Flexible and adaptable to what you're working on", "dimension": "P"}
                ],
                "category": "J/P"
            },
            {
                "id": 25,
                "text": "You prefer to:",
                "options": [
                    {"key": "A", "text": "Make decisions and settle matters quickly", "dimension": "J"},
                    {"key": "B", "text": "Keep options open and decide later", "dimension": "P"}
                ],
                "category": "J/P"
            },
            {
                "id": 26,
                "text": "You tend to:",
                "options": [
                    {"key": "A", "text": "Plan ahead and follow schedules", "dimension": "J"},
                    {"key": "B", "text": "Be spontaneous and adapt to circumstances", "dimension": "P"}
                ],
                "category": "J/P"
            },
            {
                "id": 27,
                "text": "You feel more comfortable when:",
                "options": [
                    {"key": "A", "text": "Things are decided and settled", "dimension": "J"},
                    {"key": "B", "text": "Options remain open and flexible", "dimension": "P"}
                ],
                "category": "J/P"
            },
            {
                "id": 28,
                "text": "When it comes to deadlines, you typically:",
                "options": [
                    {"key": "A", "text": "Complete work well ahead of time", "dimension": "J"},
                    {"key": "B", "text": "Work in bursts of energy, often close to the deadline", "dimension": "P"}
                ],
                "category": "J/P"
            },

            # Additional E/I questions
            {
                "id": 29,
                "text": "You find phone calls with friends:",
                "options": [
                    {"key": "A", "text": "Energizing and you often initiate them", "dimension": "E"},
                    {"key": "B", "text": "Sometimes draining and you often let them go to voicemail", "dimension": "I"}
                ],
                "category": "E/I"
            },
            {
                "id": 30,
                "text": "You prefer working:",
                "options": [
                    {"key": "A", "text": "In a team environment with lots of interaction", "dimension": "E"},
                    {"key": "B", "text": "Independently or with minimal interruptions", "dimension": "I"}
                ],
                "category": "E/I"
            },

            # Additional S/N questions
            {
                "id": 31,
                "text": "You are more drawn to:",
                "options": [
                    {"key": "A", "text": "Realistic stories about actual events", "dimension": "S"},
                    {"key": "B", "text": "Fantasy and science fiction that explore possibilities", "dimension": "N"}
                ],
                "category": "S/N"
            },
            {
                "id": 32,
                "text": "You find more value in people who are:",
                "options": [
                    {"key": "A", "text": "Practical and reliable", "dimension": "S"},
                    {"key": "B", "text": "Imaginative and insightful", "dimension": "N"}
                ],
                "category": "S/N"
            },

            # Additional T/F questions
            {
                "id": 33,
                "text": "You find it easier to:",
                "options": [
                    {"key": "A", "text": "Apply logical analysis to problems", "dimension": "T"},
                    {"key": "B", "text": "Understand how people feel and what they need", "dimension": "F"}
                ],
                "category": "T/F"
            },
            {
                "id": 34,
                "text": "You are more likely to be described as:",
                "options": [
                    {"key": "A", "text": "Fair and reasonable", "dimension": "T"},
                    {"key": "B", "text": "Warm and compassionate", "dimension": "F"}
                ],
                "category": "T/F"
            },

            # Additional J/P questions
            {
                "id": 35,
                "text": "You prefer:",
                "options": [
                    {"key": "A", "text": "Clear expectations and structure", "dimension": "J"},
                    {"key": "B", "text": "Freedom to adapt and change course", "dimension": "P"}
                ],
                "category": "J/P"
            },
            {
                "id": 36,
                "text": "You find more satisfaction in:",
                "options": [
                    {"key": "A", "text": "Following through on commitments", "dimension": "J"},
                    {"key": "B", "text": "Exploring new possibilities", "dimension": "P"}
                ],
                "category": "J/P"
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
        Get a specified number of MBTI questions, balanced across all dimensions
        
        Args:
            num_questions: Number of questions to return (default: 20)
            
        Returns:
            List of question dictionaries
        """
        # Ensure we don't request more questions than available
        num_questions = min(num_questions, len(self.questions))
        
        # Create dictionary to store questions by dimension
        dimension_questions = {
            "E/I": [],
            "S/N": [],
            "T/F": [],
            "J/P": []
        }
        
        # Categorize questions by dimension
        dimensions = ["E/I", "S/N", "T/F", "J/P"]
        
        # Sort questions into appropriate dimensions
        for question in self.questions:
            if "category" in question:
                dimension = question["category"]
                if dimension in dimensions:
                    dimension_questions[dimension].append(question)
            else:
                # For questions without a category, check the first option's dimension
                if "options" in question and len(question["options"]) > 0:
                    option_dimension = question["options"][0]["dimension"]
                    if option_dimension in "EI":
                        dimension_questions["E/I"].append(question)
                    elif option_dimension in "SN":
                        dimension_questions["S/N"].append(question)
                    elif option_dimension in "TF":
                        dimension_questions["T/F"].append(question)
                    elif option_dimension in "JP":
                        dimension_questions["J/P"].append(question)

        # Calculate how many questions to select from each dimension
        questions_per_dimension = num_questions // 4
        remaining = num_questions % 4
        
        # Select questions for each dimension
        selected_questions = []
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
            Dictionary containing assessment results including MBTI type and scores
        """
        try:
            # Initialize dimension scores
            dimension_scores = {
                "E": 0, "I": 0,  # Extraversion vs. Introversion
                "S": 0, "N": 0,  # Sensing vs. Intuition
                "T": 0, "F": 0,  # Thinking vs. Feeling
                "J": 0, "P": 0   # Judging vs. Perceiving
            }
            
            # Count responses for each question
            for question_id, response_key in responses.items():
                # Find the corresponding question
                question = next((q for q in self.questions if str(q["id"]) == str(question_id)), None)
                
                if question and "options" in question:
                    # Find the selected option
                    selected_option = next((opt for opt in question["options"] if opt["key"] == response_key), None)
                    
                    if selected_option and "dimension" in selected_option:
                        # Increment the score for the selected dimension
                        dimension = selected_option["dimension"]
                        if dimension in dimension_scores:
                            dimension_scores[dimension] += 1
            
            # Determine the personality type based on the highest score in each dimension pair
            personality_type = ""
            
            # E/I dimension
            if dimension_scores["E"] > dimension_scores["I"]:
                personality_type += "E"
            else:
                personality_type += "I"
                
            # S/N dimension
            if dimension_scores["S"] > dimension_scores["N"]:
                personality_type += "S"
            else:
                personality_type += "N"
                
            # T/F dimension
            if dimension_scores["T"] > dimension_scores["F"]:
                personality_type += "T"
            else:
                personality_type += "F"
                
            # J/P dimension
            if dimension_scores["J"] > dimension_scores["P"]:
                personality_type += "J"
            else:
                personality_type += "P"
                
            # Calculate percentages for each dimension
            dimension_percentages = {
                "E/I": {
                    "E": self._calculate_percentage(dimension_scores["E"], dimension_scores["E"] + dimension_scores["I"]),
                    "I": self._calculate_percentage(dimension_scores["I"], dimension_scores["E"] + dimension_scores["I"])
                },
                "S/N": {
                    "S": self._calculate_percentage(dimension_scores["S"], dimension_scores["S"] + dimension_scores["N"]),
                    "N": self._calculate_percentage(dimension_scores["N"], dimension_scores["S"] + dimension_scores["N"])
                },
                "T/F": {
                    "T": self._calculate_percentage(dimension_scores["T"], dimension_scores["T"] + dimension_scores["F"]),
                    "F": self._calculate_percentage(dimension_scores["F"], dimension_scores["T"] + dimension_scores["F"])
                },
                "J/P": {
                    "J": self._calculate_percentage(dimension_scores["J"], dimension_scores["J"] + dimension_scores["P"]),
                    "P": self._calculate_percentage(dimension_scores["P"], dimension_scores["J"] + dimension_scores["P"])
                }
            }
            
            # Get the type description
            type_description = self.type_descriptions.get(personality_type, {})
            type_name = type_description.get("name", "Unknown Type")
            description = type_description.get("description", "No description available.")
            strengths = type_description.get("strengths", [])
            weaknesses = type_description.get("weaknesses", [])
            
            # Prepare and return the results
            results = {
                "type": personality_type,
                "type_name": type_name,
                "description": description,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "scores": dimension_scores,
                "dimensions": {
                    "E/I": {
                        "dominant": "E" if dimension_scores["E"] > dimension_scores["I"] else "I",
                        "scores": {
                            "E": dimension_scores["E"],
                            "I": dimension_scores["I"]
                        },
                        "percentages": dimension_percentages["E/I"]
                    },
                    "S/N": {
                        "dominant": "S" if dimension_scores["S"] > dimension_scores["N"] else "N",
                        "scores": {
                            "S": dimension_scores["S"],
                            "N": dimension_scores["N"]
                        },
                        "percentages": dimension_percentages["S/N"]
                    },
                    "T/F": {
                        "dominant": "T" if dimension_scores["T"] > dimension_scores["F"] else "F",
                        "scores": {
                            "T": dimension_scores["T"],
                            "F": dimension_scores["F"]
                        },
                        "percentages": dimension_percentages["T/F"]
                    },
                    "J/P": {
                        "dominant": "J" if dimension_scores["J"] > dimension_scores["P"] else "P",
                        "scores": {
                            "J": dimension_scores["J"],
                            "P": dimension_scores["P"]
                        },
                        "percentages": dimension_percentages["J/P"]
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
    
    def _calculate_percentage(self, score: int, total: int) -> float:
        """Calculate percentage, avoiding division by zero"""
        if total == 0:
            return 50.0  # Default to middle if no data
        return round((score / total) * 100, 1)
    
    def _get_fallback_descriptions(self) -> Dict[str, Any]:
        """Return a minimal set of descriptions as fallback"""
        return {
            # Just include a few common types as fallback
            "INTJ": {
                "name": "The Architect",
                "description": "Strategic, thoughtful, and detail-oriented planners with a vision for the future.",
                "strengths": ["Strategic thinking", "Independent", "Analytical"],
                "weaknesses": ["Perfectionist", "Critical", "Dismissive of emotions"]
            },
            "ENFP": {
                "name": "The Champion",
                "description": "Enthusiastic, creative, and sociable free spirits who find meaning and connection everywhere.",
                "strengths": ["Enthusiastic", "Creative", "Empathetic"],
                "weaknesses": ["Disorganized", "Overcommitted", "Trouble focusing"]
            },
            "ISFJ": {
                "name": "The Protector",
                "description": "Dedicated, warm-hearted protectors who respect traditions and seek to help others.",
                "strengths": ["Reliable", "Patient", "Supportive"],
                "weaknesses": ["Overly reserved", "Resistant to change", "Overworking"]
            }
        }
