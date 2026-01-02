from typing import Dict, Any, List, Optional
import logging
import json
import os
import sys
import importlib.util

logger = logging.getLogger(__name__)

class BigFiveAssessment:
    """
    Implementation of the Big Five (OCEAN) personality assessment using the five-factor-e library.
    """

    def __init__(self):
        """Initialize the Big Five assessment"""
        self.questions = self._load_questions()
        self.has_five_factor_e = self._check_five_factor_e()

    def _check_five_factor_e(self) -> bool:
        """Check if the five-factor-e library is installed"""
        try:
            spec = importlib.util.find_spec('ipipneo')
            if spec is None:
                logger.warning("five-factor-e library not found. Using fallback implementation.")
                return False
            return True
        except ImportError:
            logger.warning("Error importing five-factor-e. Using fallback implementation.")
            return False

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load the Big Five questions from the data file"""
        try:
            # Try to load questions from the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'personality')
            os.makedirs(data_dir, exist_ok=True)

            questions_path = os.path.join(data_dir, 'big_five_questions.json')

            # If the file doesn't exist, create it with default questions
            if not os.path.exists(questions_path):
                self._create_default_questions(questions_path)

            with open(questions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading Big Five questions: {str(e)}")
            # Return a minimal set of questions as fallback
            return self._get_fallback_questions()

    def _create_default_questions(self, file_path: str) -> None:
        """Create a default questions file"""
        # This is a comprehensive version based on the IPIP-NEO and bigfive-test.com
        default_questions = [
            # Extraversion questions
            {
                "id": 1,
                "text": "I am the life of the party.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "gregariousness"
            },
            {
                "id": 2,
                "text": "I don't talk a lot.",
                "trait": "extraversion",
                "reversed": True,
                "facet": "gregariousness"
            },
            {
                "id": 3,
                "text": "I feel comfortable around people.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "warmth"
            },
            {
                "id": 4,
                "text": "I keep in the background.",
                "trait": "extraversion",
                "reversed": True,
                "facet": "assertiveness"
            },
            {
                "id": 5,
                "text": "I start conversations.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "gregariousness"
            },
            {
                "id": 6,
                "text": "I have little to say.",
                "trait": "extraversion",
                "reversed": True,
                "facet": "gregariousness"
            },
            {
                "id": 7,
                "text": "I talk to a lot of different people at parties.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "gregariousness"
            },
            {
                "id": 8,
                "text": "I don't like to draw attention to myself.",
                "trait": "extraversion",
                "reversed": True,
                "facet": "assertiveness"
            },
            {
                "id": 9,
                "text": "I don't mind being the center of attention.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "assertiveness"
            },
            {
                "id": 10,
                "text": "I am quiet around strangers.",
                "trait": "extraversion",
                "reversed": True,
                "facet": "warmth"
            },
            {
                "id": 11,
                "text": "I get excited by new ideas and projects.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "excitement-seeking"
            },
            {
                "id": 12,
                "text": "I prefer one-on-one conversations to group activities.",
                "trait": "extraversion",
                "reversed": True,
                "facet": "gregariousness"
            },
            {
                "id": 61,
                "text": "I find it easy to approach and talk with strangers.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "sociability"
            },
            {
                "id": 62,
                "text": "I enjoy being part of a loud, energetic crowd.",
                "trait": "extraversion",
                "reversed": False,
                "facet": "gregariousness"
            },
            {
                "id": 63,
                "text": "I prefer a quiet evening at home to a big social event.",
                "trait": "extraversion",
                "reversed": True,
                "facet": "sociability"
            },

            # Agreeableness questions
            {
                "id": 13,
                "text": "I feel little concern for others.",
                "trait": "agreeableness",
                "reversed": True,
                "facet": "altruism"
            },
            {
                "id": 14,
                "text": "I am interested in people.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "warmth"
            },
            {
                "id": 15,
                "text": "I insult people.",
                "trait": "agreeableness",
                "reversed": True,
                "facet": "cooperation"
            },
            {
                "id": 16,
                "text": "I sympathize with others' feelings.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "empathy"
            },
            {
                "id": 17,
                "text": "I am not interested in other people's problems.",
                "trait": "agreeableness",
                "reversed": True,
                "facet": "altruism"
            },
            {
                "id": 18,
                "text": "I have a soft heart.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "empathy"
            },
            {
                "id": 19,
                "text": "I am not really interested in others.",
                "trait": "agreeableness",
                "reversed": True,
                "facet": "warmth"
            },
            {
                "id": 20,
                "text": "I take time out for others.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "altruism"
            },
            {
                "id": 21,
                "text": "I feel others' emotions.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "empathy"
            },
            {
                "id": 22,
                "text": "I make people feel at ease.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "warmth"
            },
            {
                "id": 23,
                "text": "I am hard to get to know.",
                "trait": "agreeableness",
                "reversed": True,
                "facet": "warmth"
            },
            {
                "id": 24,
                "text": "I tend to trust people and give them the benefit of the doubt.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "trust"
            },
            {
                "id": 64,
                "text": "I believe most people have good intentions.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "trust"
            },
            {
                "id": 65,
                "text": "I am willing to compromise my own comfort to help others.",
                "trait": "agreeableness",
                "reversed": False,
                "facet": "altruism"
            },
            {
                "id": 66,
                "text": "I find it hard to forgive people who have hurt me.",
                "trait": "agreeableness",
                "reversed": True,
                "facet": "forgiveness"
            },

            # Conscientiousness questions
            {
                "id": 25,
                "text": "I am always prepared.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "self-discipline"
            },
            {
                "id": 26,
                "text": "I leave my belongings around.",
                "trait": "conscientiousness",
                "reversed": True,
                "facet": "orderliness"
            },
            {
                "id": 27,
                "text": "I pay attention to details.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "orderliness"
            },
            {
                "id": 28,
                "text": "I make a mess of things.",
                "trait": "conscientiousness",
                "reversed": True,
                "facet": "orderliness"
            },
            {
                "id": 29,
                "text": "I get chores done right away.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "dutifulness"
            },
            {
                "id": 30,
                "text": "I often forget to put things back in their proper place.",
                "trait": "conscientiousness",
                "reversed": True,
                "facet": "orderliness"
            },
            {
                "id": 31,
                "text": "I like order.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "orderliness"
            },
            {
                "id": 32,
                "text": "I shirk my duties.",
                "trait": "conscientiousness",
                "reversed": True,
                "facet": "dutifulness"
            },
            {
                "id": 33,
                "text": "I follow a schedule.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "self-discipline"
            },
            {
                "id": 34,
                "text": "I am exacting in my work.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "achievement-striving"
            },
            {
                "id": 35,
                "text": "I leave a mess in my room.",
                "trait": "conscientiousness",
                "reversed": True,
                "facet": "orderliness"
            },
            {
                "id": 36,
                "text": "I keep my promises.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "dutifulness"
            },
            {
                "id": 67,
                "text": "I set high standards for myself and others.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "achievement-striving"
            },
            {
                "id": 68,
                "text": "I often rush into things without thinking them through.",
                "trait": "conscientiousness",
                "reversed": True,
                "facet": "deliberation"
            },
            {
                "id": 69,
                "text": "I stick to my plans even when it's difficult.",
                "trait": "conscientiousness",
                "reversed": False,
                "facet": "self-discipline"
            },

            # Neuroticism questions
            {
                "id": 37,
                "text": "I get stressed out easily.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "anxiety"
            },
            {
                "id": 38,
                "text": "I am relaxed most of the time.",
                "trait": "neuroticism",
                "reversed": True,
                "facet": "anxiety"
            },
            {
                "id": 39,
                "text": "I worry about things.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "anxiety"
            },
            {
                "id": 40,
                "text": "I seldom feel blue.",
                "trait": "neuroticism",
                "reversed": True,
                "facet": "depression"
            },
            {
                "id": 41,
                "text": "I am easily disturbed.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "vulnerability"
            },
            {
                "id": 42,
                "text": "I get upset easily.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "anger"
            },
            {
                "id": 43,
                "text": "I change my mood a lot.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "emotional-volatility"
            },
            {
                "id": 44,
                "text": "I have frequent mood swings.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "emotional-volatility"
            },
            {
                "id": 45,
                "text": "I get irritated easily.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "anger"
            },
            {
                "id": 46,
                "text": "I often feel blue.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "depression"
            },
            {
                "id": 47,
                "text": "I am much more anxious than most people.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "anxiety"
            },
            {
                "id": 48,
                "text": "I remain calm under pressure.",
                "trait": "neuroticism",
                "reversed": True,
                "facet": "vulnerability"
            },
            {
                "id": 70,
                "text": "I often experience feelings of helplessness or sadness.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "depression"
            },
            {
                "id": 71,
                "text": "I tend to feel uncomfortable in unfamiliar situations.",
                "trait": "neuroticism",
                "reversed": False,
                "facet": "anxiety"
            },
            {
                "id": 72,
                "text": "I find it easy to bounce back after disappointments.",
                "trait": "neuroticism",
                "reversed": True,
                "facet": "resilience"
            },

            # Openness questions
            {
                "id": 49,
                "text": "I have a rich vocabulary.",
                "trait": "openness",
                "reversed": False,
                "facet": "intellect"
            },
            {
                "id": 50,
                "text": "I have difficulty understanding abstract ideas.",
                "trait": "openness",
                "reversed": True,
                "facet": "intellect"
            },
            {
                "id": 51,
                "text": "I have a vivid imagination.",
                "trait": "openness",
                "reversed": False,
                "facet": "imagination"
            },
            {
                "id": 52,
                "text": "I am not interested in abstract ideas.",
                "trait": "openness",
                "reversed": True,
                "facet": "intellect"
            },
            {
                "id": 53,
                "text": "I have excellent ideas.",
                "trait": "openness",
                "reversed": False,
                "facet": "intellect"
            },
            {
                "id": 54,
                "text": "I do not have a good imagination.",
                "trait": "openness",
                "reversed": True,
                "facet": "imagination"
            },
            {
                "id": 55,
                "text": "I am quick to understand things.",
                "trait": "openness",
                "reversed": False,
                "facet": "intellect"
            },
            {
                "id": 56,
                "text": "I use difficult words.",
                "trait": "openness",
                "reversed": False,
                "facet": "intellect"
            },
            {
                "id": 57,
                "text": "I spend time reflecting on things.",
                "trait": "openness",
                "reversed": False,
                "facet": "introspection"
            },
            {
                "id": 58,
                "text": "I am full of ideas.",
                "trait": "openness",
                "reversed": False,
                "facet": "creativity"
            },
            {
                "id": 59,
                "text": "I enjoy thinking about complex problems.",
                "trait": "openness",
                "reversed": False,
                "facet": "intellect"
            },
            {
                "id": 60,
                "text": "I avoid philosophical discussions.",
                "trait": "openness",
                "reversed": True,
                "facet": "intellect"
            },
            {
                "id": 73,
                "text": "I enjoy experiencing new cultures and customs.",
                "trait": "openness",
                "reversed": False,
                "facet": "adventurousness"
            },
            {
                "id": 74,
                "text": "I prefer familiar routines to exploring new activities.",
                "trait": "openness",
                "reversed": True,
                "facet": "adventurousness"
            },
            {
                "id": 75,
                "text": "I enjoy artistic and creative pursuits.",
                "trait": "openness",
                "reversed": False,
                "facet": "artistic-interests"
            }
        ]

        try:
            with open(file_path, 'w') as f:
                json.dump(default_questions, f, indent=2)
        except Exception as e:
            logger.error(f"Error creating default questions file: {str(e)}")

    def _get_fallback_questions(self) -> List[Dict[str, Any]]:
        """Return a minimal set of questions as fallback"""
        return [
            {
                "id": 1,
                "text": "I am outgoing and sociable.",
                "trait": "extraversion",
                "reversed": False
            },
            {
                "id": 2,
                "text": "I am compassionate and kind to others.",
                "trait": "agreeableness",
                "reversed": False
            },
            {
                "id": 3,
                "text": "I am organized and detail-oriented.",
                "trait": "conscientiousness",
                "reversed": False
            },
            {
                "id": 4,
                "text": "I tend to worry and feel anxious.",
                "trait": "neuroticism",
                "reversed": False
            },
            {
                "id": 5,
                "text": "I am creative and open to new experiences.",
                "trait": "openness",
                "reversed": False
            }
        ]

    def get_questions(self, num_questions: int = 50) -> List[Dict[str, Any]]:
        """
        Get a specified number of Big Five questions

        Args:
            num_questions: Number of questions to return (default: 50)

        Returns:
            List of question dictionaries
        """
        # Ensure we don't request more questions than available
        num_questions = min(num_questions, len(self.questions))

        # Return a balanced set of questions across all traits
        questions_per_trait = num_questions // 5
        remaining = num_questions % 5

        selected_questions = []
        traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]

        for trait in traits:
            trait_questions = [q for q in self.questions if q["trait"] == trait]
            # Take an equal number of questions for each trait
            selected_questions.extend(trait_questions[:questions_per_trait])

        # Add remaining questions
        if remaining > 0:
            remaining_questions = []
            for trait in traits[:remaining]:
                trait_questions = [q for q in self.questions if q["trait"] == trait]
                if len(trait_questions) > questions_per_trait:
                    remaining_questions.append(trait_questions[questions_per_trait])
            selected_questions.extend(remaining_questions)

        return selected_questions

    def compute_results(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Big Five personality assessment results

        Args:
            responses: Dictionary containing user responses to assessment questions

        Returns:
            Dictionary containing assessment results
        """
        try:
            # If five-factor-e is available, use it for computation
            if self.has_five_factor_e:
                return self._compute_with_five_factor_e(responses)
            else:
                # Otherwise use our simplified implementation
                return self._compute_simplified(responses)
        except Exception as e:
            logger.error(f"Error computing Big Five results: {str(e)}")
            return {
                "error": str(e),
                "message": "Failed to compute personality assessment results"
            }

    def _compute_with_five_factor_e(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Compute results using the five-factor-e library"""
        try:
            # Try to import ipipneo
            try:
                from ipipneo import IpipNeo
                has_library = True
            except ImportError:
                has_library = False
                logger.warning("The 'ipipneo' module from five-factor-e library could not be imported.")
                return self._compute_simplified(responses)

            # Proceed only if import was successful
            if not has_library:
                return self._compute_simplified(responses)

            # Convert our response format to five-factor-e format
            ipip_responses = {"answers": []}

            for question_id, response_value in responses.items():
                ipip_responses["answers"].append({
                    "id_question": int(question_id),
                    "id_select": response_value
                })

            # Use the library to compute results
            ipip = IpipNeo(question=120)  # Use the 120-question version
            results = ipip.compute(sex="M", age=30, answers=ipip_responses)  # Default values

            # Process and return the results
            return self._process_five_factor_e_results(results)
        except Exception as e:
            logger.error(f"Error using five-factor-e: {str(e)}")
            # Fall back to simplified computation
            return self._compute_simplified(responses)

    def _process_five_factor_e_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format results from five-factor-e"""
        processed_results = {
            "model": "five-factor-e (IPIP-NEO)",
            "traits": {}
        }

        try:
            # Extract the main personality traits
            personalities = results.get("personalities", [])

            for personality in personalities:
                # Process each of the Big Five traits
                for trait_key in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                    if trait_key in personality:
                        trait_data = personality[trait_key]

                        # Get the main score and facets
                        main_score = trait_data.get(trait_key[0].upper(), 0)
                        score_category = trait_data.get("score", "average")

                        # Process facets/traits
                        facets = []
                        for facet in trait_data.get("traits", []):
                            facet_name = next((k for k in facet.keys() if k not in ["trait", "score"]), None)
                            if facet_name:
                                facets.append({
                                    "name": facet_name,
                                    "score": facet.get(facet_name, 0),
                                    "category": facet.get("score", "average")
                                })

                        # Add to processed results
                        processed_results["traits"][trait_key] = {
                            "score": main_score,
                            "category": score_category,
                            "facets": facets
                        }

            return processed_results
        except Exception as e:
            logger.error(f"Error processing five-factor-e results: {str(e)}")
            return {
                "error": str(e),
                "raw_results": results
            }

    def _compute_simplified(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Compute results using a simplified algorithm"""
        # Initialize scores for each trait
        trait_scores = {
            "openness": [],
            "conscientiousness": [],
            "extraversion": [],
            "agreeableness": [],
            "neuroticism": []
        }

        # Process each response
        for question_id, response_value in responses.items():
            # Find the corresponding question
            question = next((q for q in self.questions if str(q["id"]) == str(question_id)), None)

            if question:
                trait = question["trait"]
                score = int(response_value)

                # Adjust score if the question is reversed
                if question.get("reversed", False):
                    score = 6 - score  # Reverse on a 1-5 scale

                # Add to the appropriate trait
                if trait in trait_scores:
                    trait_scores[trait].append(score)

        # Calculate average scores and categories
        results = {
            "model": "simplified Big Five",
            "traits": {}
        }

        for trait, scores in trait_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                # Convert to percentile (simplified)
                percentile = (avg_score - 1) / 4 * 100

                # Determine category
                if percentile < 30:
                    category = "low"
                elif percentile > 70:
                    category = "high"
                else:
                    category = "average"

                results["traits"][trait] = {
                    "score": percentile,
                    "raw_score": avg_score,
                    "category": category
                }
            else:
                # No scores for this trait
                results["traits"][trait] = {
                    "score": 50,  # Default to middle
                    "raw_score": 3,
                    "category": "average"
                }

        return results
