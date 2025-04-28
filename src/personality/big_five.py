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
        # This is a shortened version of the IPIP-NEO-120 for demonstration
        default_questions = [
            {
                "id": 1,
                "text": "I am the life of the party.",
                "trait": "extraversion",
                "reversed": False
            },
            {
                "id": 2,
                "text": "I feel little concern for others.",
                "trait": "agreeableness",
                "reversed": True
            },
            {
                "id": 3,
                "text": "I am always prepared.",
                "trait": "conscientiousness",
                "reversed": False
            },
            {
                "id": 4,
                "text": "I get stressed out easily.",
                "trait": "neuroticism",
                "reversed": False
            },
            {
                "id": 5,
                "text": "I have a rich vocabulary.",
                "trait": "openness",
                "reversed": False
            },
            {
                "id": 6,
                "text": "I don't talk a lot.",
                "trait": "extraversion",
                "reversed": True
            },
            {
                "id": 7,
                "text": "I am interested in people.",
                "trait": "agreeableness",
                "reversed": False
            },
            {
                "id": 8,
                "text": "I leave my belongings around.",
                "trait": "conscientiousness",
                "reversed": True
            },
            {
                "id": 9,
                "text": "I am relaxed most of the time.",
                "trait": "neuroticism",
                "reversed": True
            },
            {
                "id": 10,
                "text": "I have difficulty understanding abstract ideas.",
                "trait": "openness",
                "reversed": True
            }
        ]
        
        # Add more questions for each trait to make it more comprehensive
        for i in range(11, 51):
            trait_index = (i - 1) % 5
            trait = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"][trait_index]
            reversed_score = (i % 2 == 0)
            
            question_text = f"Sample question {i} for {trait}."
            if trait == "extraversion":
                if reversed_score:
                    question_text = f"I prefer to spend time alone rather than in large groups."
                else:
                    question_text = f"I enjoy being the center of attention."
            elif trait == "agreeableness":
                if reversed_score:
                    question_text = f"I can be critical of others."
                else:
                    question_text = f"I sympathize with others' feelings."
            elif trait == "conscientiousness":
                if reversed_score:
                    question_text = f"I often forget to put things back in their proper place."
                else:
                    question_text = f"I pay attention to details."
            elif trait == "neuroticism":
                if reversed_score:
                    question_text = f"I rarely feel blue or sad."
                else:
                    question_text = f"I worry about things."
            elif trait == "openness":
                if reversed_score:
                    question_text = f"I am not interested in abstract ideas."
                else:
                    question_text = f"I have a vivid imagination."
            
            default_questions.append({
                "id": i,
                "text": question_text,
                "trait": trait,
                "reversed": reversed_score
            })
        
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
            from ipipneo import IpipNeo
            
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
