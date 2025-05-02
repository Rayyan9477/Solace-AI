"""
Personality assessment module for the mental health chatbot.
Provides implementations of various personality assessment models.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from .big_five import BigFiveAssessment
from .mbti import MBTIAssessment
import logging
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'PersonalityManager',
    'BigFiveAssessment',
    'MBTIAssessment'
]

class PersonalityManager:
    """Manages personality assessments and results"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the personality manager
        
        Args:
            data_dir: Directory to store personality data (if None, uses default)
        """
        self.big_five = BigFiveAssessment()
        self.mbti = MBTIAssessment()
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data' / 'personality'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_cache = {}
        
    def get_assessment(self, assessment_type: str):
        """Get an assessment instance by type"""
        if assessment_type.lower() == 'big_five':
            return self.big_five
        elif assessment_type.lower() == 'mbti':
            return self.mbti
        else:
            raise ValueError(f"Unknown assessment type: {assessment_type}")
    
    def get_questions(self, assessment_type: str, num_questions: int = 20) -> List[Dict[str, Any]]:
        """
        Get questions for an assessment
        
        Args:
            assessment_type: Type of assessment ('big_five' or 'mbti')
            num_questions: Number of questions to retrieve
            
        Returns:
            List of question dictionaries
        """
        assessment = self.get_assessment(assessment_type)
        return assessment.get_questions(num_questions)
    
    def compute_results(self, assessment_type: str, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute assessment results
        
        Args:
            assessment_type: Type of assessment ('big_five' or 'mbti')
            responses: User responses to assessment questions
            
        Returns:
            Assessment results
        """
        assessment = self.get_assessment(assessment_type)
        results = assessment.compute_results(responses)
        
        # Cache results
        result_entry = {
            'type': assessment_type,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        self.results_cache[assessment_type] = result_entry
        
        # Save results to file
        self._save_results(assessment_type, result_entry)
        
        return results
    
    def get_latest_results(self, assessment_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the latest assessment results
        
        Args:
            assessment_type: Type of assessment ('big_five' or 'mbti')
                            If None, returns all available results
                            
        Returns:
            Assessment results or empty dict if none found
        """
        if assessment_type:
            # Return specific assessment results
            if assessment_type in self.results_cache:
                return self.results_cache[assessment_type]
            
            # Try to load from file
            loaded = self._load_results(assessment_type)
            if loaded:
                self.results_cache[assessment_type] = loaded
                return loaded
                
            return {}
        else:
            # Return all available results
            all_results = {}
            
            # Get cached results
            for assessment_type, results in self.results_cache.items():
                all_results[assessment_type] = results
            
            # Load any uncached results from files
            for assessment_type in ['big_five', 'mbti']:
                if assessment_type not in all_results:
                    loaded = self._load_results(assessment_type)
                    if loaded:
                        all_results[assessment_type] = loaded
                        self.results_cache[assessment_type] = loaded
            
            return all_results
    
    def _save_results(self, assessment_type: str, results: Dict[str, Any]) -> bool:
        """Save results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{assessment_type}_{timestamp}.json"
            filepath = self.data_dir / filename
            
            # Save with timestamp in filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Also save to latest file
            latest_path = self.data_dir / f"{assessment_type}_latest.json"
            with open(latest_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save {assessment_type} results: {str(e)}")
            return False
    
    def _load_results(self, assessment_type: str) -> Dict[str, Any]:
        """Load latest results from file"""
        try:
            latest_path = self.data_dir / f"{assessment_type}_latest.json"
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load {assessment_type} results: {str(e)}")
            return {}
