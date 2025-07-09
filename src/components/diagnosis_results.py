"""
Component for rendering diagnosis results in a user-friendly format.
"""

from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class DiagnosisResults:
    """
    Manager for diagnosis results, handling storage, formatting, and retrieval.
    Provides a consistent interface for working with diagnosis data.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the diagnosis results manager
        
        Args:
            storage_path: Optional path to store diagnosis results
        """
        self.storage_path = storage_path or os.path.join(
            Path(__file__).parents[2], "data", "diagnostic_data"
        )
        os.makedirs(self.storage_path, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_results(self, user_id: str, results: Dict[str, Any]) -> bool:
        """
        Save diagnosis results for a user
        
        Args:
            user_id: Unique identifier for the user
            results: Dictionary of diagnosis results
            
        Returns:
            bool: True if save was successful
        """
        try:
            file_path = os.path.join(self.storage_path, f"{user_id}_diagnosis.json")
            with open(file_path, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved diagnosis results for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving diagnosis results: {str(e)}")
            return False
    
    def load_results(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load diagnosis results for a user
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Optional[Dict[str, Any]]: Diagnosis results or None if not found
        """
        try:
            file_path = os.path.join(self.storage_path, f"{user_id}_diagnosis.json")
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, "r") as f:
                results = json.load(f)
            return results
        except Exception as e:
            self.logger.error(f"Error loading diagnosis results: {str(e)}")
            return None
    
    def format_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format diagnosis results for display
        
        Args:
            results: Raw diagnosis results
            
        Returns:
            Dict[str, Any]: Formatted results
        """
        formatted = {}
        
        # Format mental health section
        if "mental_health" in results:
            formatted["mental_health"] = self._format_mental_health(results["mental_health"])
        
        # Format personality section
        if "personality" in results:
            formatted["personality"] = self._format_personality(results["personality"])
        
        # Format recommendations
        if "recommendations" in results:
            formatted["recommendations"] = results["recommendations"]
        
        return formatted
    
    def _format_mental_health(self, mental_health: Dict[str, Any]) -> Dict[str, Any]:
        """Format mental health section"""
        formatted = {
            "overall_status": mental_health.get("overall_status", "mild"),
            "areas_of_concern": [],
            "strengths": []
        }
        
        # Format areas of concern
        for area in mental_health.get("areas_of_concern", []):
            formatted["areas_of_concern"].append({
                "name": area.get("name", ""),
                "severity": area.get("severity", "mild"),
                "description": area.get("description", ""),
                "score": area.get("score", 0)
            })
        
        # Format strengths
        for strength in mental_health.get("strengths", []):
            formatted["strengths"].append({
                "name": strength.get("name", ""),
                "description": strength.get("description", ""),
                "score": strength.get("score", 0)
            })
        
        return formatted
    
    def _format_personality(self, personality: Dict[str, Any]) -> Dict[str, Any]:
        """Format personality section"""
        formatted = {
            "traits": [],
            "summary": personality.get("summary", "")
        }
        
        # Format traits
        for trait in personality.get("traits", []):
            formatted["traits"].append({
                "name": trait.get("name", ""),
                "score": trait.get("score", 0),
                "description": trait.get("description", "")
            })
        
        return formatted
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a text report from diagnosis results
        
        Args:
            results: Diagnosis results
            
        Returns:
            str: Formatted text report
        """
        report = []
        
        # Add title
        report.append("===== Diagnostic Results =====")
        report.append("")
        
        # Mental health section
        if "mental_health" in results:
            mh = results["mental_health"]
            report.append(f"Mental Health Status: {mh.get('overall_status', 'Unknown').capitalize()}")
            report.append("")
            
            # Areas of concern
            if "areas_of_concern" in mh and mh["areas_of_concern"]:
                report.append("Areas of Concern:")
                for area in mh["areas_of_concern"]:
                    report.append(f"- {area.get('name', 'Unknown')}: {area.get('severity', 'Unknown')} " +
                                f"({area.get('score', 0)}/10)")
                report.append("")
            
            # Strengths
            if "strengths" in mh and mh["strengths"]:
                report.append("Strengths:")
                for strength in mh["strengths"]:
                    report.append(f"- {strength.get('name', 'Unknown')} ({strength.get('score', 0)}/10)")
                report.append("")
        
        # Personality section
        if "personality" in results:
            pers = results["personality"]
            report.append("Personality Profile:")
            report.append("")
            
            if "summary" in pers:
                report.append(f"{pers['summary']}")
                report.append("")
            
            # Traits
            if "traits" in pers and pers["traits"]:
                report.append("Key Traits:")
                for trait in pers["traits"]:
                    report.append(f"- {trait.get('name', 'Unknown')}: {trait.get('score', 0)}/10")
                report.append("")
        
        # Recommendations
        if "recommendations" in results and results["recommendations"]:
            report.append("Recommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        report.append("===============================")
        
        return "\n".join(report)
    
    def generate_json_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured JSON report from diagnosis results
        for use with API/mobile interfaces
        
        Args:
            results: Diagnosis results
            
        Returns:
            Dict[str, Any]: Structured results for API consumption
        """
        # Format the results for API consumption
        # This could be customized based on the mobile app requirements
        api_format = {
            "diagnosis": {
                "timestamp": results.get("timestamp", ""),
                "mental_health": {},
                "personality": {},
                "recommendations": []
            }
        }
        
        # Format mental health data
        if "mental_health" in results:
            mh = results["mental_health"]
            api_format["diagnosis"]["mental_health"] = {
                "status": mh.get("overall_status", "unknown"),
                "concerns": [
                    {
                        "name": area.get("name", ""),
                        "severity": area.get("severity", ""),
                        "score": area.get("score", 0)
                    } for area in mh.get("areas_of_concern", [])
                ],
                "strengths": [
                    {
                        "name": strength.get("name", ""),
                        "score": strength.get("score", 0)
                    } for strength in mh.get("strengths", [])
                ]
            }
        
        # Format personality data
        if "personality" in results:
            pers = results["personality"]
            api_format["diagnosis"]["personality"] = {
                "summary": pers.get("summary", ""),
                "traits": [
                    {
                        "name": trait.get("name", ""),
                        "score": trait.get("score", 0)
                    } for trait in pers.get("traits", [])
                ]
            }
        
        # Add recommendations
        if "recommendations" in results:
            api_format["diagnosis"]["recommendations"] = results["recommendations"]
        
        return api_format
