"""
Therapeutic techniques knowledge base module.

This module provides access to a database of therapeutic techniques from various
evidence-based approaches (CBT, DBT, ACT, Mindfulness, etc.), enabling the chatbot
to provide practical, actionable steps in its responses.
"""

import json
import os
from typing import Dict, List, Optional, Union

class TherapeuticKnowledgeBase:
    """A class to manage and retrieve therapeutic techniques."""
    
    def __init__(self, techniques_file: str = None):
        """Initialize the therapeutic knowledge base.
        
        Args:
            techniques_file: Path to the JSON file containing therapeutic techniques.
                If None, uses the default techniques.json in the same directory.
        """
        if techniques_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            techniques_file = os.path.join(current_dir, "techniques.json")
        
        self.techniques_file = techniques_file
        self.techniques = self._load_techniques()
        self.technique_map = {t["id"]: t for t in self.techniques}
        
    def _load_techniques(self) -> List[Dict]:
        """Load therapeutic techniques from the JSON file."""
        try:
            with open(self.techniques_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data.get("techniques", [])
        except Exception as e:
            print(f"Error loading therapeutic techniques: {e}")
            return []
    
    def get_technique_by_id(self, technique_id: str) -> Optional[Dict]:
        """Retrieve a specific technique by its ID."""
        return self.technique_map.get(technique_id)
    
    def get_techniques_by_category(self, category: str) -> List[Dict]:
        """Retrieve all techniques from a specific category."""
        return [t for t in self.techniques if t.get("category", "").lower() == category.lower()]
    
    def get_techniques_by_emotion(self, emotion: str) -> List[Dict]:
        """Retrieve techniques relevant to a specific emotion."""
        return [t for t in self.techniques if emotion.lower() in [e.lower() for e in t.get("emotions", [])]]
    
    def get_techniques_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Retrieve techniques by difficulty level."""
        return [t for t in self.techniques if t.get("difficulty", "").lower() == difficulty.lower()]
    
    def get_all_techniques(self) -> List[Dict]:
        """Return all therapeutic techniques."""
        return self.techniques
    
    def format_technique_steps(self, technique: Dict) -> str:
        """Format a technique's steps into a readable string format."""
        if not technique:
            return ""
            
        result = f"## {technique['name']} ({technique['category']})\n\n"
        result += f"{technique['description']}\n\n"
        result += "**Steps:**\n"
        
        for i, step in enumerate(technique.get("steps", []), 1):
            result += f"{i}. {step}\n"
            
        if technique.get("examples"):
            result += "\n**Example:**\n"
            for example in technique.get("examples", []):
                result += f"- {example}\n"
                
        return result