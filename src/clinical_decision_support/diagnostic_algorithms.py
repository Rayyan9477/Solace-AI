"""
Diagnostic Algorithms and Clinical Decision Pathways
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DiagnosticConfidence(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

class PathwayStepType(Enum):
    ASSESSMENT = "assessment"
    TEST = "test"
    DECISION = "decision"
    DIAGNOSIS = "diagnosis"
    REFERRAL = "referral"
    TREATMENT = "treatment"

class SeverityLevel(Enum):
    MINIMAL = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
    CRITICAL = 5

@dataclass
class DiagnosticCriterion:
    """Individual diagnostic criterion"""
    criterion_id: str
    name: str
    description: str
    weight: float = 1.0
    required: bool = False
    evidence_level: str = "B"
    
    def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate this criterion against clinical data
        
        Returns:
            (met, confidence, explanation)
        """
        # This would be implemented with specific logic for each criterion type
        # For now, providing a template implementation
        return False, 0.0, "Not implemented"

@dataclass
class DiagnosticCondition:
    """Diagnostic condition with criteria and scoring"""
    condition_id: str
    name: str
    description: str
    icd_10_codes: List[str]
    criteria: List[DiagnosticCriterion]
    minimum_criteria: int = 1
    scoring_method: str = "weighted_sum"  # "weighted_sum", "threshold", "bayesian"
    diagnostic_threshold: float = 0.5
    severity_mapping: Dict[str, SeverityLevel] = field(default_factory=dict)
    
    def evaluate_diagnosis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if this condition is present based on clinical data"""
        met_criteria = []
        total_score = 0.0
        explanations = []
        
        for criterion in self.criteria:
            met, confidence, explanation = criterion.evaluate(data)
            
            if met:
                met_criteria.append(criterion)
                total_score += criterion.weight * confidence
                explanations.append(f"{criterion.name}: {explanation}")
        
        # Determine if diagnosis is met
        diagnosis_met = False
        overall_confidence = 0.0
        
        if self.scoring_method == "weighted_sum":
            max_possible_score = sum(c.weight for c in self.criteria)
            overall_confidence = total_score / max_possible_score if max_possible_score > 0 else 0
            diagnosis_met = overall_confidence >= self.diagnostic_threshold
            
        elif self.scoring_method == "threshold":
            diagnosis_met = len(met_criteria) >= self.minimum_criteria
            overall_confidence = len(met_criteria) / len(self.criteria) if self.criteria else 0
            
        # Determine severity if diagnosis is met
        severity = SeverityLevel.MINIMAL
        if diagnosis_met and overall_confidence > 0.7:
            severity = SeverityLevel.MODERATE
        elif diagnosis_met and overall_confidence > 0.85:
            severity = SeverityLevel.SEVERE
        
        return {
            'condition_id': self.condition_id,
            'condition_name': self.name,
            'diagnosis_met': diagnosis_met,
            'confidence': overall_confidence,
            'severity': severity.name,
            'met_criteria': [c.criterion_id for c in met_criteria],
            'total_criteria': len(self.criteria),
            'score': total_score,
            'explanations': explanations,
            'icd_10_codes': self.icd_10_codes if diagnosis_met else []
        }

@dataclass
class PathwayStep:
    """Individual step in a diagnostic pathway"""
    step_id: str
    name: str
    step_type: PathwayStepType
    description: str
    conditions: List[str] = field(default_factory=list)  # Conditions to check
    next_steps: Dict[str, str] = field(default_factory=dict)  # Outcome -> next_step_id
    required_data: List[str] = field(default_factory=list)
    estimated_time: int = 5  # minutes
    cost_estimate: float = 0.0
    
    def can_execute(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if this step can be executed with available data"""
        missing_data = []
        
        for required_field in self.required_data:
            if required_field not in data:
                missing_data.append(required_field)
        
        return len(missing_data) == 0, missing_data
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this pathway step"""
        can_execute, missing = self.can_execute(data)
        
        if not can_execute:
            return {
                'step_id': self.step_id,
                'success': False,
                'error': f"Missing required data: {missing}",
                'next_step': None
            }
        
        # Step-specific execution logic would go here
        # For now, return a template result
        return {
            'step_id': self.step_id,
            'success': True,
            'result': f"Executed {self.name}",
            'next_step': self.next_steps.get('default'),
            'execution_time': self.estimated_time
        }

@dataclass
class DiagnosticPathway:
    """Complete diagnostic pathway"""
    pathway_id: str
    name: str
    description: str
    specialty: str
    conditions: List[DiagnosticCondition]
    steps: List[PathwayStep]
    entry_point: str  # Initial step ID
    evidence_level: str = "B"
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def execute_pathway(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete diagnostic pathway"""
        execution_log = []
        current_step_id = self.entry_point
        total_time = 0
        total_cost = 0.0
        
        step_map = {step.step_id: step for step in self.steps}
        
        while current_step_id:
            if current_step_id not in step_map:
                execution_log.append({
                    'error': f"Step {current_step_id} not found in pathway"
                })
                break
            
            current_step = step_map[current_step_id]
            step_result = current_step.execute(data)
            
            execution_log.append({
                'step_id': current_step_id,
                'step_name': current_step.name,
                'result': step_result,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            total_time += step_result.get('execution_time', current_step.estimated_time)
            total_cost += current_step.cost_estimate
            
            if not step_result.get('success'):
                break
            
            current_step_id = step_result.get('next_step')
        
        # Evaluate final diagnostic results
        diagnostic_results = []
        for condition in self.conditions:
            result = condition.evaluate_diagnosis(data)
            if result['diagnosis_met']:
                diagnostic_results.append(result)
        
        return {
            'pathway_id': self.pathway_id,
            'pathway_name': self.name,
            'execution_log': execution_log,
            'diagnostic_results': diagnostic_results,
            'total_execution_time': total_time,
            'total_estimated_cost': total_cost,
            'completed_at': datetime.utcnow().isoformat()
        }

class DiagnosticAlgorithm:
    """Main diagnostic algorithm engine"""
    
    def __init__(self):
        self.pathways: Dict[str, DiagnosticPathway] = {}
        self.conditions: Dict[str, DiagnosticCondition] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_pathway(self, pathway: DiagnosticPathway):
        """Add a diagnostic pathway"""
        self.pathways[pathway.pathway_id] = pathway
        
        # Also add individual conditions
        for condition in pathway.conditions:
            self.conditions[condition.condition_id] = condition
        
        logger.info(f"Added diagnostic pathway: {pathway.pathway_id}")
    
    def add_condition(self, condition: DiagnosticCondition):
        """Add a diagnostic condition"""
        self.conditions[condition.condition_id] = condition
        logger.info(f"Added diagnostic condition: {condition.condition_id}")
    
    def run_differential_diagnosis(self, data: Dict[str, Any], 
                                 specialty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run differential diagnosis across all conditions"""
        results = []
        
        conditions_to_check = list(self.conditions.values())
        if specialty:
            # Filter by specialty if specified
            relevant_pathways = [p for p in self.pathways.values() if p.specialty == specialty]
            condition_ids = set()
            for pathway in relevant_pathways:
                condition_ids.update(c.condition_id for c in pathway.conditions)
            conditions_to_check = [c for c in conditions_to_check if c.condition_id in condition_ids]
        
        for condition in conditions_to_check:
            try:
                result = condition.evaluate_diagnosis(data)
                if result['confidence'] > 0.1:  # Only include plausible diagnoses
                    results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating condition {condition.condition_id}: {e}")
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Log execution
        self.execution_history.append({
            'type': 'differential_diagnosis',
            'specialty': specialty,
            'results_count': len(results),
            'timestamp': datetime.utcnow(),
            'top_diagnosis': results[0]['condition_name'] if results else None
        })
        
        return results
    
    def execute_pathway(self, pathway_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific diagnostic pathway"""
        if pathway_id not in self.pathways:
            raise ValueError(f"Pathway {pathway_id} not found")
        
        pathway = self.pathways[pathway_id]
        result = pathway.execute_pathway(data)
        
        # Log execution
        self.execution_history.append({
            'type': 'pathway_execution',
            'pathway_id': pathway_id,
            'pathway_name': pathway.name,
            'diagnostic_results': len(result['diagnostic_results']),
            'execution_time': result['total_execution_time'],
            'timestamp': datetime.utcnow()
        })
        
        return result
    
    def get_recommended_pathway(self, data: Dict[str, Any]) -> Optional[str]:
        """Get recommended pathway based on presenting symptoms/data"""
        # This would implement logic to select the best pathway
        # Based on presenting symptoms, specialty, etc.
        
        # Simple implementation: return first pathway that has conditions matching data
        for pathway in self.pathways.values():
            for condition in pathway.conditions:
                result = condition.evaluate_diagnosis(data)
                if result['confidence'] > 0.3:
                    return pathway.pathway_id
        
        return None
    
    def validate_pathway(self, pathway: DiagnosticPathway) -> List[str]:
        """Validate a diagnostic pathway for issues"""
        issues = []
        
        # Check for entry point
        if not pathway.entry_point:
            issues.append("Pathway has no entry point")
        
        step_ids = {step.step_id for step in pathway.steps}
        
        # Check if entry point exists
        if pathway.entry_point not in step_ids:
            issues.append(f"Entry point {pathway.entry_point} not found in steps")
        
        # Check step references
        for step in pathway.steps:
            for outcome, next_step_id in step.next_steps.items():
                if next_step_id and next_step_id not in step_ids:
                    issues.append(f"Step {step.step_id} references non-existent step {next_step_id}")
        
        # Check for unreachable steps
        reachable_steps = set()
        to_visit = [pathway.entry_point] if pathway.entry_point in step_ids else []
        
        while to_visit:
            current = to_visit.pop()
            if current in reachable_steps:
                continue
            reachable_steps.add(current)
            
            step = next((s for s in pathway.steps if s.step_id == current), None)
            if step:
                for next_step_id in step.next_steps.values():
                    if next_step_id and next_step_id not in reachable_steps:
                        to_visit.append(next_step_id)
        
        unreachable = step_ids - reachable_steps
        if unreachable:
            issues.append(f"Unreachable steps: {list(unreachable)}")
        
        # Check conditions
        if not pathway.conditions:
            issues.append("Pathway has no diagnostic conditions")
        
        return issues
    
    def export_pathway(self, pathway_id: str) -> str:
        """Export pathway to JSON"""
        if pathway_id not in self.pathways:
            raise ValueError(f"Pathway {pathway_id} not found")
        
        pathway = self.pathways[pathway_id]
        
        export_data = {
            'pathway_id': pathway.pathway_id,
            'name': pathway.name,
            'description': pathway.description,
            'specialty': pathway.specialty,
            'entry_point': pathway.entry_point,
            'evidence_level': pathway.evidence_level,
            'last_updated': pathway.last_updated.isoformat(),
            'conditions': [
                {
                    'condition_id': cond.condition_id,
                    'name': cond.name,
                    'description': cond.description,
                    'icd_10_codes': cond.icd_10_codes,
                    'minimum_criteria': cond.minimum_criteria,
                    'scoring_method': cond.scoring_method,
                    'diagnostic_threshold': cond.diagnostic_threshold,
                    'criteria': [
                        {
                            'criterion_id': crit.criterion_id,
                            'name': crit.name,
                            'description': crit.description,
                            'weight': crit.weight,
                            'required': crit.required,
                            'evidence_level': crit.evidence_level
                        }
                        for crit in cond.criteria
                    ]
                }
                for cond in pathway.conditions
            ],
            'steps': [
                {
                    'step_id': step.step_id,
                    'name': step.name,
                    'step_type': step.step_type.value,
                    'description': step.description,
                    'conditions': step.conditions,
                    'next_steps': step.next_steps,
                    'required_data': step.required_data,
                    'estimated_time': step.estimated_time,
                    'cost_estimate': step.cost_estimate
                }
                for step in pathway.steps
            ]
        }
        
        return json.dumps(export_data, indent=2)
    
    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """Get statistics about the diagnostic algorithm"""
        total_pathways = len(self.pathways)
        total_conditions = len(self.conditions)
        
        # Count by specialty
        specialties = {}
        for pathway in self.pathways.values():
            if pathway.specialty not in specialties:
                specialties[pathway.specialty] = 0
            specialties[pathway.specialty] += 1
        
        # Execution statistics
        recent_executions = self.execution_history[-100:]
        pathway_executions = [e for e in recent_executions if e['type'] == 'pathway_execution']
        differential_executions = [e for e in recent_executions if e['type'] == 'differential_diagnosis']
        
        return {
            'total_pathways': total_pathways,
            'total_conditions': total_conditions,
            'specialties': specialties,
            'recent_executions': len(recent_executions),
            'pathway_executions': len(pathway_executions),
            'differential_executions': len(differential_executions),
            'average_execution_time': sum(e.get('execution_time', 0) for e in pathway_executions) / len(pathway_executions) if pathway_executions else 0
        }