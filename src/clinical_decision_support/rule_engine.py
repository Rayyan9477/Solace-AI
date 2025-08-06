"""
Clinical Rule Engine for Evidence-Based Decision Support
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re
import json

logger = logging.getLogger(__name__)

class RuleOperator(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    REGEX_MATCH = "regex_match"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

class RulePriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ActionType(Enum):
    RECOMMEND = "recommend"
    ALERT = "alert"
    REQUIRE = "require"
    CONTRAINDICATE = "contraindicate"
    MODIFY = "modify"

@dataclass
class RuleCondition:
    """Individual condition within a clinical rule"""
    field: str
    operator: RuleOperator
    value: Any
    description: str = ""
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate this condition against provided data"""
        try:
            field_value = self._get_nested_value(data, self.field)
            
            if self.operator == RuleOperator.EQUALS:
                return field_value == self.value
            elif self.operator == RuleOperator.NOT_EQUALS:
                return field_value != self.value
            elif self.operator == RuleOperator.GREATER_THAN:
                return float(field_value) > float(self.value)
            elif self.operator == RuleOperator.LESS_THAN:
                return float(field_value) < float(self.value)
            elif self.operator == RuleOperator.GREATER_EQUAL:
                return float(field_value) >= float(self.value)
            elif self.operator == RuleOperator.LESS_EQUAL:
                return float(field_value) <= float(self.value)
            elif self.operator == RuleOperator.CONTAINS:
                return str(self.value).lower() in str(field_value).lower()
            elif self.operator == RuleOperator.NOT_CONTAINS:
                return str(self.value).lower() not in str(field_value).lower()
            elif self.operator == RuleOperator.IN:
                return field_value in self.value
            elif self.operator == RuleOperator.NOT_IN:
                return field_value not in self.value
            elif self.operator == RuleOperator.REGEX_MATCH:
                return bool(re.match(str(self.value), str(field_value)))
            elif self.operator == RuleOperator.IS_NULL:
                return field_value is None
            elif self.operator == RuleOperator.IS_NOT_NULL:
                return field_value is not None
            else:
                return False
                
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error evaluating condition {self.field} {self.operator.value} {self.value}: {e}")
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field_path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Field {field_path} not found in data")
        
        return value

@dataclass
class RuleAction:
    """Action to be taken when rule conditions are met"""
    action_type: ActionType
    target: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"

@dataclass
class ClinicalRule:
    """Clinical decision support rule"""
    rule_id: str
    name: str
    description: str
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    priority: RulePriority = RulePriority.MEDIUM
    condition_logic: str = "AND"  # "AND" or "OR"
    enabled: bool = True
    version: str = "1.0"
    author: str = ""
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    evidence_level: str = "A"  # A, B, C, D evidence levels
    source_guideline: str = ""
    tags: List[str] = field(default_factory=list)
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate if this rule should fire based on provided data"""
        if not self.enabled or not self.conditions:
            return False
        
        condition_results = [condition.evaluate(data) for condition in self.conditions]
        
        if self.condition_logic == "AND":
            return all(condition_results)
        elif self.condition_logic == "OR":
            return any(condition_results)
        else:
            logger.warning(f"Unknown condition logic: {self.condition_logic}")
            return False

class ClinicalRuleEngine:
    """Engine for evaluating clinical decision support rules"""
    
    def __init__(self):
        self.rules: Dict[str, ClinicalRule] = {}
        self.rule_categories: Dict[str, List[str]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_rule(self, rule: ClinicalRule, category: str = "general"):
        """Add a clinical rule to the engine"""
        self.rules[rule.rule_id] = rule
        
        if category not in self.rule_categories:
            self.rule_categories[category] = []
        
        if rule.rule_id not in self.rule_categories[category]:
            self.rule_categories[category].append(rule.rule_id)
        
        logger.info(f"Added rule {rule.rule_id} to category {category}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            
            # Remove from categories
            for category, rule_ids in self.rule_categories.items():
                if rule_id in rule_ids:
                    rule_ids.remove(rule_id)
            
            logger.info(f"Removed rule {rule_id}")
            return True
        
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            return True
        return False
    
    def evaluate_rules(self, data: Dict[str, Any], 
                      categories: Optional[List[str]] = None,
                      rule_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate rules against provided data
        
        Args:
            data: Clinical data to evaluate
            categories: Optional list of categories to evaluate
            rule_ids: Optional list of specific rule IDs to evaluate
            
        Returns:
            List of triggered rules with their actions
        """
        triggered_rules = []
        
        # Determine which rules to evaluate
        rules_to_evaluate = []
        
        if rule_ids:
            rules_to_evaluate = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        elif categories:
            for category in categories:
                if category in self.rule_categories:
                    rules_to_evaluate.extend([
                        self.rules[rid] for rid in self.rule_categories[category] 
                        if rid in self.rules
                    ])
        else:
            rules_to_evaluate = list(self.rules.values())
        
        # Evaluate rules by priority (highest first)
        rules_to_evaluate.sort(key=lambda r: r.priority.value, reverse=True)
        
        for rule in rules_to_evaluate:
            try:
                if rule.evaluate(data):
                    rule_result = {
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'description': rule.description,
                        'priority': rule.priority.name,
                        'evidence_level': rule.evidence_level,
                        'actions': [
                            {
                                'type': action.action_type.value,
                                'target': action.target,
                                'message': action.message,
                                'severity': action.severity,
                                'data': action.data
                            }
                            for action in rule.actions
                        ],
                        'triggered_at': datetime.utcnow().isoformat(),
                        'source_guideline': rule.source_guideline,
                        'version': rule.version
                    }
                    
                    triggered_rules.append(rule_result)
                    
                    # Log execution
                    self.execution_history.append({
                        'rule_id': rule.rule_id,
                        'triggered': True,
                        'timestamp': datetime.utcnow(),
                        'data_hash': hash(str(data))
                    })
                    
                    logger.info(f"Rule triggered: {rule.rule_id}")
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
                
                # Log failed execution
                self.execution_history.append({
                    'rule_id': rule.rule_id,
                    'triggered': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow(),
                    'data_hash': hash(str(data))
                })
        
        return triggered_rules
    
    def load_rules_from_json(self, json_data: Union[str, Dict[str, Any]], category: str = "imported"):
        """Load rules from JSON configuration"""
        try:
            if isinstance(json_data, str):
                config = json.loads(json_data)
            else:
                config = json_data
            
            for rule_data in config.get('rules', []):
                # Parse conditions
                conditions = []
                for cond_data in rule_data.get('conditions', []):
                    condition = RuleCondition(
                        field=cond_data['field'],
                        operator=RuleOperator(cond_data['operator']),
                        value=cond_data['value'],
                        description=cond_data.get('description', '')
                    )
                    conditions.append(condition)
                
                # Parse actions
                actions = []
                for action_data in rule_data.get('actions', []):
                    action = RuleAction(
                        action_type=ActionType(action_data['action_type']),
                        target=action_data['target'],
                        message=action_data['message'],
                        data=action_data.get('data', {}),
                        severity=action_data.get('severity', 'medium')
                    )
                    actions.append(action)
                
                # Create rule
                rule = ClinicalRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    conditions=conditions,
                    actions=actions,
                    priority=RulePriority(rule_data.get('priority', 2)),
                    condition_logic=rule_data.get('condition_logic', 'AND'),
                    enabled=rule_data.get('enabled', True),
                    version=rule_data.get('version', '1.0'),
                    author=rule_data.get('author', ''),
                    evidence_level=rule_data.get('evidence_level', 'B'),
                    source_guideline=rule_data.get('source_guideline', ''),
                    tags=rule_data.get('tags', [])
                )
                
                self.add_rule(rule, category)
            
            logger.info(f"Loaded {len(config.get('rules', []))} rules from JSON")
            
        except Exception as e:
            logger.error(f"Error loading rules from JSON: {e}")
            raise
    
    def export_rules_to_json(self, category: Optional[str] = None) -> str:
        """Export rules to JSON format"""
        try:
            rules_to_export = []
            
            if category:
                rule_ids = self.rule_categories.get(category, [])
                rules_to_export = [self.rules[rid] for rid in rule_ids if rid in self.rules]
            else:
                rules_to_export = list(self.rules.values())
            
            export_data = {
                'rules': [],
                'exported_at': datetime.utcnow().isoformat(),
                'total_rules': len(rules_to_export)
            }
            
            for rule in rules_to_export:
                rule_data = {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'description': rule.description,
                    'priority': rule.priority.value,
                    'condition_logic': rule.condition_logic,
                    'enabled': rule.enabled,
                    'version': rule.version,
                    'author': rule.author,
                    'evidence_level': rule.evidence_level,
                    'source_guideline': rule.source_guideline,
                    'tags': rule.tags,
                    'conditions': [
                        {
                            'field': cond.field,
                            'operator': cond.operator.value,
                            'value': cond.value,
                            'description': cond.description
                        }
                        for cond in rule.conditions
                    ],
                    'actions': [
                        {
                            'action_type': action.action_type.value,
                            'target': action.target,
                            'message': action.message,
                            'data': action.data,
                            'severity': action.severity
                        }
                        for action in rule.actions
                    ]
                }
                
                export_data['rules'].append(rule_data)
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting rules to JSON: {e}")
            raise
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rules and execution"""
        total_rules = len(self.rules)
        enabled_rules = sum(1 for rule in self.rules.values() if rule.enabled)
        
        # Count rules by priority
        priority_counts = {}
        for priority in RulePriority:
            priority_counts[priority.name] = sum(
                1 for rule in self.rules.values() if rule.priority == priority
            )
        
        # Count rules by category
        category_counts = {
            category: len(rule_ids) 
            for category, rule_ids in self.rule_categories.items()
        }
        
        # Execution statistics
        recent_executions = [
            exec_data for exec_data in self.execution_history[-100:]
        ]
        
        successful_executions = sum(1 for exec_data in recent_executions if exec_data.get('triggered'))
        failed_executions = sum(1 for exec_data in recent_executions if 'error' in exec_data)
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'disabled_rules': total_rules - enabled_rules,
            'priority_distribution': priority_counts,
            'category_distribution': category_counts,
            'recent_executions': len(recent_executions),
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': successful_executions / len(recent_executions) if recent_executions else 0
        }
    
    def validate_rule(self, rule: ClinicalRule) -> List[str]:
        """Validate a rule for common issues"""
        issues = []
        
        # Check for empty conditions
        if not rule.conditions:
            issues.append("Rule has no conditions")
        
        # Check for empty actions
        if not rule.actions:
            issues.append("Rule has no actions")
        
        # Check for invalid condition logic
        if rule.condition_logic not in ["AND", "OR"]:
            issues.append(f"Invalid condition logic: {rule.condition_logic}")
        
        # Check for duplicate rule ID
        if rule.rule_id in self.rules:
            issues.append(f"Rule ID {rule.rule_id} already exists")
        
        # Validate conditions
        for i, condition in enumerate(rule.conditions):
            if not condition.field:
                issues.append(f"Condition {i+1} has empty field")
            
            if condition.operator in [RuleOperator.IN, RuleOperator.NOT_IN]:
                if not isinstance(condition.value, (list, tuple)):
                    issues.append(f"Condition {i+1} uses IN/NOT_IN but value is not a list")
        
        # Validate actions
        for i, action in enumerate(rule.actions):
            if not action.target:
                issues.append(f"Action {i+1} has empty target")
            
            if not action.message:
                issues.append(f"Action {i+1} has empty message")
        
        return issues