"""
Model Management and Serialization for Enterprise Multi-Modal Diagnostic Pipeline

This module provides comprehensive model management capabilities including:
1. Model checkpointing and versioning
2. Serialization and deserialization of trained models
3. Model deployment and rollback mechanisms
4. Performance tracking and model comparison
5. Automated model updates and fine-tuning
6. Model validation and testing frameworks

Author: Solace-AI Development Team
Version: 1.0.0
"""

import os
import json
import hashlib
import shutil
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio
import warnings

# Security: Removed pickle import - CWE-502 unsafe deserialization
# Using JSON with custom serialization for feature extractors

# ML/AI imports
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# Import pipeline components
from .enterprise_multimodal_pipeline import (
    EnterpriseMultiModalDiagnosticPipeline,
    MultiModalAttention,
    BayesianDiagnosticLayer,
    TemporalSequenceModel
)

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata"""
    model_id: str
    version: str
    timestamp: datetime
    model_type: str
    performance_metrics: Dict[str, float]
    model_config: Dict[str, Any]
    file_path: str
    file_size: int
    checksum: str
    validation_results: Dict[str, Any]
    training_info: Optional[Dict[str, Any]] = None

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    calibration_error: float
    uncertainty_quality: float
    clinical_validity: float
    processing_speed: float  # samples per second

class ModelManager:
    """
    Comprehensive model management system for enterprise diagnostic pipeline
    """
    
    def __init__(self, 
                 models_directory: str = "models",
                 max_checkpoints: int = 10,
                 validation_split: float = 0.2):
        """
        Initialize model manager
        
        Args:
            models_directory: Directory to store model checkpoints
            max_checkpoints: Maximum number of checkpoints to keep per model
            validation_split: Fraction of data to use for validation
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.validation_split = validation_split
        
        # Model registry
        self.model_registry = {}
        self.checkpoint_history = {}
        self.performance_tracking = {}
        
        # Load existing registry
        self._load_model_registry()
        
        logger.info(f"Model manager initialized with directory: {self.models_directory}")

    def _load_model_registry(self):
        """Load existing model registry from disk"""
        registry_path = self.models_directory / "model_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                for model_id, checkpoints in registry_data.items():
                    self.model_registry[model_id] = []
                    for checkpoint_data in checkpoints:
                        checkpoint_data['timestamp'] = datetime.fromisoformat(checkpoint_data['timestamp'])
                        checkpoint = ModelCheckpoint(**checkpoint_data)
                        self.model_registry[model_id].append(checkpoint)
                
                logger.info(f"Loaded {len(self.model_registry)} models from registry")
                
            except Exception as e:
                logger.error(f"Error loading model registry: {str(e)}")
                self.model_registry = {}

    def _serialize_feature_extractors(self, extractors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize feature extractors to JSON-safe format (SEC: CWE-502 fix).

        Converts numpy arrays and other non-JSON-serializable types to safe formats.

        Args:
            extractors: Dictionary of feature extractors

        Returns:
            JSON-serializable dictionary
        """
        def convert_value(v):
            if isinstance(v, np.ndarray):
                return {"__type__": "ndarray", "data": v.tolist(), "dtype": str(v.dtype)}
            elif isinstance(v, (np.integer, np.floating)):
                return {"__type__": "numpy_scalar", "value": float(v), "dtype": str(type(v).__name__)}
            elif isinstance(v, datetime):
                return {"__type__": "datetime", "value": v.isoformat()}
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            elif isinstance(v, (str, int, float, bool, type(None))):
                return v
            else:
                # For objects we can't serialize, store their string representation
                return {"__type__": "unsupported", "repr": str(v)}

        if extractors is None:
            return {}
        return {k: convert_value(v) for k, v in extractors.items()}

    def _deserialize_feature_extractors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize feature extractors from JSON format (SEC: CWE-502 fix).

        Restores numpy arrays and other types from their JSON-safe representations.

        Args:
            data: JSON-serializable dictionary from _serialize_feature_extractors

        Returns:
            Dictionary of feature extractors with proper types restored
        """
        def restore_value(v):
            if isinstance(v, dict):
                type_marker = v.get("__type__")
                if type_marker == "ndarray":
                    return np.array(v["data"], dtype=v.get("dtype", "float32"))
                elif type_marker == "numpy_scalar":
                    return np.float64(v["value"])
                elif type_marker == "datetime":
                    return datetime.fromisoformat(v["value"])
                elif type_marker == "unsupported":
                    logger.warning(f"Unsupported type found during deserialization: {v.get('repr')}")
                    return None
                else:
                    return {k: restore_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [restore_value(item) for item in v]
            else:
                return v

        if data is None:
            return {}
        return {k: restore_value(v) for k, v in data.items()}

    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_path = self.models_directory / "model_registry.json"
        
        try:
            # Convert datetime objects to strings for JSON serialization
            registry_data = {}
            for model_id, checkpoints in self.model_registry.items():
                registry_data[model_id] = []
                for checkpoint in checkpoints:
                    checkpoint_dict = asdict(checkpoint)
                    checkpoint_dict['timestamp'] = checkpoint.timestamp.isoformat()
                    registry_data[model_id].append(checkpoint_dict)
            
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
            logger.info("Model registry saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")

    def save_model_checkpoint(self, 
                            pipeline: EnterpriseMultiModalDiagnosticPipeline,
                            model_id: str,
                            version: str,
                            performance_metrics: Dict[str, float],
                            validation_results: Dict[str, Any],
                            training_info: Optional[Dict[str, Any]] = None) -> ModelCheckpoint:
        """
        Save a model checkpoint with metadata
        
        Args:
            pipeline: The pipeline to save
            model_id: Unique identifier for the model
            version: Version string (e.g., "1.0.0", "2.1.3")
            performance_metrics: Performance metrics dictionary
            validation_results: Validation results
            training_info: Optional training information
            
        Returns:
            ModelCheckpoint object with metadata
        """
        try:
            # Create checkpoint directory
            checkpoint_dir = self.models_directory / model_id / version
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model components
            model_files = {}
            
            # Save neural network models
            for component_name, model in pipeline.model_components.items():
                if isinstance(model, nn.Module):
                    model_path = checkpoint_dir / f"{component_name}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': model.__dict__ if hasattr(model, '__dict__') else {},
                        'model_class': model.__class__.__name__
                    }, model_path)
                    model_files[component_name] = str(model_path)
            
            # Save feature extractors as JSON (SEC: CWE-502 fix - no pickle)
            extractors_path = checkpoint_dir / "feature_extractors.json"
            try:
                # Serialize feature extractors to JSON-safe format
                serialized = self._serialize_feature_extractors(pipeline.feature_extractors)
                with open(extractors_path, 'w', encoding='utf-8') as f:
                    json.dump(serialized, f, indent=2)
                model_files['feature_extractors'] = str(extractors_path)
            except Exception as e:
                logger.warning(f"Could not save feature extractors: {str(e)}")
            
            # Save pipeline configuration
            config_path = checkpoint_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(pipeline.config, f, indent=2)
            model_files['config'] = str(config_path)
            
            # Save clinical knowledge
            knowledge_path = checkpoint_dir / "clinical_knowledge.json"
            with open(knowledge_path, 'w') as f:
                # Convert any non-serializable objects to strings
                serializable_knowledge = self._make_serializable(pipeline.clinical_knowledge)
                json.dump(serializable_knowledge, f, indent=2)
            model_files['clinical_knowledge'] = str(knowledge_path)
            
            # Calculate file sizes and checksum
            total_size = sum(os.path.getsize(path) for path in model_files.values())
            checksum = self._calculate_directory_checksum(checkpoint_dir)
            
            # Create checkpoint metadata
            checkpoint = ModelCheckpoint(
                model_id=model_id,
                version=version,
                timestamp=datetime.now(),
                model_type="enterprise_multimodal_pipeline",
                performance_metrics=performance_metrics,
                model_config=pipeline.config,
                file_path=str(checkpoint_dir),
                file_size=total_size,
                checksum=checksum,
                validation_results=validation_results,
                training_info=training_info
            )
            
            # Add to registry
            if model_id not in self.model_registry:
                self.model_registry[model_id] = []
            
            self.model_registry[model_id].append(checkpoint)
            
            # Keep only the latest N checkpoints
            self.model_registry[model_id] = sorted(
                self.model_registry[model_id], 
                key=lambda x: x.timestamp, 
                reverse=True
            )[:self.max_checkpoints]
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints(model_id)
            
            # Save registry
            self._save_model_registry()
            
            logger.info(f"Model checkpoint saved: {model_id} v{version}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error saving model checkpoint: {str(e)}")
            raise

    def load_model_checkpoint(self, 
                            model_id: str, 
                            version: Optional[str] = None,
                            device: Optional[str] = None) -> EnterpriseMultiModalDiagnosticPipeline:
        """
        Load a model checkpoint
        
        Args:
            model_id: Model identifier
            version: Specific version to load (latest if None)
            device: Device to load model on
            
        Returns:
            Loaded EnterpriseMultiModalDiagnosticPipeline
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            # Find the right checkpoint
            checkpoints = self.model_registry[model_id]
            if version:
                checkpoint = next((cp for cp in checkpoints if cp.version == version), None)
                if not checkpoint:
                    raise ValueError(f"Version {version} not found for model {model_id}")
            else:
                # Use latest version
                checkpoint = max(checkpoints, key=lambda x: x.timestamp)
            
            checkpoint_dir = Path(checkpoint.file_path)
            
            # Load configuration
            config_path = checkpoint_dir / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create new pipeline instance
            pipeline = EnterpriseMultiModalDiagnosticPipeline(
                config=config,
                device=device,
                enable_monitoring=True
            )
            
            # Load neural network models
            for component_name in ['fusion', 'bayesian_classifier', 'temporal', 'severity']:
                model_path = checkpoint_dir / f"{component_name}.pth"
                if model_path.exists():
                    # SECURITY: Use weights_only=True to prevent arbitrary code execution
                    checkpoint_data = torch.load(model_path, map_location=device, weights_only=True)
                    pipeline.model_components[component_name].load_state_dict(
                        checkpoint_data['model_state_dict']
                    )
            
            # Load feature extractors from JSON (SEC: CWE-502 fix - no pickle)
            extractors_path = checkpoint_dir / "feature_extractors.json"
            # Also check for legacy pkl file for backward compatibility
            legacy_extractors_path = checkpoint_dir / "feature_extractors.pkl"
            if extractors_path.exists():
                try:
                    with open(extractors_path, 'r', encoding='utf-8') as f:
                        serialized = json.load(f)
                    pipeline.feature_extractors = self._deserialize_feature_extractors(serialized)
                except Exception as e:
                    logger.warning(f"Could not load feature extractors: {str(e)}")
            elif legacy_extractors_path.exists():
                # SEC: Skip loading legacy pickle files due to security risk (CWE-502)
                logger.warning(
                    f"Legacy pickle file found at {legacy_extractors_path}. "
                    "Skipping due to security risk. Please re-save checkpoint."
                )
            
            # Load clinical knowledge
            knowledge_path = checkpoint_dir / "clinical_knowledge.json"
            if knowledge_path.exists():
                with open(knowledge_path, 'r') as f:
                    pipeline.clinical_knowledge = json.load(f)
            
            logger.info(f"Model checkpoint loaded: {model_id} v{checkpoint.version}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading model checkpoint: {str(e)}")
            raise

    def validate_model(self, 
                      pipeline: EnterpriseMultiModalDiagnosticPipeline,
                      validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate model performance on validation dataset
        
        Args:
            pipeline: Pipeline to validate
            validation_data: List of validation samples
            
        Returns:
            Validation results dictionary
        """
        try:
            results = {
                'total_samples': len(validation_data),
                'successful_predictions': 0,
                'failed_predictions': 0,
                'average_processing_time': 0.0,
                'predictions': [],
                'performance_metrics': {}
            }
            
            processing_times = []
            predictions = []
            ground_truths = []
            
            # Process validation samples
            for sample in validation_data:
                try:
                    start_time = datetime.now()
                    
                    # Extract input data and ground truth
                    input_data = sample['input_data']
                    ground_truth = sample.get('ground_truth', {})
                    user_id = sample.get('user_id', 'validation_user')
                    session_id = sample.get('session_id', f'validation_{hash(str(sample))}')
                    
                    # Run prediction
                    result = asyncio.run(pipeline.process_multimodal_input(
                        input_data, user_id, session_id, enable_adaptation=False
                    ))
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    processing_times.append(processing_time)
                    
                    if result.get('success'):
                        results['successful_predictions'] += 1
                        
                        # Extract prediction for evaluation
                        if result.get('diagnostic_results', {}).get('conditions'):
                            primary_condition = result['diagnostic_results']['conditions'][0]
                            predictions.append({
                                'condition': primary_condition['name'],
                                'probability': primary_condition['probability'],
                                'confidence': result.get('confidence_level', 'unknown')
                            })
                        
                        # Store ground truth if available
                        if ground_truth:
                            ground_truths.append(ground_truth)
                    else:
                        results['failed_predictions'] += 1
                    
                    results['predictions'].append({
                        'input_sample': sample,
                        'prediction_result': result,
                        'processing_time': processing_time
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing validation sample: {str(e)}")
                    results['failed_predictions'] += 1
            
            # Calculate summary statistics
            if processing_times:
                results['average_processing_time'] = np.mean(processing_times)
                results['p95_processing_time'] = np.percentile(processing_times, 95)
                results['processing_time_std'] = np.std(processing_times)
            
            # Calculate performance metrics if ground truth available
            if predictions and ground_truths and len(predictions) == len(ground_truths):
                results['performance_metrics'] = self._calculate_performance_metrics(
                    predictions, ground_truths
                )
            
            # Success rate
            results['success_rate'] = results['successful_predictions'] / results['total_samples']
            
            logger.info(f"Model validation completed: {results['success_rate']:.3f} success rate")
            return results
            
        except Exception as e:
            logger.error(f"Error in model validation: {str(e)}")
            return {'error': str(e)}

    def compare_models(self, 
                      model_comparisons: List[Tuple[str, str]],
                      validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance of multiple model versions
        
        Args:
            model_comparisons: List of (model_id, version) tuples to compare
            validation_data: Validation dataset
            
        Returns:
            Comparison results
        """
        try:
            comparison_results = {
                'models_compared': len(model_comparisons),
                'validation_samples': len(validation_data),
                'results': {},
                'summary': {},
                'recommendation': None
            }
            
            # Validate each model
            for model_id, version in model_comparisons:
                try:
                    logger.info(f"Loading and validating {model_id} v{version}")
                    
                    # Load model
                    pipeline = self.load_model_checkpoint(model_id, version)
                    
                    # Validate
                    validation_results = self.validate_model(pipeline, validation_data)
                    
                    comparison_results['results'][f"{model_id}_v{version}"] = validation_results
                    
                except Exception as e:
                    logger.error(f"Error validating {model_id} v{version}: {str(e)}")
                    comparison_results['results'][f"{model_id}_v{version}"] = {'error': str(e)}
            
            # Generate summary and recommendation
            comparison_results['summary'] = self._generate_comparison_summary(
                comparison_results['results']
            )
            comparison_results['recommendation'] = self._generate_model_recommendation(
                comparison_results['results']
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in model comparison: {str(e)}")
            return {'error': str(e)}

    def auto_update_model(self, 
                         model_id: str,
                         new_training_data: List[Dict[str, Any]],
                         update_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Automatically update model with new training data if performance improves
        
        Args:
            model_id: Model to update
            new_training_data: New training samples
            update_threshold: Minimum improvement threshold
            
        Returns:
            Update results
        """
        try:
            logger.info(f"Starting auto-update for model {model_id}")
            
            # Load current best model
            current_pipeline = self.load_model_checkpoint(model_id)
            
            # Validate current model on new data
            current_performance = self.validate_model(current_pipeline, new_training_data)
            current_score = current_performance.get('success_rate', 0.0)
            
            # Create updated version (simplified incremental learning)
            # In production, this would involve proper fine-tuning
            updated_pipeline = self._create_updated_model(current_pipeline, new_training_data)
            
            # Validate updated model
            updated_performance = self.validate_model(updated_pipeline, new_training_data)
            updated_score = updated_performance.get('success_rate', 0.0)
            
            improvement = updated_score - current_score
            
            update_results = {
                'model_id': model_id,
                'current_performance': current_score,
                'updated_performance': updated_score,
                'improvement': improvement,
                'threshold': update_threshold,
                'should_update': improvement >= update_threshold,
                'timestamp': datetime.now().isoformat()
            }
            
            # Deploy updated model if improvement is significant
            if improvement >= update_threshold:
                # Generate new version number
                latest_checkpoint = max(self.model_registry[model_id], key=lambda x: x.timestamp)
                current_version = latest_checkpoint.version
                new_version = self._increment_version(current_version)
                
                # Save updated model
                checkpoint = self.save_model_checkpoint(
                    pipeline=updated_pipeline,
                    model_id=model_id,
                    version=new_version,
                    performance_metrics={'success_rate': updated_score, 'improvement': improvement},
                    validation_results=updated_performance,
                    training_info={
                        'update_type': 'auto_update',
                        'training_samples': len(new_training_data),
                        'base_version': current_version
                    }
                )
                
                update_results['new_checkpoint'] = checkpoint
                logger.info(f"Model {model_id} auto-updated to version {new_version}")
            else:
                logger.info(f"Model {model_id} auto-update skipped (improvement {improvement:.3f} < {update_threshold})")
            
            return update_results
            
        except Exception as e:
            logger.error(f"Error in auto-update: {str(e)}")
            return {'error': str(e)}

    def rollback_model(self, model_id: str, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rollback model to a previous version
        
        Args:
            model_id: Model to rollback
            target_version: Version to rollback to (previous if None)
            
        Returns:
            Rollback results
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")
            
            checkpoints = sorted(self.model_registry[model_id], key=lambda x: x.timestamp, reverse=True)
            
            if len(checkpoints) < 2:
                raise ValueError(f"No previous version available for rollback")
            
            if target_version:
                target_checkpoint = next((cp for cp in checkpoints if cp.version == target_version), None)
                if not target_checkpoint:
                    raise ValueError(f"Target version {target_version} not found")
            else:
                # Rollback to previous version
                target_checkpoint = checkpoints[1]  # Second most recent
            
            # Create rollback checkpoint (copy of target)
            current_time = datetime.now()
            rollback_version = f"rollback_{current_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Load target model
            target_pipeline = self.load_model_checkpoint(model_id, target_checkpoint.version)
            
            # Save as new checkpoint
            rollback_checkpoint = self.save_model_checkpoint(
                pipeline=target_pipeline,
                model_id=model_id,
                version=rollback_version,
                performance_metrics=target_checkpoint.performance_metrics,
                validation_results=target_checkpoint.validation_results,
                training_info={
                    'rollback_from': checkpoints[0].version,
                    'rollback_to': target_checkpoint.version,
                    'rollback_reason': 'manual_rollback',
                    'rollback_timestamp': current_time.isoformat()
                }
            )
            
            rollback_results = {
                'model_id': model_id,
                'rollback_from': checkpoints[0].version,
                'rollback_to': target_checkpoint.version,
                'rollback_version': rollback_version,
                'rollback_checkpoint': rollback_checkpoint,
                'timestamp': current_time.isoformat()
            }
            
            logger.info(f"Model {model_id} rolled back from {checkpoints[0].version} to {target_checkpoint.version}")
            return rollback_results
            
        except Exception as e:
            logger.error(f"Error in model rollback: {str(e)}")
            return {'error': str(e)}

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a model"""
        if model_id not in self.model_registry:
            return {'error': f"Model {model_id} not found"}
        
        checkpoints = self.model_registry[model_id]
        latest_checkpoint = max(checkpoints, key=lambda x: x.timestamp)
        
        return {
            'model_id': model_id,
            'total_checkpoints': len(checkpoints),
            'latest_version': latest_checkpoint.version,
            'latest_timestamp': latest_checkpoint.timestamp.isoformat(),
            'latest_performance': latest_checkpoint.performance_metrics,
            'model_type': latest_checkpoint.model_type,
            'total_storage_size': sum(cp.file_size for cp in checkpoints),
            'checkpoints': [
                {
                    'version': cp.version,
                    'timestamp': cp.timestamp.isoformat(),
                    'performance': cp.performance_metrics,
                    'file_size': cp.file_size
                }
                for cp in sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
            ]
        }

    def list_models(self) -> Dict[str, Any]:
        """List all models in the registry"""
        models_info = {}
        
        for model_id in self.model_registry:
            models_info[model_id] = self.get_model_info(model_id)
        
        return {
            'total_models': len(self.model_registry),
            'models': models_info,
            'registry_last_updated': datetime.now().isoformat()
        }

    def cleanup_models(self, older_than_days: int = 30) -> Dict[str, Any]:
        """Clean up old model checkpoints"""
        cleanup_results = {
            'models_processed': 0,
            'checkpoints_removed': 0,
            'storage_freed': 0,
            'errors': []
        }
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        for model_id, checkpoints in self.model_registry.items():
            cleanup_results['models_processed'] += 1
            
            # Keep at least 2 checkpoints per model
            checkpoints_to_keep = sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)[:2]
            checkpoints_to_remove = [cp for cp in checkpoints if cp.timestamp < cutoff_date and cp not in checkpoints_to_keep]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    # Remove checkpoint files
                    checkpoint_path = Path(checkpoint.file_path)
                    if checkpoint_path.exists():
                        shutil.rmtree(checkpoint_path)
                        cleanup_results['storage_freed'] += checkpoint.file_size
                        cleanup_results['checkpoints_removed'] += 1
                    
                    # Remove from registry
                    self.model_registry[model_id].remove(checkpoint)
                    
                except Exception as e:
                    cleanup_results['errors'].append(f"Error removing {checkpoint.file_path}: {str(e)}")
        
        # Save updated registry
        self._save_model_registry()
        
        logger.info(f"Cleanup completed: {cleanup_results['checkpoints_removed']} checkpoints removed, "
                   f"{cleanup_results['storage_freed']} bytes freed")
        
        return cleanup_results

    # Helper methods
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif callable(obj):
            return f"<function: {obj.__name__}>"
        elif hasattr(obj, '__dict__'):
            return f"<object: {obj.__class__.__name__}>"
        else:
            return obj

    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate MD5 checksum of directory contents"""
        md5_hash = hashlib.md5()
        
        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        md5_hash.update(chunk)
        
        return md5_hash.hexdigest()

    def _cleanup_old_checkpoints(self, model_id: str):
        """Remove old checkpoint files that are no longer in registry"""
        model_dir = self.models_directory / model_id
        if not model_dir.exists():
            return
        
        # Get versions that should be kept
        kept_versions = {cp.version for cp in self.model_registry[model_id]}
        
        # Remove directories for versions not in registry
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir() and version_dir.name not in kept_versions:
                try:
                    shutil.rmtree(version_dir)
                    logger.info(f"Removed old checkpoint: {version_dir}")
                except Exception as e:
                    logger.error(f"Error removing {version_dir}: {str(e)}")

    def _calculate_performance_metrics(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from predictions and ground truth"""
        try:
            # Extract labels for classification metrics
            pred_labels = [p['condition'] for p in predictions]
            true_labels = [gt.get('condition', 'unknown') for gt in ground_truths]
            
            # Filter out unknown ground truths
            valid_indices = [i for i, label in enumerate(true_labels) if label != 'unknown']
            
            if not valid_indices:
                return {'error': 'No valid ground truth labels available'}
            
            pred_labels_valid = [pred_labels[i] for i in valid_indices]
            true_labels_valid = [true_labels[i] for i in valid_indices]
            
            # Calculate metrics
            unique_labels = list(set(true_labels_valid + pred_labels_valid))
            
            metrics = {}
            
            if len(unique_labels) > 1:
                metrics['accuracy'] = accuracy_score(true_labels_valid, pred_labels_valid)
                metrics['precision'] = precision_score(true_labels_valid, pred_labels_valid, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(true_labels_valid, pred_labels_valid, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(true_labels_valid, pred_labels_valid, average='weighted', zero_division=0)
            
            # Average confidence
            confidences = [p.get('probability', 0.5) for p in predictions]
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'error': str(e)}

    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of model comparison results"""
        summary = {
            'best_model': None,
            'worst_model': None,
            'avg_success_rate': 0.0,
            'performance_range': {'min': 1.0, 'max': 0.0}
        }
        
        success_rates = {}
        
        for model_name, result in results.items():
            if 'error' not in result:
                success_rate = result.get('success_rate', 0.0)
                success_rates[model_name] = success_rate
        
        if success_rates:
            summary['best_model'] = max(success_rates.keys(), key=lambda x: success_rates[x])
            summary['worst_model'] = min(success_rates.keys(), key=lambda x: success_rates[x])
            summary['avg_success_rate'] = np.mean(list(success_rates.values()))
            summary['performance_range']['min'] = min(success_rates.values())
            summary['performance_range']['max'] = max(success_rates.values())
        
        return summary

    def _generate_model_recommendation(self, results: Dict[str, Any]) -> str:
        """Generate model recommendation based on comparison results"""
        success_rates = {}
        
        for model_name, result in results.items():
            if 'error' not in result:
                success_rate = result.get('success_rate', 0.0)
                avg_time = result.get('average_processing_time', float('inf'))
                
                # Score combines accuracy and speed (weighted)
                score = success_rate * 0.8 + (1.0 / (1.0 + avg_time)) * 0.2
                success_rates[model_name] = score
        
        if not success_rates:
            return "No valid models to recommend"
        
        best_model = max(success_rates.keys(), key=lambda x: success_rates[x])
        return f"Recommend using {best_model} (score: {success_rates[best_model]:.3f})"

    def _create_updated_model(self, base_pipeline: EnterpriseMultiModalDiagnosticPipeline, 
                            new_data: List[Dict[str, Any]]) -> EnterpriseMultiModalDiagnosticPipeline:
        """Create updated model with new training data (simplified version)"""
        # In production, this would involve proper incremental learning
        # For now, we return the base pipeline with minimal modifications
        
        # Create a copy of the base pipeline
        updated_pipeline = EnterpriseMultiModalDiagnosticPipeline(
            config=base_pipeline.config.copy(),
            device=base_pipeline.device,
            enable_monitoring=base_pipeline.enable_monitoring
        )
        
        # Copy model states
        for name, model in base_pipeline.model_components.items():
            if hasattr(model, 'state_dict'):
                updated_pipeline.model_components[name].load_state_dict(model.state_dict())
        
        # Copy other components
        updated_pipeline.feature_extractors = base_pipeline.feature_extractors
        updated_pipeline.clinical_knowledge = base_pipeline.clinical_knowledge
        
        # In a real implementation, you would:
        # 1. Extract features from new_data
        # 2. Perform incremental learning/fine-tuning
        # 3. Update model parameters
        
        return updated_pipeline

    def _increment_version(self, version: str) -> str:
        """Increment version number"""
        try:
            parts = version.split('.')
            if len(parts) >= 3:
                # Increment patch version
                parts[2] = str(int(parts[2]) + 1)
            else:
                # Simple increment
                parts.append('1')

            return '.'.join(parts)
        except (ValueError, TypeError, AttributeError, IndexError):
            # Fallback to timestamp-based version
            return f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Factory function for model manager
def create_model_manager(models_directory: str = "models", **kwargs) -> ModelManager:
    """
    Create a model manager instance
    
    Args:
        models_directory: Directory for storing models
        **kwargs: Additional arguments for ModelManager
        
    Returns:
        Configured ModelManager instance
    """
    return ModelManager(models_directory=models_directory, **kwargs)

# Example usage and testing
async def example_model_management():
    """Example of using the model management system"""
    
    # Create model manager
    manager = create_model_manager("example_models")
    
    # Create a sample pipeline
    from .enterprise_multimodal_pipeline import create_enterprise_pipeline
    pipeline = create_enterprise_pipeline()
    
    # Save initial checkpoint
    performance_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85
    }
    
    validation_results = {
        'success_rate': 0.85,
        'average_processing_time': 0.25
    }
    
    checkpoint = manager.save_model_checkpoint(
        pipeline=pipeline,
        model_id="diagnostic_model_v1",
        version="1.0.0",
        performance_metrics=performance_metrics,
        validation_results=validation_results
    )
    
    print(f"Saved checkpoint: {checkpoint.model_id} v{checkpoint.version}")
    
    # Load the model
    loaded_pipeline = manager.load_model_checkpoint("diagnostic_model_v1", "1.0.0")
    print("Model loaded successfully")
    
    # List models
    models_info = manager.list_models()
    print(f"Total models: {models_info['total_models']}")
    
    return manager

if __name__ == "__main__":
    # Run example
    asyncio.run(example_model_management())