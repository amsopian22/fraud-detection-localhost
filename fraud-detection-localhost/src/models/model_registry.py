# src/models/model_registry.py
import json
import joblib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    version: str
    name: str
    algorithm: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    features: List[str]
    training_data_hash: str
    model_size_mb: float
    author: str
    description: str
    tags: List[str]

class ModelRegistry:
    """Central model registry for managing model versions"""
    
    def __init__(self, registry_path: str = "/app/models/model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path = self.registry_path.parent / "trained_models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self.registry_file = self.registry_path / "registry.json"
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load model registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    
                models = {}
                for model_id, model_data in data.items():
                    # Convert datetime strings back to datetime objects
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                    model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                    model_data['status'] = ModelStatus(model_data['status'])
                    
                    models[model_id] = ModelMetadata(**model_data)
                
                logger.info(f"Loaded {len(models)} models from registry")
                return models
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                return {}
        
        return {}
    
    def _save_registry(self):
        """Save model registry to file"""
        try:
            data = {}
            for model_id, metadata in self.models.items():
                model_dict = {
                    'model_id': metadata.model_id,
                    'version': metadata.version,
                    'name': metadata.name,
                    'algorithm': metadata.algorithm,
                    'status': metadata.status.value,
                    'created_at': metadata.created_at.isoformat(),
                    'updated_at': metadata.updated_at.isoformat(),
                    'metrics': metadata.metrics,
                    'hyperparameters': metadata.hyperparameters,
                    'features': metadata.features,
                    'training_data_hash': metadata.training_data_hash,
                    'model_size_mb': metadata.model_size_mb,
                    'author': metadata.author,
                    'description': metadata.description,
                    'tags': metadata.tags
                }
                data[model_id] = model_dict
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved registry with {len(data)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, 
                      model,
                      name: str,
                      algorithm: str,
                      metrics: Dict[str, float],
                      features: List[str],
                      hyperparameters: Dict[str, Any] = None,
                      description: str = "",
                      author: str = "system",
                      tags: List[str] = None) -> str:
        """Register a new model in the registry"""
        
        # Generate model ID and version
        model_id = f"{name}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model file
        model_filename = f"{model_id}.joblib"
        model_path = self.models_path / model_filename
        
        try:
            joblib.dump(model, model_path)
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            algorithm=algorithm,
            status=ModelStatus.TRAINING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metrics=metrics,
            hyperparameters=hyperparameters or {},
            features=features,
            training_data_hash="",  # Would be calculated from training data
            model_size_mb=model_size_mb,
            author=author,
            description=description,
            tags=tags or []
        )
        
        # Add to registry
        self.models[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata by ID"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.models[model_id]
        model_filename = f"{model_id}.joblib"
        model_path = self.models_path / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model: {model_id}")
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def get_production_model(self) -> Optional[Tuple[Any, ModelMetadata]]:
        """Get the current production model"""
        production_models = [
            (model_id, metadata) for model_id, metadata in self.models.items()
            if metadata.status == ModelStatus.PRODUCTION
        ]
        
        if not production_models:
            logger.warning("No production model found")
            return None
        
        # Return the most recent production model
        latest_model = max(production_models, key=lambda x: x[1].updated_at)
        model_id = latest_model[0]
        
        return self.get_model(model_id)
    
    def promote_model(self, model_id: str, status: ModelStatus):
        """Promote model to a new status"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # If promoting to production, demote current production model
        if status == ModelStatus.PRODUCTION:
            for mid, metadata in self.models.items():
                if metadata.status == ModelStatus.PRODUCTION:
                    metadata.status = ModelStatus.DEPRECATED
                    metadata.updated_at = datetime.now()
                    logger.info(f"Demoted model {mid} from production")
        
        # Update status
        self.models[model_id].status = status
        self.models[model_id].updated_at = datetime.now()
        self._save_registry()
        
        logger.info(f"Promoted model {model_id} to {status.value}")
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List models, optionally filtered by status"""
        models = list(self.models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        # Sort by updated_at descending
        models.sort(key=lambda x: x.updated_at, reverse=True)
        return models
    
    def delete_model(self, model_id: str):
        """Delete model from registry and filesystem"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Don't delete production models
        if self.models[model_id].status == ModelStatus.PRODUCTION:
            raise ValueError("Cannot delete production model")
        
        # Delete model file
        model_filename = f"{model_id}.joblib"
        model_path = self.models_path / model_filename
        
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted model file: {model_path}")
        
        # Remove from registry
        del self.models[model_id]
        self._save_registry()
        
        logger.info(f"Deleted model: {model_id}")