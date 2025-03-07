from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import os
import uuid
from loguru import logger


# Type Definitions ---------------------------------------------------------------------------

class ModelType(str, Enum):
    """Types of models based on their creation method"""
    BASE = "base"                # Pre-trained base model
    FINE_TUNED = "fine-tuned"    # Model that underwent full fine-tuning


class ModelOrigin(str, Enum):
    """Origin of the model"""
    LOCAL = "local"      # Model created/fine-tuned locally
    REMOTE = "remote"    # Model from external source/hub


class AdapterType(str, Enum):
    """Types of parameter-efficient fine-tuning adapters"""
    LORA = "lora"         # Low-Rank Adaptation
    QLORA = "qlora"       # Quantized LoRA


class FineTuningMethod(str, Enum):
    """Available fine-tuning techniques"""
    SFT = "sft"           # Standard Supervised Fine-Tuning
    LORA = "lora"         # Low-Rank Adaptation
    QLORA = "qlora"       # Quantized LoRA


class FineTuningJobState(str, Enum):
    """States of a fine-tuning job through its lifecycle"""
    PENDING = "pending"
    VALIDATING = "validating"
    PREPARING = "preparing"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    PAUSED = "paused"


# Common Configuration Models ---------------------------------------------------------------------------

class Hyperparameters(BaseModel):
    """
    Configuration for fine-tuning processes
    Controls all aspects of training behavior and model adaptation
    """
    # Core method selection
    method: FineTuningMethod = Field(
        FineTuningMethod.SFT, 
        description="Fine-tuning approach to use"
    )
    
    # Basic training parameters (common across methods)
    epochs: int = Field(3, description="Number of training epochs", ge=1, le=10)
    learning_rate: float = Field(5e-5, description="Learning rate", ge=1e-6, le=1e-3)
    batch_size: int = Field(32, description="Training batch size", ge=1, le=256)
    
    # LoRA/QLoRA specific parameters
    lora_rank: Optional[int] = Field(16, description="LoRA rank (r)", ge=4, le=64)
    lora_alpha: Optional[float] = Field(32, description="LoRA alpha scaling factor", ge=8, le=64)
    lora_dropout: Optional[float] = Field(0.1, description="LoRA dropout probability", ge=0, le=0.5)
    
    # QLoRA specific
    quantization_bits: Optional[int] = Field(8, description="Bits for quantization (4 or 8)")
    
    # Advanced configuration (optional)
    advanced_config: Optional[Dict[str, Any]] = Field(None, description="Advanced configuration parameters")
    
    class Config:
        use_enum_values = True
        validate_assignment = True

    @model_validator(mode='after')
    def validate_method_requirements(self):
        """Ensure required parameters are provided based on selected method"""
        if self.method in [FineTuningMethod.LORA, FineTuningMethod.QLORA]:
            if self.lora_rank is None:
                raise ValueError("lora_rank is required for LoRA and QLoRA methods")
            if self.lora_alpha is None:
                raise ValueError("lora_alpha is required for LoRA and QLoRA methods")
                
        if self.method == FineTuningMethod.QLORA and self.quantization_bits not in [4, 8]:
            raise ValueError("quantization_bits must be 4 or 8 for QLoRA method")
            
        return self


class Metrics(BaseModel):
    """Metrics collected during training or evaluation"""
    train_loss: Optional[float] = Field(None, description="Training loss")
    valid_loss: Optional[float] = Field(None, description="Validation loss")
    accuracy: Optional[float] = Field(None, description="Accuracy (if applicable)")
    custom_metrics: Optional[Dict[str, float]] = Field(None, description="Custom metrics")


# Primary Entity Classes ---------------------------------------------------------------------------

class Model(BaseModel):
    """
    Model representation
    According to spec section 2.1.2 and 6
    """
    # Core identity fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique model identifier")
    name: str = Field(..., description="Model name")
    organization_id: str = Field(..., description="Organization identifier")
    
    # Model-specific attributes
    type: ModelType = Field(..., description="Model type (base or fine-tuned)")
    origin: ModelOrigin = Field(..., description="Origin of the model")
    
    # Additional details
    description: Optional[str] = Field(None, description="Model description")
    base_model_id: Optional[str] = Field(None, description="Base model identifier if fine-tuned")
    path: str = Field(..., description="Model filesystem path")
    
    # Fine-tuning related
    hyperparameters: Optional[Hyperparameters] = Field(None, description="Training hyperparameters")
    evaluation_metrics: Optional[Dict[str, Any]] = Field(None, description="Evaluation metrics")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    def get_full_path(self) -> str:
        """Construct the complete filesystem path to the model"""
        dir_path = os.path.join(self.path, self.name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created model directory at {dir_path}")
        return dir_path
        
    def get_info(self) -> Dict[str, Any]:
        """Get a summary of model information"""
        info = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "origin": self.origin,
            "created_at": self.created_at,
            "path": self.path
        }
        
        if self.description:
            info["description"] = self.description
            
        if self.type == ModelType.FINE_TUNED and self.base_model_id:
            info["base_model_id"] = self.base_model_id
            
        return info


class Adapter(BaseModel):
    """
    Adapter representation
    According to spec section 2.1.2 and 6
    """
    # Core identity fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique adapter identifier")
    name: str = Field(..., description="Adapter name")
    organization_id: str = Field(..., description="Organization identifier")
    
    # Adapter-specific attributes
    adapter_type: AdapterType = Field(..., description="Type of adapter")
    model_id: str = Field(..., description="Associated model identifier")
    dataset: str = Field(..., description="Dataset used for training")
    path: str = Field(..., description="Adapter filesystem path")
    
    # Fine-tuning details
    hyperparameters: Hyperparameters = Field(..., description="Training hyperparameters")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Adapter parameters")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @field_validator('model_id')
    def validate_model_id(cls, v):
        """Ensure model_id is valid"""
        if not v:
            raise ValueError("Model ID cannot be empty")
        return v
    
    def get_full_path(self) -> str:
        """Construct the complete filesystem path to the adapter"""
        dir_path = os.path.join(self.path, self.name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created adapter directory at {dir_path}")
        return dir_path
        
    def get_info(self) -> Dict[str, Any]:
        """Get a summary of adapter information"""
        return {
            "id": self.id,
            "name": self.name,
            "adapter_type": self.adapter_type,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "path": self.path
        }


class Checkpoint(BaseModel):
    """
    Training checkpoint representation
    According to spec section 2.1.4 and 6
    """
    # Core identity fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique checkpoint identifier")
    job_id: str = Field(..., description="Parent fine-tuning job identifier")
    
    # Checkpoint details
    step_number: int = Field(..., description="Step at which checkpoint was created")
    path: str = Field(..., description="Path to checkpoint files")
    
    # Metrics at checkpoint
    metrics: Metrics = Field(default_factory=Metrics, description="Metrics at checkpoint")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    def get_full_path(self) -> str:
        """Get the complete path to the checkpoint files"""
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
            logger.info(f"Created checkpoint directory at {self.path}")
        return self.path
        
    def get_info(self) -> Dict[str, Any]:
        """Get a summary of checkpoint information"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "step_number": self.step_number,
            "metrics": {
                "train_loss": self.metrics.train_loss,
                "valid_loss": self.metrics.valid_loss
            },
            "created_at": self.created_at,
            "path": self.path
        }


class File(BaseModel):
    """
    File representation for datasets, models, and logs
    According to spec section 6
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique file identifier")
    path: str = Field(..., description="File system path")
    type: Literal["dataset", "model", "log"] = Field(..., description="File type")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    size: Optional[int] = Field(None, description="File size in bytes")
    
    @field_validator('path')
    def validate_path(cls, v):
        """Ensure path exists"""
        if not os.path.exists(v):
            logger.error(f"File path does not exist: {v}")
            raise ValueError(f"File path does not exist: {v}")
        return v
    
    @model_validator(mode='after')
    def set_size(self):
        """Set file size if not provided"""
        if self.size is None and os.path.isfile(self.path):
            self.size = os.path.getsize(self.path)
            logger.debug(f"Set file size for {self.path}: {self.size} bytes")
        return self
        
    def get_info(self) -> Dict[str, Any]:
        """Get a summary of file information"""
        return {
            "id": self.id,
            "path": self.path,
            "type": self.type,
            "size": self.size,
            "created_at": self.created_at
        }


class Event(BaseModel):
    """
    Event representation for fine-tuning job notifications
    According to spec section 6
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event identifier")
    job_id: str = Field(..., description="Related fine-tuning job identifier")
    level: Literal["info", "error"] = Field(..., description="Event severity level")
    message: str = Field(..., description="Event message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    
    def __post_init__(self):
        """Log the event when created"""
        if self.level == "info":
            logger.info(f"[Job {self.job_id}] {self.message}")
        else:
            logger.error(f"[Job {self.job_id}] {self.message}")
            
    def get_info(self) -> Dict[str, Any]:
        """Get a summary of event information"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "level": self.level,
            "message": self.message,
            "created_at": self.created_at
        }


class Hub(BaseModel):
    """
    Organization's model and adapter repository
    According to spec section 2.1.3 and 3.1
    """
    # Core identity fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique hub identifier")
    name: str = Field(..., description="Hub name")
    organization_id: str = Field(..., description="Organization identifier")
    
    # Storage
    path: str = Field(..., description="Hub filesystem path")
    
    # Registered assets - dictionaries for faster lookups
    models: Dict[str, Model] = Field(default_factory=dict, description="Models by ID")
    models_by_name: Dict[str, str] = Field(default_factory=dict, description="Model names to IDs mapping")
    adapters: Dict[str, Adapter] = Field(default_factory=dict, description="Adapters by ID")
    adapters_by_name: Dict[str, str] = Field(default_factory=dict, description="Adapter names to IDs mapping")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    description: Optional[str] = Field(None, description="Hub description")
    
    def register_model(self, model: Model) -> bool:
        """
        Register a model in the hub
        
        Args:
            model: Model to register
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if model.id in self.models:
            logger.warning(f"Model with ID {model.id} already exists in hub {self.name}")
            return False
            
        if model.name in self.models_by_name:
            existing_id = self.models_by_name[model.name]
            logger.warning(f"Model with name '{model.name}' already exists with ID {existing_id}")
            return False
            
        # Register the model
        self.models[model.id] = model
        self.models_by_name[model.name] = model.id
        logger.info(f"Model '{model.name}' (ID: {model.id}) registered in hub {self.name}")
        return True
    
    def register_adapter(self, adapter: Adapter) -> bool:
        """
        Register an adapter in the hub
        
        Args:
            adapter: Adapter to register
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if adapter.id in self.adapters:
            logger.warning(f"Adapter with ID {adapter.id} already exists in hub {self.name}")
            return False
            
        if adapter.name in self.adapters_by_name:
            existing_id = self.adapters_by_name[adapter.name]
            logger.warning(f"Adapter with name '{adapter.name}' already exists with ID {existing_id}")
            return False
            
        # Register the adapter
        self.adapters[adapter.id] = adapter
        self.adapters_by_name[adapter.name] = adapter.id
        logger.info(f"Adapter '{adapter.name}' (ID: {adapter.id}) registered in hub {self.name}")
        return True
    
    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """
        Get a model by its ID
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model if found, None otherwise
        """
        if model_id not in self.models:
            logger.warning(f"Model with ID {model_id} not found in hub {self.name}")
            return None
        return self.models[model_id]
    
    def get_model_by_name(self, model_name: str) -> Optional[Model]:
        """
        Get a model by its name
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model if found, None otherwise
        """
        if model_name not in self.models_by_name:
            logger.warning(f"Model with name '{model_name}' not found in hub {self.name}")
            return None
        model_id = self.models_by_name[model_name]
        return self.models[model_id]
    
    def get_adapter_by_id(self, adapter_id: str) -> Optional[Adapter]:
        """
        Get an adapter by its ID
        
        Args:
            adapter_id: ID of the adapter to retrieve
            
        Returns:
            Adapter if found, None otherwise
        """
        if adapter_id not in self.adapters:
            logger.warning(f"Adapter with ID {adapter_id} not found in hub {self.name}")
            return None
        return self.adapters[adapter_id]
    
    def get_adapter_by_name(self, adapter_name: str) -> Optional[Adapter]:
        """
        Get an adapter by its name
        
        Args:
            adapter_name: Name of the adapter to retrieve
            
        Returns:
            Adapter if found, None otherwise
        """
        if adapter_name not in self.adapters_by_name:
            logger.warning(f"Adapter with name '{adapter_name}' not found in hub {self.name}")
            return None
        adapter_id = self.adapters_by_name[adapter_name]
        return self.adapters[adapter_id]
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the hub
        
        Args:
            model_id: ID of the model to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        if model_id not in self.models:
            logger.warning(f"Cannot remove model with ID {model_id} - not found in hub {self.name}")
            return False
            
        model = self.models[model_id]
        del self.models_by_name[model.name]
        del self.models[model_id]
        logger.info(f"Model '{model.name}' (ID: {model_id}) removed from hub {self.name}")
        return True
    
    def remove_adapter(self, adapter_id: str) -> bool:
        """
        Remove an adapter from the hub
        
        Args:
            adapter_id: ID of the adapter to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        if adapter_id not in self.adapters:
            logger.warning(f"Cannot remove adapter with ID {adapter_id} - not found in hub {self.name}")
            return False
            
        adapter = self.adapters[adapter_id]
        del self.adapters_by_name[adapter.name]
        del self.adapters[adapter_id]
        logger.info(f"Adapter '{adapter.name}' (ID: {adapter_id}) removed from hub {self.name}")
        return True
        
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all models in the hub
        
        Returns:
            List of model summaries
        """
        return [
            {
                "id": model.id,
                "name": model.name,
                "type": model.type,
                "origin": model.origin,
                "created_at": model.created_at
            }
            for model in self.models.values()
        ]
        
    def list_adapters(self) -> List[Dict[str, Any]]:
        """
        Get a list of all adapters in the hub
        
        Returns:
            List of adapter summaries
        """
        return [
            {
                "id": adapter.id,
                "name": adapter.name,
                "adapter_type": adapter.adapter_type,
                "model_id": adapter.model_id,
                "created_at": adapter.created_at
            }
            for adapter in self.adapters.values()
        ]


class FineTuningJob(BaseModel):
    """
    Fine-tuning process tracker
    According to spec section 2.1.4 and 6
    """
    # Core identity fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job identifier")
    name: str = Field(..., description="Job name")
    organization_id: str = Field(..., description="Organization identifier")
    
    # Job configuration
    base_model: str = Field(..., description="Base model name or path")
    dataset: str = Field(..., description="Training dataset path")
    val_dataset: Optional[str] = Field(None, description="Validation dataset path")
    hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters, 
        description="Training hyperparameters"
    )
    
    # Timing information
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    
    # State tracking
    status: FineTuningJobState = Field(FineTuningJobState.PENDING, description="Current job status") 
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    
    # Progress tracking
    current_epoch: int = Field(0, description="Current training epoch")
    total_epochs: int = Field(None, description="Total number of epochs")
    
    # Resources
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    gpu_usage: Optional[float] = Field(None, description="GPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    
    # Artifacts and outputs
    checkpoints: List[str] = Field(default_factory=list, description="Checkpoint identifiers")
    result_files: List[str] = Field(default_factory=list, description="Result file paths")
    logs_path: Optional[str] = Field(None, description="Path to log files")
    
    # Hub integration
    hub_id: Optional[str] = Field(None, description="Target hub identifier")
    
    # Output artifacts
    output_model_id: Optional[str] = Field(None, description="ID of resulting model (if applicable)")
    output_adapter_id: Optional[str] = Field(None, description="ID of resulting adapter (if applicable)")
    
    # Additional metadata
    description: Optional[str] = Field(None, description="Job description")
    
    class Config:
        validate_assignment = True

    @field_validator('dataset')
    def validate_dataset(cls, v):
        """Ensure dataset exists"""
        if not os.path.exists(v):
            raise ValueError(f"Dataset path does not exist: {v}")
        return v
    
    @field_validator('val_dataset')
    def validate_val_dataset(cls, v):
        """Ensure validation dataset exists if specified"""
        if v is not None and not os.path.exists(v):
            raise ValueError(f"Validation dataset path does not exist: {v}")
        return v
    
    @property
    def method(self) -> str:
        """Get the fine-tuning method from hyperparameters"""
        return self.hyperparameters.method
    
    @property
    def is_peft(self) -> bool:
        """Check if job uses parameter-efficient fine-tuning"""
        return self.method in [FineTuningMethod.LORA, FineTuningMethod.QLORA]
    
    @property
    def is_complete(self) -> bool:
        """Check if job is in a terminal state"""
        return self.status in [
            FineTuningJobState.SUCCEEDED, 
            FineTuningJobState.FAILED, 
            FineTuningJobState.CANCELED
        ]
    
    @property
    def progress(self) -> float:
        """Calculate job progress as percentage"""
        if self.is_complete:
            return 100.0
        
        if not self.started_at or self.total_epochs is None or self.total_epochs == 0:
            return 0.0
            
        # Calculate progress based on epochs
        if self.current_epoch is not None and self.total_epochs > 0:
            return (self.current_epoch / self.total_epochs) * 100
            
        return 0.0
    
    def start(self) -> bool:
        """Initiate job execution"""
        if self.status in [FineTuningJobState.PENDING, FineTuningJobState.QUEUED]:
            logger.info(f"Starting job {self.name} (ID: {self.id})")
            self.status = FineTuningJobState.RUNNING
            self.started_at = datetime.utcnow()
            self.total_epochs = self.hyperparameters.epochs
            logger.info(f"Job {self.name} started at {self.started_at} with {self.total_epochs} epochs")
            return True
        logger.warning(f"Cannot start job {self.name} (ID: {self.id}) with status {self.status}")
        return False
    
    def complete(self, success: bool = True, error: str = None) -> bool:
        """Mark job as finished"""
        if self.status in [FineTuningJobState.RUNNING, FineTuningJobState.PAUSED]:
            self.completed_at = datetime.utcnow()
            if self.started_at:
                self.duration = (self.completed_at - self.started_at).total_seconds()
            
            if success:
                self.status = FineTuningJobState.SUCCEEDED
                self.current_epoch = self.total_epochs
                logger.info(f"Job {self.name} (ID: {self.id}) completed successfully in {self.duration:.2f} seconds")
            else:
                self.status = FineTuningJobState.FAILED
                self.error_message = error
                logger.error(f"Job {self.name} (ID: {self.id}) failed: {error}")
            return True
        logger.warning(f"Cannot complete job {self.name} (ID: {self.id}) with status {self.status}")
        return False
    
    def pause(self) -> bool:
        """Temporarily halt job execution"""
        if self.status == FineTuningJobState.RUNNING:
            logger.info(f"Pausing job {self.name} (ID: {self.id})")
            self.status = FineTuningJobState.PAUSED
            return True
        logger.warning(f"Cannot pause job {self.name} (ID: {self.id}) with status {self.status}")
        return False
    
    def resume(self) -> bool:
        """Continue job execution after pause"""
        if self.status == FineTuningJobState.PAUSED:
            logger.info(f"Resuming job {self.name} (ID: {self.id})")
            self.status = FineTuningJobState.RUNNING
            return True
        logger.warning(f"Cannot resume job {self.name} (ID: {self.id}) with status {self.status}")
        return False
    
    def cancel(self) -> bool:
        """Terminate job execution"""
        if self.status in [
            FineTuningJobState.RUNNING, 
            FineTuningJobState.PAUSED, 
            FineTuningJobState.QUEUED, 
            FineTuningJobState.PENDING
        ]:
            logger.info(f"Canceling job {self.name} (ID: {self.id})")
            self.status = FineTuningJobState.CANCELED
            self.completed_at = datetime.utcnow()
            if self.started_at:
                self.duration = (self.completed_at - self.started_at).total_seconds()
                logger.info(f"Job {self.name} was active for {self.duration:.2f} seconds before cancellation")
            return True
        logger.warning(f"Cannot cancel job {self.name} (ID: {self.id}) with status {self.status}")
        return False
    
    def update_progress(self, epoch: int, metrics: Optional[Metrics] = None, resource_usage: Dict[str, float] = None) -> None:
        """
        Update training progress and metrics
        
        Args:
            epoch: Current epoch number
            metrics: Current training metrics
            resource_usage: Resource usage information
        """
        self.current_epoch = epoch
        logger.info(f"Job {self.name} (ID: {self.id}) progress: Epoch {epoch}/{self.total_epochs}")
        
        if metrics:
            if metrics.train_loss is not None:
                logger.info(f"Job {self.name} - Train loss: {metrics.train_loss:.4f}")
            if metrics.valid_loss is not None:
                logger.info(f"Job {self.name} - Validation loss: {metrics.valid_loss:.4f}")
                
        if resource_usage:
            if 'cpu' in resource_usage:
                self.cpu_usage = resource_usage['cpu']
            if 'gpu' in resource_usage:
                self.gpu_usage = resource_usage['gpu']
            if 'memory' in resource_usage:
                self.memory_usage = resource_usage['memory']
                
            logger.debug(f"Job {self.name} resources - CPU: {self.cpu_usage}%, GPU: {self.gpu_usage}%, Memory: {self.memory_usage}MB")
    
    def add_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Register a model checkpoint
        
        Args:
            checkpoint: Checkpoint to register
            
        Returns:
            bool: True if registered successfully
        """
        if checkpoint.id in self.checkpoints:
            logger.warning(f"Checkpoint {checkpoint.id} already registered for job {self.name}")
            return False
            
        self.checkpoints.append(checkpoint.id)
        logger.info(f"Checkpoint at step {checkpoint.step_number} registered for job {self.name}")
        
        if checkpoint.metrics and checkpoint.metrics.train_loss:
            logger.info(f"Checkpoint metrics - Train loss: {checkpoint.metrics.train_loss:.4f}, " +
                       f"Valid loss: {checkpoint.metrics.valid_loss or 'N/A'}")
        
        return True
    
    def register_results_in_hub(self, hub: Hub, model: Optional[Model] = None, adapter: Optional[Adapter] = None) -> Dict[str, Any]:
        """Register job outputs in the hub"""
        results = {}
        
        if self.status != FineTuningJobState.SUCCEEDED:
            logger.warning(f"Cannot register results for job {self.id} with status {self.status}")
            return results
        
        # Register model if produced
        if model:
            self.output_model_id = model.id
            if hub.register_model(model):
                results["model_registered"] = True
                results["model_id"] = model.id
                logger.info(f"Model '{model.name}' registered from job {self.name}")
            else:
                logger.warning(f"Failed to register model '{model.name}' from job {self.name}")
        
        # Register adapter if produced
        if adapter:
            self.output_adapter_id = adapter.id
            if hub.register_adapter(adapter):
                results["adapter_registered"] = True
                results["adapter_id"] = adapter.id
                logger.info(f"Adapter '{adapter.name}' registered from job {self.name}")
            else:
                logger.warning(f"Failed to register adapter '{adapter.name}' from job {self.name}")
                
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate a concise job summary"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "method": self.method,
            "base_model": self.base_model,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "error": self.error_message,
            "output_model_id": self.output_model_id,
            "output_adapter_id": self.output_adapter_id,
        }