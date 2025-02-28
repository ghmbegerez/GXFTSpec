from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any  
from datetime import datetime
import os


# Base Entity Class ---------------------------------------------------------------------------
class EntityTypes(str, Enum):
    """Types of entities"""
    DATASET = "dataset"
    MODEL = "model"
    ADAPTER = "adapter"
    DEPLOYMENT = "deployment"
    FINE_TUNING_JOB = "fine_tuning_job"
    
class BaseEntity(BaseModel):
    """
    Base class for all entities in the system.
    Provides common fields shared across entity types.
    """
    id: str = Field(..., description="Entity ID")
    name: str = Field(..., description="Entity name")
    organization_id: str = Field(..., description="Organization ID")
    project_id: str = Field(..., description="Project ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation date")
    version: str = Field("1.0", description="Entity version")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
# Datatsets ---------------------------------------------------------------------------------------    
class DatasetType(str, Enum):
    """Types of datasets"""
    JSON = "json" 
    CSV = "csv"
    PARQUET = "parquet"
    JSONL = "jsonl"
    HUGGINGFACE = "huggingface"

class DatasetOrigin(str, Enum):
    """Dataset origin"""
    FILE = "file"
    SINTHETIC = "sinthetic"
    REMOTE = "remote"
    HUGGINFACE = "huggingface"

class DatasetStatus(str, Enum):
    """Dataset status"""
    PENDING = "pending"
    VALIDATING = "validating"
    VALIDATED = "validated"
    FAILED = "failed"

class DatasetSchema(str, Enum):
    """Dataset schema defines the expected format and structure"""
    COMPLETION = "completion"  # Text completion format (e.g., single completions)
    CHAT = "chat"  # Chat format (e.g., conversation turns)
    INSTRUCTION = "instruction"  # Instruction-response pairs
    PREFERENCE = "preference"  # For preference data
    CUSTOM = "custom"  # Custom schema

class DatasetFormatField(BaseModel):
    """
    DatasetFormat class
    This class is used to store the format of a dataset.
    """
    name: str = Field(..., description="Field name")
    data_type: str = Field("string", description="Data type (string, number, boolean, etc.)")
    required: bool = Field(False, description="Whether this field is required")
    description: Optional[str] = Field(None, description="Field description")
    examples: Optional[List[Any]] = Field(None, description="Example values")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DatasetFormat(BaseModel):
    """
    DatasetFormat class
    This class is used to store the format of a dataset.
    """
    name: str = Field(..., description="Format name")
    fields: List[DatasetFormatField] = Field([], description="List of fields")
    primary_input_field: Optional[str] = Field(None, description="Primary input field name")
    primary_output_field: Optional[str] = Field(None, description="Primary output field name")
    data_schema: DatasetSchema = Field(DatasetSchema.CUSTOM, description="Dataset schema")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @field_validator('fields')
    def validate_field_names_unique(cls, fields):
        """Ensure field names are unique"""
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("Field names must be unique")
        return fields
    
    @field_validator('primary_input_field', 'primary_output_field')
    def validate_field_exists(cls, field_name, values):
        """Ensure referenced fields exist in the fields list"""
        if field_name is None:
            return field_name
            
        if 'fields' not in values.data:
            return field_name
            
        field_names = [f.name for f in values.data['fields']]
        if field_name not in field_names:
            raise ValueError(f"Field '{field_name}' not found in fields list")
        return field_name
    
class DatasetValidationResult(BaseModel):
    """
    Validation results for a dataset
    """
    is_valid: bool = Field(..., description="Whether the dataset is valid")
    errors: List[str] = Field([], description="List of validation errors")
    warnings: List[str] = Field([], description="List of validation warnings")
    statistics: Dict[str, Any] = Field({}, description="Dataset statistics")

class Dataset(BaseEntity):
    """
    Dataset class
    This class is used to store the information of a dataset.
    """
    content: Any = Field(None, description="Dataset content")
    format: DatasetFormat = Field(None, description="Dataset format")
    type: DatasetType = Field(DatasetType.JSON, description="Dataset type")
    origin: DatasetOrigin = Field(DatasetOrigin.FILE, description="Dataset origin")
    sample_count: Optional[int] = Field(None, description="Number of samples")
    validation_result: Optional[DatasetValidationResult] = Field(None, description="Validation result")
    status: DatasetStatus = Field(DatasetStatus.PENDING, description="Dataset status")

    def validate_dataset(self) -> DatasetValidationResult:
        """Validate dataset against its format"""
        raise NotImplementedError("Method must be implemented by subclasses")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        raise NotImplementedError("Method must be implemented by subclasses")
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, 'Dataset']:
        """Split dataset into training, validation, and test sets"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Ratios must sum to 1.0")
        
        # Implementation to be provided in subclasses
        raise NotImplementedError("Method must be implemented by subclasses")
    
# Models -----------------------------------------------------------------------------------------
class ModelOrigin(str, Enum):
    """Model origin"""
    LOCAL = "local"
    REMOTE = "remote"
    HUGGINGFACE = "huggingface"

class ModelType(str, Enum):
    """Types of models"""
    BASE = "base"
    FINE_TUNED = "fine_tuned"
    MERGED = "merged"
        
class Hyperparameters(BaseModel):
    """
    Hyperparameters class
    This class is used to store the hyperparameters of a model.
    """
    # Base training parameters
    epochs: int = Field(1, description="Number of epochs", ge=1)
    learning_rate: float = Field(5e-5, description="Learning rate", gt=0)
    batch_size: int = Field(32, description="Batch size", gt=0)
    weight_decay: float = Field(0.01, description="Weight decay", ge=0)
    warmup_ratio: float = Field(0.1, description="Warmup ratio", ge=0, le=1)

    # LoRA specific parameters
    lora_rank: Optional[int] = Field(8, description="LoRA rank", ge=1)
    lora_alpha: Optional[float] = Field(16, description="LoRA alpha scaling", gt=0)
    lora_dropout: Optional[float] = Field(0.05, description="LoRA dropout", ge=0, le=1)
    target_modules: Optional[List[str]] = Field(None, description="Target modules for LoRA")

    # Quantization specific parameters
    quantization_bits: Optional[int] = Field(None, description="Quantization bits (4/8)")
    quantized: Optional[bool] = Field(False, description="Quantized model")   

    # Optimization parameters
    gradient_accumulation_steps: Optional[int] = Field(1, description="Gradient accumulation steps", ge=1)
    gradient_checkpointing: Optional[bool] = Field(False, description="Use gradient checkpointing")
    max_grad_norm: Optional[float] = Field(1.0, description="Maximum gradient norm", gt=0)
    
    # Additional parameters
    seed: Optional[int] = Field(42, description="Random seed")
    fp16: Optional[bool] = Field(False, description="Use FP16 precision")
    bf16: Optional[bool] = Field(False, description="Use BF16 precision")    

    @field_validator('quantization_bits')
    def validate_quantization_bits(cls, v):
        if v is not None and v not in [4, 8]:
            raise ValueError("Quantization bits must be 4 or 8")
        return v
    
    @field_validator('fp16', 'bf16')
    def validate_precision(cls, v, values):
        if v and 'fp16' in values and 'bf16' in values and values['fp16'] and values['bf16']:
            raise ValueError("Cannot use both FP16 and BF16 precision")
        return v
        
class Model(BaseEntity):
    """
    Model class
    This class is used to store the information of a model.
    """
    type: str = Field(ModelType.BASE, description="Model type") 
    origin: str = Field(ModelOrigin.LOCAL, description="Model origin") 
    path: str = Field(..., description="Model path")
    base_model: Optional[str] = Field(None, description="Base model")
    dataset_id: str = Field(..., description="Dataset path")
    hyperparameters: Hyperparameters = Field(None, description="Hyperparameters") 
    adapters: Optional[List[str]] = Field(None, description="List of adapter IDs")
    merged: Optional[bool] = Field(False, description="Merged model")
    #Evaluation metrics
    evaluation_metrics: Optional[Dict[str, Any]] = Field(None, description="Evaluation metrics")

    @field_validator('dataset_id')
    def validate_dataset_id(cls, v):
        """Ensure dataset_id is a valid UUID format"""
        # Could add logic to check if dataset exists in database
        if not v:
            raise ValueError("Dataset ID cannot be empty")
        return v
    
    def get_full_path(self) -> str:
        """Get the full path to the model"""
        path = os.path.join(self.model_path, self.name, self.version)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

# Adapters ---------------------------------------------------------------------------------------
class AdapterType(str, Enum):    
    """Types of adapters for fine-tuning"""
    LORA = "lora"
    QLORA = "qlora"
    IA3 = "ia3"  # Addition of IA³ adapters
    PREFIX = "prefix"  # Prefix tuning
    PROMPT = "prompt"  # Prompt tuning (soft prompts)

class Adapter(BaseEntity):
    """
    Adapter class
    This class is used to store the information of an adapter.
    """
    type: str = Field(AdapterType.LORA, description="Adapter type")    
    base_model_id: str = Field(..., description="Model ID")
    dataset_id: str = Field(..., description="Dataset ID")
    hyperparameters: Hyperparameters = Field(None, description="Hyperparameters")
    path: str = Field(..., description="Adapter path")
    version: str = Field("1.0", description="Adapter version")
    compatible_models: Optional[List[str]] = Field(None, description="Compatible model IDs")
    # Evaluation
    evaluation_metrics: Optional[Dict[str, Any]] = Field(None, description="Evaluation metrics")
    
    @field_validator('base_model_id')
    def validate_model_id(cls, v):
        """Ensure model_id is a valid UUID format"""
        # Could add logic to check if model exists in database
        if not v:
            raise ValueError("Model ID cannot be empty")
        return v
    
    def is_compatible_with(self, model_id: str) -> bool:
        """Check if adapter is compatible with a model"""
        if not self.compatible_models:
            return self.base_model_id == model_id
        return model_id in self.compatible_models
    
    def get_full_path(self) -> str:
        """Get the full path to the adapter"""
        return os.path.join(self.path, self.name, self.version)    

# Deployments ------------------------------------------------------------------------------------
class DeploymentStatus(str, Enum):
    """
    Deployment status
    Defines the possible states of a deployment
    """
    PENDING = "pending"      # Initial state when deployment is created
    PROVISIONING = "provisioning"  # Resources are being allocated
    DEPLOYING = "deploying"  # Model is being deployed
    DEPLOYED = "deployed"    # Deployment is active and serving
    FAILED = "failed"        # Deployment failed
    STOPPING = "stopping"    # Deployment is being stopped
    STOPPED = "stopped"      # Deployment is stopped but can be restarted
    DELETING = "deleting"    # Deployment is being deleted
    DELETED = "deleted"      # Deployment has been deleted

class DeploymentEnvironment(str, Enum):
    """
    Deployment environment
    Defines the environment where the deployment runs
    """
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

    
class Deployment(BaseEntity):
    """
    Deployment class
    This class is used to store the information of a deployment.
    Can generate a deployment from a model.
    Can generate a deployment from a model and a list of adapters. This deployment generates a new model with the adapters declared. This adapters can be merged with the model if merge is indicated.    
    """
    name: str = Field(..., description="Unique name for the deployment")
    description: Optional[str] = Field(None, description="Description of the deployment")
    base_model_id: str = Field(..., description="Model ID")
    adapters_id: Optional[List[str]] = Field(None, description="List of adapter IDs")
    status: str = Field(DeploymentStatus.PENDING, description="Deployment status")
    environment: DeploymentEnvironment = Field(DeploymentEnvironment.DEVELOPMENT, description="Deployment environment")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update date")
    merge: Optional[bool] = Field(False, description="Merge model")
    
    @field_validator('base_model_id')
    def validate_model_id(cls, v):
        """Ensure model_id is valid"""
        # Could add logic to check if model exists in database
        if not v:
            raise ValueError("Model ID cannot be empty")
        return v
    
    @field_validator('adapters_id')
    def validate_adapters_id(cls, v):
        """Ensure adapters are valid"""
        if v and len(v) == 0:
            return None
        return v

# Fine-tuning Jobs -------------------------------------------------------------------------------   
class FineTuningJobState(str, Enum):
    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    PAUSED = "paused"  # Added paused state
 
class FineTuningJobType(str, Enum):
    SFT = "sft"
    LORA = "lora"
    QLORA = "qlora"
    IA3 = "ia3"
    PREFIX = "prefix"
    PROMPT = "prompt"  

class ResourceRequirements(BaseModel):
    """
    Resource requirements for fine-tuning jobs
    """
    gpu_type: Optional[str] = Field(None, description="GPU type")
    gpu_count: int = Field(1, description="Number of GPUs", ge=1)
    cpu_count: int = Field(4, description="Number of CPUs", ge=1)
    memory_gb: int = Field(16, description="Memory in GB", ge=1)
    disk_gb: int = Field(100, description="Disk space in GB", ge=10)
    
class FineTuningJobMetrics(BaseModel):
    """
    Metrics collected during fine-tuning job
    """
    training_loss: List[float] = Field([], description="Training loss history")
    validation_loss: Optional[List[float]] = Field(None, description="Validation loss history")
    learning_rate: List[float] = Field([], description="Learning rate history")
    gpu_utilization: Optional[List[float]] = Field(None, description="GPU utilization history")
    memory_utilization: Optional[List[float]] = Field(None, description="Memory utilization history")
    training_speed: Optional[List[float]] = Field(None, description="Training examples per second")
    elapsed_time: Optional[float] = Field(None, description="Elapsed time in seconds")
    checkpoint_sizes: Optional[List[int]] = Field(None, description="Checkpoint sizes in bytes")

class FineTuningJob(BaseEntity):
    """
    FineTuningJob class
    This class is used to store the information of a fine-tuning job.
    """
    started_at: Optional[datetime] = Field(None, description="Start date")
    completed_at: Optional[datetime] = Field(None, description="Completion date")
    status: str = Field(FineTuningJobState.PENDING, description="Job status") 
    base_model: str =  Field(..., description="Base model")
    dataset_id: str = Field(..., description="Dataset path")

    # Datasets
    train_dataset: Optional[str] = Field(None, description="Train dataset path")
    val_dataset: Optional[str] = Field(None, description="Validation dataset path")
    test_dataset: Optional[str] = Field(None, description="Test dataset path")

    hyperparameters: Hyperparameters = Field(None, description="Hyperparameters")
    result_files: List[str] = Field([], description="Result files")
    logs_path: Optional[str] = Field(None, description = "Logs path")

    job_type: str = Field(FineTuningJobType.SFT, description="Job type")
    
    resources: ResourceRequirements = Field(ResourceRequirements(), description="Resource requirements")
    metrics: Optional[FineTuningJobMetrics] = Field(None, description="Job metrics")

    result_model_id: Optional[str] = Field(None, description="Result model ID")
    result_adapter_ids: Optional[List[str]] = Field(None, description="Result adapter IDs")

    evaluation_metrics: Optional[Dict[str, Any]] = Field(None, description="Evaluation metrics")

    checkpoint: Optional[str] = Field(None, description="Checkpoint")

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric during training"""
        if not self.metrics:
            self.metrics = FineTuningJobMetrics()
        
        metric_mapping = {
            "training_loss": lambda: self.metrics.training_loss.append(value),
            "validation_loss": lambda: self._append_to_optional_list('validation_loss', value),
            "learning_rate": lambda: self.metrics.learning_rate.append(value),
            "gpu_utilization": lambda: self._append_to_optional_list('gpu_utilization', value),
            "memory_utilization": lambda: self._append_to_optional_list('memory_utilization', value),
            "training_speed": lambda: self._append_to_optional_list('training_speed', value),
        }
        
        if name in metric_mapping:
            metric_mapping[name]()
    
    def _append_to_optional_list(self, list_name: str, value: float) -> None:
        """Helper method to append to optional lists"""
        if not getattr(self.metrics, list_name):
            setattr(self.metrics, list_name, [])
        getattr(self.metrics, list_name).append(value)
    
    def pause(self) -> bool:
        """Pause the job"""
        if self.status == FineTuningJobState.RUNNING:
            self.status = FineTuningJobState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """Resume the job"""
        if self.status == FineTuningJobState.PAUSED:
            self.status = FineTuningJobState.RUNNING
            return True
        return False
    
    def cancel(self) -> bool:
        """Cancel the job"""
        if self.status in [FineTuningJobState.RUNNING, FineTuningJobState.PAUSED, 
                    FineTuningJobState.QUEUED, FineTuningJobState.PENDING]:
            self.status = FineTuningJobState.CANCELED
            return True
        return False