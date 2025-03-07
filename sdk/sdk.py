import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from loguru import logger

from managers import FineTuningJobManager, HubManager, CheckpointManager
from entities import (
    FineTuningJob, Model, Adapter, Hub, Checkpoint, 
    Metrics, Event, File, ModelType, ModelOrigin,
    FineTuningMethod, AdapterType, Hyperparameters
)
from utils import generate_uuid

# Configure logger
logger.add("logs/finetuning_{time:YYYY-MM-DD}.log", rotation="00:00", level="INFO")
logger.add("logs/finetuning_errors.log", level="ERROR")

class SDK:
    """
    Main SDK for fine-tuning models and managing the model hub.
    
    This SDK provides a unified interface to interact with all components
    of the fine-tuning system according to the specification in spec.md.
    
    It supports:
    - Creation and management of fine-tuning jobs
    - Management of models and adapters in the hub
    - Tracking of checkpoints during training
    """   
    def __init__(self, organization_id: str = None):
        """
        Initialize the SDK with necessary resource managers.
        
        Args:
            organization_id: Organization ID to use for all created resources
        """
        self.organization_id = organization_id or "default_org"
        
        # Initialize managers
        self.hub = HubManager(
            organization_id=self.organization_id,
            name="model_hub", 
            path=str(Path("fine_tuning_data/hub"))
        )
        self.fine_tuning_jobs = FineTuningJobManager(organization_id=self.organization_id)
        self.checkpoints = CheckpointManager()
        
        # Ensure directories exist
        os.makedirs("fine_tuning_data/models", exist_ok=True)
        os.makedirs("fine_tuning_data/adapters", exist_ok=True)
        os.makedirs("fine_tuning_data/checkpoints", exist_ok=True)
        
        logger.info(f"SDK initialized for organization {self.organization_id}")

    # Hub Models ------------------------------------------------------------------------------------
    def list_hub_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all models in the hub
        
        Returns:
            List[Dict[str, Any]]: List of model summaries
        """
        return self.hub.list_models()
        
    def get_hub_model(self, model_id_or_name: str) -> Optional[Model]:
        """
        Get a model from the hub by ID or name
        
        Args:
            model_id_or_name: Model ID or name
            
        Returns:
            Model: Model object if found
            
        Raises:
            ValueError: If model not found
        """
        model = self.hub.get_model(model_id_or_name)
        if not model:
            raise ValueError(f"Model '{model_id_or_name}' not found in hub")
        return model
    
    def create_hub_model(self, 
                       name: str, 
                       model_type: str, 
                       path: str,
                       description: Optional[str] = None,
                       base_model_id: Optional[str] = None) -> Model:
        """
        Register a new model in the hub
        
        Args:
            name: Name for the model
            model_type: Type of model (base or fine-tuned)
            path: Path to model files
            description: Optional description
            base_model_id: Base model ID for fine-tuned models
            
        Returns:
            Model: The created model
            
        Raises:
            ValueError: If model with same name exists
            FileNotFoundError: If path doesn't exist
        """
        # Validate path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
            
        # Create model object
        model = Model(
            organization_id=self.organization_id,
            name=name,
            type=ModelType(model_type),
            origin=ModelOrigin.LOCAL,
            path=path,
            description=description,
            base_model_id=base_model_id,
            created_at=datetime.utcnow()
        )
        
        # Register in hub
        model_id = self.hub.create_model(model)
        if not model_id:
            raise ValueError(f"Failed to register model '{name}' in hub")
            
        logger.success(f"Model '{name}' registered in hub")
        return model
    
    def delete_hub_model(self, model_id_or_name: str) -> bool:
        """
        Remove a model from the hub
        
        Args:
            model_id_or_name: Model ID or name
            
        Returns:
            bool: True if deleted successfully
            
        Raises:
            ValueError: If model not found
        """
        if not self.hub.delete_model(model_id_or_name):
            raise ValueError(f"Failed to delete model '{model_id_or_name}' from hub")
            
        logger.success(f"Model '{model_id_or_name}' removed from hub")
        return True
        
    # Hub Adapters ----------------------------------------------------------------------------------
    def list_hub_adapters(self) -> List[Dict[str, Any]]:
        """
        Get a list of all adapters in the hub
        
        Returns:
            List[Dict[str, Any]]: List of adapter summaries
        """
        return self.hub.list_adapters()
        
    def get_hub_adapter(self, adapter_id_or_name: str) -> Optional[Adapter]:
        """
        Get an adapter from the hub by ID or name
        
        Args:
            adapter_id_or_name: Adapter ID or name
            
        Returns:
            Adapter: Adapter object if found
            
        Raises:
            ValueError: If adapter not found
        """
        adapter = self.hub.get_adapter(adapter_id_or_name)
        if not adapter:
            raise ValueError(f"Adapter '{adapter_id_or_name}' not found in hub")
        return adapter
    
    def create_hub_adapter(self, 
                         name: str,
                         adapter_type: str,
                         model_id: str,
                         dataset: str,
                         path: str,
                         description: Optional[str] = None,
                         hyperparameters: Optional[Dict[str, Any]] = None) -> Adapter:
        """
        Register a new adapter in the hub
        
        Args:
            name: Name for the adapter
            adapter_type: Type of adapter (lora or qlora)
            model_id: ID or name of the associated model
            dataset: Dataset used for training
            path: Path to adapter files
            description: Optional description
            hyperparameters: Optional training hyperparameters
            
        Returns:
            Adapter: The created adapter
            
        Raises:
            ValueError: If adapter with same name exists
            FileNotFoundError: If path doesn't exist
        """
        # Validate path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
            
        # Validate model exists
        model = self.hub.get_model(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found in hub")
            
        # Validate dataset exists
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"Dataset not found: {dataset}")
        
        # Create hyperparameters object if provided
        hp = Hyperparameters()
        if hyperparameters:
            hp = Hyperparameters(
                method=FineTuningMethod(adapter_type),
                **hyperparameters
            )
        else:
            hp = Hyperparameters(method=FineTuningMethod(adapter_type))
        
        # Create adapter object
        adapter = Adapter(
            organization_id=self.organization_id,
            name=name,
            adapter_type=AdapterType(adapter_type),
            model_id=model.id,
            dataset=dataset,
            path=path,
            hyperparameters=hp,
            created_at=datetime.utcnow()
        )
        
        # Register in hub
        adapter_id = self.hub.create_adapter(adapter)
        if not adapter_id:
            raise ValueError(f"Failed to register adapter '{name}' in hub")
            
        logger.success(f"Adapter '{name}' registered in hub")
        return adapter
    
    def delete_hub_adapter(self, adapter_id_or_name: str) -> bool:
        """
        Remove an adapter from the hub
        
        Args:
            adapter_id_or_name: Adapter ID or name
            
        Returns:
            bool: True if deleted successfully
            
        Raises:
            ValueError: If adapter not found
        """
        if not self.hub.delete_adapter(adapter_id_or_name):
            raise ValueError(f"Failed to delete adapter '{adapter_id_or_name}' from hub")
            
        logger.success(f"Adapter '{adapter_id_or_name}' removed from hub")
        return True
    
    # Fine-tuning Jobs -------------------------------------------------------------------------------
    def list_fine_tuning_jobs(self) -> List[Dict[str, Any]]:
        """
        List all fine-tuning jobs
        
        Returns:
            List[Dict[str, Any]]: List of job summaries
        """
        return self.fine_tuning_jobs.list()
        
    def get_fine_tuning_job(self, job_id_or_name: str) -> FineTuningJob:
        """
        Get a fine-tuning job by ID or name
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            FineTuningJob: The fine-tuning job
            
        Raises:
            ValueError: If job not found
        """
        job = self.fine_tuning_jobs.get(job_id_or_name)
        if not job:
            raise ValueError(f"Fine-tuning job '{job_id_or_name}' not found")
        return job
        
    def create_fine_tuning_job(self,
                             name: str,
                             base_model: str,
                             dataset: str,
                             val_dataset: Optional[str] = None,
                             job_type: str = "sft",
                             output_model_name: Optional[str] = None,
                             hyperparameters: Optional[Dict[str, Any]] = None) -> FineTuningJob:
        """
        Create a new fine-tuning job
        
        Args:
            name: Unique name for the job
            base_model: Name or path of base model to fine-tune
            dataset: Path to training dataset
            val_dataset: Optional path to validation dataset
            job_type: Type of fine-tuning (sft, lora, qlora)
            output_model_name: Optional name for output model
            hyperparameters: Optional hyperparameters for training
            
        Returns:
            FineTuningJob: The created fine-tuning job
            
        Raises:
            ValueError: If job with same name exists
            FileNotFoundError: If dataset doesn't exist
        """
        # Validate paths
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"Training dataset not found: {dataset}")
            
        if val_dataset and not os.path.exists(val_dataset):
            raise FileNotFoundError(f"Validation dataset not found: {val_dataset}")
            
        # Check if base_model is a path or name in hub
        if os.path.exists(base_model):
            base_model_path = base_model
        else:
            # Try to get from hub
            try:
                model = self.hub.get_model(base_model)
                base_model_path = model.get_full_path()
            except:
                raise ValueError(f"Base model '{base_model}' not found in hub or filesystem")
        
        # Create hyperparameters object
        hp = Hyperparameters(method=FineTuningMethod(job_type))
        if hyperparameters:
            # Update default hyperparameters with user values
            for key, value in hyperparameters.items():
                setattr(hp, key, value)
                
        # Output paths setup
        logs_path = f"logs/{name}"
        os.makedirs(logs_path, exist_ok=True)
        
        # Create output model name if not provided
        if not output_model_name:
            output_model_name = f"{name}-model"
            
        # Create job object
        job = FineTuningJob(
            name=name,
            organization_id=self.organization_id,
            base_model=base_model_path,
            dataset=dataset,
            val_dataset=val_dataset,
            hyperparameters=hp,
            logs_path=logs_path,
            description=f"Fine-tuning job for {output_model_name}"
        )
        
        # Register job
        job_id = self.fine_tuning_jobs.create(job)
        if not job_id:
            raise ValueError(f"Failed to create fine-tuning job '{name}'")
            
        logger.success(f"Fine-tuning job '{name}' created successfully")
        return job
        
    def start_fine_tuning_job(self, job_id_or_name: str) -> bool:
        """
        Start a fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if started successfully
            
        Raises:
            ValueError: If job not found or cannot be started
        """
        if not self.fine_tuning_jobs.start_job(job_id_or_name):
            raise ValueError(f"Failed to start fine-tuning job '{job_id_or_name}'")
            
        logger.success(f"Fine-tuning job '{job_id_or_name}' started")
        return True
        
    def pause_fine_tuning_job(self, job_id_or_name: str) -> bool:
        """
        Pause a running fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if paused successfully
            
        Raises:
            ValueError: If job not found or cannot be paused
        """
        if not self.fine_tuning_jobs.pause_job(job_id_or_name):
            raise ValueError(f"Failed to pause fine-tuning job '{job_id_or_name}'")
            
        logger.success(f"Fine-tuning job '{job_id_or_name}' paused")
        return True
        
    def resume_fine_tuning_job(self, job_id_or_name: str) -> bool:
        """
        Resume a paused fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if resumed successfully
            
        Raises:
            ValueError: If job not found or cannot be resumed
        """
        if not self.fine_tuning_jobs.resume_job(job_id_or_name):
            raise ValueError(f"Failed to resume fine-tuning job '{job_id_or_name}'")
            
        logger.success(f"Fine-tuning job '{job_id_or_name}' resumed")
        return True
        
    def cancel_fine_tuning_job(self, job_id_or_name: str) -> bool:
        """
        Cancel a fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if canceled successfully
            
        Raises:
            ValueError: If job not found or cannot be canceled
        """
        if not self.fine_tuning_jobs.cancel_job(job_id_or_name):
            raise ValueError(f"Failed to cancel fine-tuning job '{job_id_or_name}'")
            
        logger.success(f"Fine-tuning job '{job_id_or_name}' canceled")
        return True
        
    def delete_fine_tuning_job(self, job_id_or_name: str) -> bool:
        """
        Delete a fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if deleted successfully
            
        Raises:
            ValueError: If job not found
        """
        if not self.fine_tuning_jobs.delete(job_id_or_name):
            raise ValueError(f"Failed to delete fine-tuning job '{job_id_or_name}'")
            
        # Also delete associated checkpoints
        job = self.fine_tuning_jobs.get(job_id_or_name)
        if job:
            self.checkpoints.delete_by_job(job.id)
            
        logger.success(f"Fine-tuning job '{job_id_or_name}' deleted")
        return True
    
    # Checkpoints ------------------------------------------------------------------------------------
    def list_checkpoints(self, job_id_or_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List checkpoints, optionally filtered by job
        
        Args:
            job_id_or_name: Optional job ID or name to filter by
            
        Returns:
            List[Dict[str, Any]]: List of checkpoint summaries
        """
        if job_id_or_name:
            # Get job first
            job = self.fine_tuning_jobs.get(job_id_or_name)
            if not job:
                raise ValueError(f"Fine-tuning job '{job_id_or_name}' not found")
                
            # Get checkpoints for this job
            checkpoints = self.checkpoints.get_by_job(job.id)
            return [checkpoint.get_info() for checkpoint in checkpoints]
        else:
            # Return all checkpoints
            return [checkpoint.get_info() for checkpoint in self.checkpoints.checkpoints.values()]
            
    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """
        Get a checkpoint by ID
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint: The checkpoint
            
        Raises:
            ValueError: If checkpoint not found
        """
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")
        return checkpoint
        
    def get_latest_checkpoint(self, job_id_or_name: str) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint for a job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            Optional[Checkpoint]: Latest checkpoint if found
            
        Raises:
            ValueError: If job not found
        """
        # Get job first
        job = self.fine_tuning_jobs.get(job_id_or_name)
        if not job:
            raise ValueError(f"Fine-tuning job '{job_id_or_name}' not found")
            
        # Get latest checkpoint
        return self.checkpoints.get_latest_by_job(job.id)
        
    def create_checkpoint(self, 
                        job_id_or_name: str, 
                        step_number: int, 
                        path: str,
                        metrics: Optional[Dict[str, float]] = None) -> Checkpoint:
        """
        Create a new checkpoint for a job
        
        Args:
            job_id_or_name: Job ID or name
            step_number: Training step number
            path: Path to checkpoint files
            metrics: Optional metrics (train_loss, valid_loss)
            
        Returns:
            Checkpoint: The created checkpoint
            
        Raises:
            ValueError: If job not found
            FileNotFoundError: If path doesn't exist
        """
        # Validate path exists
        if not os.path.exists(path):
            # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
        # Get job first
        job = self.fine_tuning_jobs.get(job_id_or_name)
        if not job:
            raise ValueError(f"Fine-tuning job '{job_id_or_name}' not found")
            
        # Create metrics object
        checkpoint_metrics = Metrics()
        if metrics:
            if 'train_loss' in metrics:
                checkpoint_metrics.train_loss = metrics['train_loss']
            if 'valid_loss' in metrics:
                checkpoint_metrics.valid_loss = metrics['valid_loss']
                
        # Create checkpoint
        checkpoint = Checkpoint(
            job_id=job.id,
            step_number=step_number,
            path=path,
            metrics=checkpoint_metrics,
            created_at=datetime.utcnow()
        )
        
        # Register checkpoint
        checkpoint_id = self.checkpoints.create(checkpoint)
        
        # Add to job
        job.add_checkpoint(checkpoint)
        self.fine_tuning_jobs.update(job)
        
        logger.success(f"Checkpoint created for job '{job.name}' at step {step_number}")
        return checkpoint
        
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            bool: True if deleted successfully
            
        Raises:
            ValueError: If checkpoint not found
        """
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")
            
        # Get the job and update it
        job = self.fine_tuning_jobs.get(checkpoint.job_id)
        if job:
            # Update job checkpoints list
            job.checkpoints = [c for c in job.checkpoints if c != checkpoint_id]
            self.fine_tuning_jobs.update(job)
            
        # Delete checkpoint
        if not self.checkpoints.delete(checkpoint_id):
            raise ValueError(f"Failed to delete checkpoint '{checkpoint_id}'")
            
        logger.success(f"Checkpoint '{checkpoint_id}' deleted")
        return True
        
    # Utility Methods ----------------------------------------------------------------------------------
    def register_fine_tuning_results(self, 
                                   job_id_or_name: str, 
                                   model_name: Optional[str] = None,
                                   adapter_name: Optional[str] = None) -> Dict[str, str]:
        """
        Register fine-tuning results (model and/or adapter) in the hub
        
        Args:
            job_id_or_name: Job ID or name
            model_name: Optional name for the model
            adapter_name: Optional name for the adapter
            
        Returns:
            Dict[str, str]: Dictionary with model_id and/or adapter_id
            
        Raises:
            ValueError: If job not found or not completed
        """
        # Get job
        job = self.fine_tuning_jobs.get(job_id_or_name)
        if not job:
            raise ValueError(f"Fine-tuning job '{job_id_or_name}' not found")
            
        # Check if job is completed
        if not job.is_complete or job.status != 'succeeded':
            raise ValueError(f"Fine-tuning job '{job_id_or_name}' is not completed successfully")
            
        results = {}
        
        # Create model if needed
        if model_name:
            # Create model directory in hub
            model_path = f"fine_tuning_data/models/{model_name}"
            os.makedirs(model_path, exist_ok=True)
            
            # Create model in hub
            model = Model(
                organization_id=self.organization_id,
                name=model_name,
                type=ModelType.FINE_TUNED,
                origin=ModelOrigin.LOCAL,
                path=model_path,
                description=f"Fine-tuned model from job {job.name}",
                base_model_id=job.base_model,
                hyperparameters=job.hyperparameters,
                created_at=datetime.utcnow()
            )
            
            # Add result files to model
            for result_file in job.result_files:
                if os.path.exists(result_file):
                    # Copy or link the file to model directory
                    pass
                    
            # Register in hub
            model_id = self.hub.create_model(model)
            if model_id:
                results["model_id"] = model_id
                job.output_model_id = model_id
                self.fine_tuning_jobs.update(job)
        
        # Create adapter if job used PEFT (LoRA/QLoRA)
        if adapter_name and job.is_peft:
            # Create adapter directory in hub
            adapter_path = f"fine_tuning_data/adapters/{adapter_name}"
            os.makedirs(adapter_path, exist_ok=True)
            
            # Create adapter in hub
            adapter = Adapter(
                organization_id=self.organization_id,
                name=adapter_name,
                adapter_type=AdapterType(job.method),
                model_id=job.base_model,
                dataset=job.dataset,
                path=adapter_path,
                hyperparameters=job.hyperparameters,
                created_at=datetime.utcnow()
            )
            
            # Add result files to adapter
            for result_file in job.result_files:
                if os.path.exists(result_file) and "adapter" in result_file.lower():
                    # Copy or link the file to adapter directory
                    pass
                    
            # Register in hub
            adapter_id = self.hub.create_adapter(adapter)
            if adapter_id:
                results["adapter_id"] = adapter_id
                job.output_adapter_id = adapter_id
                self.fine_tuning_jobs.update(job)
        
        return results


