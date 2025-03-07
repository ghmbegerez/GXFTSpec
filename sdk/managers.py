import json 
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

from entities import (
    FineTuningJob, Model, Adapter, Hub, Checkpoint, 
    Metrics, Event, File, ModelType, ModelOrigin
)
from utils import generate_uuid, Logs, Events

# Fine-tuning jobs -----------------------------------------------------------------------------
class FineTuningJobManager:
    """
    Manager for fine-tuning jobs
    """
    def __init__(self, organization_id: str):
        """
        Initialize a new fine-tuning job manager
        
        Args:
            organization_id: Organization identifier
        """
        self.organization_id = organization_id
        self.jobs: Dict[str, FineTuningJob] = {}
        self.name_to_id_map: Dict[str, str] = {}  # Maps names to IDs for easier lookup
        logger.info(f"FineTuningJob manager initialized for organization {organization_id}")

    def create(self, job: FineTuningJob) -> str:
        """
        Create a new fine-tuning job
        
        Args:
            job: FineTuningJob to create
            
        Returns:
            str: Job ID
        """
        if job.name in self.name_to_id_map:
            logger.warning(f"Job with name {job.name} already exists")
            return None
            
        # Ensure the job has the correct organization ID
        if job.organization_id != self.organization_id:
            job.organization_id = self.organization_id
            
        self.jobs[job.id] = job
        self.name_to_id_map[job.name] = job.id
        
        logger.info(f"Created fine-tuning job: {job.name} (ID: {job.id})")
        Events.notify(job.id, "Fine-tuning job created")
        return job.id

    def get(self, job_id_or_name: str) -> Optional[FineTuningJob]:
        """
        Get a fine-tuning job by ID or name
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            Optional[FineTuningJob]: Job if found, None otherwise
        """
        # Try to get by ID first
        if job_id_or_name in self.jobs:
            logger.debug(f"Found job by ID: {job_id_or_name}")
            return self.jobs[job_id_or_name]
            
        # Then try by name
        elif job_id_or_name in self.name_to_id_map:
            job_id = self.name_to_id_map[job_id_or_name]
            logger.debug(f"Found job by name: {job_id_or_name} (ID: {job_id})")
            return self.jobs[job_id]
            
        logger.warning(f"Job not found: {job_id_or_name}")
        return None
    
    def list(self) -> List[Dict[str, Any]]:
        """
        List all fine-tuning jobs
        
        Returns:
            List[Dict[str, Any]]: List of job summaries
        """
        return [job.get_summary() for job in self.jobs.values()]
    
    def delete(self, job_id_or_name: str) -> bool:
        """
        Delete a fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if deleted, False otherwise
        """
        job_id = job_id_or_name
        
        # If name provided, get the ID
        if job_id_or_name in self.name_to_id_map:
            job_id = self.name_to_id_map[job_id_or_name]
            job_name = job_id_or_name
            del self.name_to_id_map[job_id_or_name]
        elif job_id in self.jobs:
            job_name = self.jobs[job_id].name
        else:
            logger.warning(f"Job not found for deletion: {job_id_or_name}")
            return False
        
        job = self.jobs.pop(job_id, None)
        if job:
            logger.info(f"Deleted fine-tuning job: {job_name} (ID: {job_id})")
            Events.notify(job_id, "Fine-tuning job deleted")
            return True
            
        return False
        
    def contains(self, job_id_or_name: str) -> bool:
        """
        Check if a fine-tuning job exists
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if job exists, False otherwise
        """
        exists = (job_id_or_name in self.jobs or job_id_or_name in self.name_to_id_map)
        if exists:
            logger.debug(f"Job exists: {job_id_or_name}")
        else:
            logger.debug(f"Job does not exist: {job_id_or_name}")
        return exists
    
    def update(self, job: FineTuningJob) -> bool:
        """
        Update an existing fine-tuning job
        
        Args:
            job: Updated FineTuningJob
            
        Returns:
            bool: True if updated, False otherwise
        """
        if job.id in self.jobs:
            # Update the job
            old_job = self.jobs[job.id]
            old_name = old_job.name
            
            # If name changed, update the name map
            if old_name != job.name:
                if job.name in self.name_to_id_map and self.name_to_id_map[job.name] != job.id:
                    logger.warning(f"Cannot rename job to {job.name} - name already in use")
                    return False
                    
                if old_name in self.name_to_id_map:
                    del self.name_to_id_map[old_name]
                    
                self.name_to_id_map[job.name] = job.id
                logger.info(f"Job renamed from {old_name} to {job.name}")
            
            self.jobs[job.id] = job
            logger.info(f"Updated fine-tuning job: {job.name} (ID: {job.id})")
            Events.notify(job.id, "Fine-tuning job updated")
            return True
        
        logger.warning(f"Cannot update job {job.id} - not found")
        return False
        
    def start_job(self, job_id_or_name: str) -> bool:
        """
        Start a fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if started, False otherwise
        """
        job = self.get(job_id_or_name)
        if job:
            if job.start():
                self.update(job)
                return True
        return False
        
    def pause_job(self, job_id_or_name: str) -> bool:
        """
        Pause a fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if paused, False otherwise
        """
        job = self.get(job_id_or_name)
        if job:
            if job.pause():
                self.update(job)
                return True
        return False
        
    def resume_job(self, job_id_or_name: str) -> bool:
        """
        Resume a paused fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if resumed, False otherwise
        """
        job = self.get(job_id_or_name)
        if job:
            if job.resume():
                self.update(job)
                return True
        return False
        
    def cancel_job(self, job_id_or_name: str) -> bool:
        """
        Cancel a fine-tuning job
        
        Args:
            job_id_or_name: Job ID or name
            
        Returns:
            bool: True if canceled, False otherwise
        """
        job = self.get(job_id_or_name)
        if job:
            if job.cancel():
                self.update(job)
                return True
        return False

# Hub ------------------------------------------------------------------------------------    
# Hub manager to work with Hub entities
class HubManager:
    """
    Manager for Hub entities that store models and adapters
    This class wraps the Hub entity to provide additional functionality
    """
    def __init__(self, organization_id: str, name: str = "default", path: str = "hub"):
        """
        Initialize a new Hub manager
        
        Args:
            organization_id: Organization identifier
            name: Hub name
            path: Path to store hub files
        """
        self.hub = Hub(
            organization_id=organization_id,
            name=name,
            path=path
        )
        logger.info(f"Hub manager initialized for organization {organization_id}")
        
    def create_model(self, model: Model) -> str:
        """
        Add a model to the hub
        
        Args:
            model: Model to add
            
        Returns:
            str: Model ID
        """
        if self.hub.register_model(model):
            logger.info(f"Model {model.name} registered in hub {self.hub.name}")
            Events.notify(model.id, "Model registered in hub")
            return model.id
        else:
            logger.warning(f"Failed to register model {model.name} in hub {self.hub.name}")
            return None

    def get_model(self, model_id_or_name: str) -> Optional[Model]:
        """
        Get a model from the hub
        
        Args:
            model_id_or_name: Model ID or name
            
        Returns:
            Optional[Model]: Model if found, None otherwise
        """
        # Try to get by ID first
        model = self.hub.get_model_by_id(model_id_or_name)
        if model:
            return model
            
        # Then try by name
        return self.hub.get_model_by_name(model_id_or_name)
    
    def list_models(self) -> List[Dict]:
        """
        List all models in the hub
        
        Returns:
            List[Dict]: List of model summaries
        """
        return self.hub.list_models()
    
    def delete_model(self, model_id_or_name: str) -> bool:
        """
        Remove a model from the hub
        
        Args:
            model_id_or_name: Model ID or name
            
        Returns:
            bool: True if deleted, False otherwise
        """
        # First try by ID
        model_id = model_id_or_name
        model = self.hub.get_model_by_id(model_id_or_name)
        
        # If not found by ID, try by name
        if not model and model_id_or_name in self.hub.models_by_name:
            model_id = self.hub.models_by_name[model_id_or_name]
            model = self.hub.models.get(model_id)
        
        if model and self.hub.remove_model(model_id):
            logger.info(f"Model {model.name} removed from hub {self.hub.name}")
            Events.notify(model_id, "Model removed from hub")
            return True
            
        logger.warning(f"Failed to remove model {model_id_or_name} from hub {self.hub.name}")
        return False

    def contains_model(self, model_id_or_name: str) -> bool:
        """
        Check if a model exists in the hub
        
        Args:
            model_id_or_name: Model ID or name
            
        Returns:
            bool: True if model exists, False otherwise
        """
        return (model_id_or_name in self.hub.models or 
                model_id_or_name in self.hub.models_by_name)

    def create_adapter(self, adapter: Adapter) -> str:
        """
        Add an adapter to the hub
        
        Args:
            adapter: Adapter to add
            
        Returns:
            str: Adapter ID
        """
        if self.hub.register_adapter(adapter):
            logger.info(f"Adapter {adapter.name} registered in hub {self.hub.name}")
            Events.notify(adapter.id, "Adapter registered in hub")
            return adapter.id
        else:
            logger.warning(f"Failed to register adapter {adapter.name} in hub {self.hub.name}")
            return None

    def get_adapter(self, adapter_id_or_name: str) -> Optional[Adapter]:
        """
        Get an adapter from the hub
        
        Args:
            adapter_id_or_name: Adapter ID or name
            
        Returns:
            Optional[Adapter]: Adapter if found, None otherwise
        """
        # Try to get by ID first
        adapter = self.hub.get_adapter_by_id(adapter_id_or_name)
        if adapter:
            return adapter
            
        # Then try by name
        return self.hub.get_adapter_by_name(adapter_id_or_name)
    
    def list_adapters(self) -> List[Dict]:
        """
        List all adapters in the hub
        
        Returns:
            List[Dict]: List of adapter summaries
        """
        return self.hub.list_adapters()
    
    def delete_adapter(self, adapter_id_or_name: str) -> bool:
        """
        Remove an adapter from the hub
        
        Args:
            adapter_id_or_name: Adapter ID or name
            
        Returns:
            bool: True if deleted, False otherwise
        """
        # First try by ID
        adapter_id = adapter_id_or_name
        adapter = self.hub.get_adapter_by_id(adapter_id_or_name)
        
        # If not found by ID, try by name
        if not adapter and adapter_id_or_name in self.hub.adapters_by_name:
            adapter_id = self.hub.adapters_by_name[adapter_id_or_name]
            adapter = self.hub.adapters.get(adapter_id)
        
        if adapter and self.hub.remove_adapter(adapter_id):
            logger.info(f"Adapter {adapter.name} removed from hub {self.hub.name}")
            Events.notify(adapter_id, "Adapter removed from hub")
            return True
            
        logger.warning(f"Failed to remove adapter {adapter_id_or_name} from hub {self.hub.name}")
        return False
    
    def contains_adapter(self, adapter_id_or_name: str) -> bool:
        """
        Check if an adapter exists in the hub
        
        Args:
            adapter_id_or_name: Adapter ID or name
            
        Returns:
            bool: True if adapter exists, False otherwise
        """
        return (adapter_id_or_name in self.hub.adapters or 
                adapter_id_or_name in self.hub.adapters_by_name)


# Checkpoint Manager --------------------------------------------------------------------
class CheckpointManager:
    """
    Manager for checkpoints from fine-tuning jobs
    """
    def __init__(self):
        """Initialize a new checkpoint manager"""
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.checkpoints_by_job: Dict[str, List[str]] = {}  # Map job IDs to checkpoint IDs
        logger.info("Checkpoint manager initialized")
        
    def create(self, checkpoint: Checkpoint) -> str:
        """
        Create a new checkpoint
        
        Args:
            checkpoint: Checkpoint to create
            
        Returns:
            str: Checkpoint ID
        """
        # Add to main dictionary
        self.checkpoints[checkpoint.id] = checkpoint
        
        # Add to job index
        if checkpoint.job_id not in self.checkpoints_by_job:
            self.checkpoints_by_job[checkpoint.job_id] = []
        self.checkpoints_by_job[checkpoint.job_id].append(checkpoint.id)
        
        logger.info(f"Created checkpoint {checkpoint.id} for job {checkpoint.job_id} at step {checkpoint.step_number}")
        return checkpoint.id
    
    def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Get a checkpoint by ID
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Optional[Checkpoint]: Checkpoint if found, None otherwise
        """
        if checkpoint_id in self.checkpoints:
            return self.checkpoints[checkpoint_id]
        logger.warning(f"Checkpoint not found: {checkpoint_id}")
        return None
    
    def get_by_job(self, job_id: str) -> List[Checkpoint]:
        """
        Get all checkpoints for a job
        
        Args:
            job_id: Job ID
            
        Returns:
            List[Checkpoint]: List of checkpoints
        """
        if job_id not in self.checkpoints_by_job:
            logger.warning(f"No checkpoints found for job: {job_id}")
            return []
            
        checkpoint_ids = self.checkpoints_by_job[job_id]
        return [self.checkpoints[checkpoint_id] for checkpoint_id in checkpoint_ids if checkpoint_id in self.checkpoints]
    
    def get_latest_by_job(self, job_id: str) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint for a job
        
        Args:
            job_id: Job ID
            
        Returns:
            Optional[Checkpoint]: Latest checkpoint if found, None otherwise
        """
        checkpoints = self.get_by_job(job_id)
        if not checkpoints:
            return None
            
        # Return the checkpoint with the highest step number
        return max(checkpoints, key=lambda c: c.step_number)
    
    def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            bool: True if deleted, False otherwise
        """
        if checkpoint_id not in self.checkpoints:
            logger.warning(f"Checkpoint not found for deletion: {checkpoint_id}")
            return False
            
        checkpoint = self.checkpoints[checkpoint_id]
        job_id = checkpoint.job_id
        
        # Remove from main dictionary
        del self.checkpoints[checkpoint_id]
        
        # Remove from job index
        if job_id in self.checkpoints_by_job:
            self.checkpoints_by_job[job_id] = [
                id for id in self.checkpoints_by_job[job_id] if id != checkpoint_id
            ]
            
        logger.info(f"Deleted checkpoint {checkpoint_id} for job {job_id}")
        return True
    
    def delete_by_job(self, job_id: str) -> int:
        """
        Delete all checkpoints for a job
        
        Args:
            job_id: Job ID
            
        Returns:
            int: Number of checkpoints deleted
        """
        if job_id not in self.checkpoints_by_job:
            logger.warning(f"No checkpoints found for job deletion: {job_id}")
            return 0
            
        checkpoint_ids = self.checkpoints_by_job[job_id].copy()
        count = 0
        
        for checkpoint_id in checkpoint_ids:
            if self.delete(checkpoint_id):
                count += 1
                
        logger.info(f"Deleted {count} checkpoints for job {job_id}")
        return count
    
