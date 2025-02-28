import json 
from entities import FineTuningJob, Model, Deployment, Dataset, Adapter
from utils import generate_uuid, Logs, Events
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Fine-tuning jobs -----------------------------------------------------------------------------
class FineTuningJobs:
    def __init__(self):
        self.jobs: Dict[str, FineTuningJob] = {}
        self.name_to_id_map: Dict[str, str] = {}  # Maps names to IDs for easier lookup

    def create(self, job: FineTuningJob):
        self.jobs[job.id] = job
        self.name_to_id_map[job.name] = job.id
        Logs.log(job.id, "Job created")
        Events.notify(job.id, "Job creation initiated")
        return job.id

    def get(self, job_id_or_name: str) -> Optional[FineTuningJob]:
        # Try to get by ID first
        if job_id_or_name in self.jobs:
            return self.jobs[job_id_or_name]
        # Then try by name
        elif job_id_or_name in self.name_to_id_map:
            return self.jobs[self.name_to_id_map[job_id_or_name]]
        return None
    
    def list(self) -> List[FineTuningJob]:
        return list(self.jobs.values())
    
    def delete(self, job_id_or_name: str) -> bool:
        job_id = job_id_or_name
        if job_id_or_name in self.name_to_id_map:
            job_id = self.name_to_id_map[job_id_or_name]
            del self.name_to_id_map[job_id_or_name]
        
        job = self.jobs.pop(job_id, None)
        if job:
            # Delete job info
            Logs.log(job_id, "Job deleted")
            Events.notify(job_id, "Job deletion initiated")
            return True
        return False
        
    def contains(self, job_id_or_name: str) -> bool:
        return (job_id_or_name in self.jobs or 
                job_id_or_name in self.name_to_id_map)
    
    def update(self, job: FineTuningJob) -> None:
        """Updates an existing job"""
        if job.id in self.jobs:
            self.jobs[job.id] = job
            self.name_to_id_map[job.name] = job.id
            Logs.log(job.id, "Job updated")
            return True
        return False

# Datasets ----------------------------------------------------------------------------------        
class Datasets:
    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}
        self.name_to_id_map: Dict[str, str] = {}  # Maps names to IDs for easier lookup
    
    def create(self, dataset: Dataset):
        self.datasets[dataset.id] = dataset
        self.name_to_id_map[dataset.name] = dataset.id
        Logs.log(dataset.id, "Dataset created")
        Events.notify(dataset.id, "Dataset creation initiated")
        return dataset.id
        
    def get(self, dataset_id_or_name: str) -> Optional[Dataset]:
        # Try to get by ID first
        if dataset_id_or_name in self.datasets:
            return self.datasets[dataset_id_or_name]
        # Then try by name
        elif dataset_id_or_name in self.name_to_id_map:
            return self.datasets[self.name_to_id_map[dataset_id_or_name]]
        return None
    
    def list(self) -> List[Dataset]:
        return list(self.datasets.values())
    
    def delete(self, dataset_id_or_name: str) -> bool:
        dataset_id = dataset_id_or_name
        if dataset_id_or_name in self.name_to_id_map:
            dataset_id = self.name_to_id_map[dataset_id_or_name]
            del self.name_to_id_map[dataset_id_or_name]
        
        dataset = self.datasets.pop(dataset_id, None)
        if dataset:
            # Delete dataset info
            Logs.log(dataset_id, "Dataset deleted")
            Events.notify(dataset_id, "Dataset deletion initiated")
            return True
        return False
        
    def contains(self, dataset_id_or_name: str) -> bool:
        return (dataset_id_or_name in self.datasets or 
                dataset_id_or_name in self.name_to_id_map)

# Models ------------------------------------------------------------------------------------    
class Models:
    def __init__(self):
        self.models: Dict[str, Model] = {}
        self.name_to_id_map: Dict[str, str] = {}  # Maps names to IDs for easier lookup

    def create(self, model: Model):
        self.models[model.id] = model
        self.name_to_id_map[model.name] = model.id
        Logs.log(model.id, "Model created")
        Events.notify(model.id, "Model creation initiated")
        return model.id

    def get(self, model_id_or_name: str) -> Optional[Model]:
        # Try to get by ID first
        if model_id_or_name in self.models:
            return self.models[model_id_or_name]
        # Then try by name
        elif model_id_or_name in self.name_to_id_map:
            return self.models[self.name_to_id_map[model_id_or_name]]
        return None
    
    def list(self) -> List[Model]:
        return list(self.models.values())
    
    def delete(self, model_id_or_name: str) -> bool:
        model_id = model_id_or_name
        if model_id_or_name in self.name_to_id_map:
            model_id = self.name_to_id_map[model_id_or_name]
            del self.name_to_id_map[model_id_or_name]
        
        model = self.models.pop(model_id, None)
        if model:
            # Delete model info
            Logs.log(model_id, "Model deleted") 
            Events.notify(model_id, "Model deletion initiated")
            return True
        return False

    def contains(self, model_id_or_name: str) -> bool:
        return (model_id_or_name in self.models or 
                model_id_or_name in self.name_to_id_map)

# Adapters ----------------------------------------------------------------------------------    
class Adapters:
    def __init__(self):
        self.adapters: Dict[str, Adapter] = {}
        self.name_to_id_map: Dict[str, str] = {}  # Maps names to IDs for easier lookup

    def create(self, adapter: Adapter):
        self.adapters[adapter.id] = adapter
        self.name_to_id_map[adapter.name] = adapter.id
        Logs.log(adapter.id, "Adapter created")
        Events.notify(adapter.id, "Adapter creation initiated")
        return adapter.id

    def get(self, adapter_id_or_name: str) -> Optional[Adapter]:
        # Try to get by ID first
        if adapter_id_or_name in self.adapters:
            return self.adapters[adapter_id_or_name]
        # Then try by name
        elif adapter_id_or_name in self.name_to_id_map:
            return self.adapters[self.name_to_id_map[adapter_id_or_name]]
        return None
    
    def list(self) -> List[Adapter]:
        return list(self.adapters.values())
    
    def delete(self, adapter_id_or_name: str) -> bool:
        adapter_id = adapter_id_or_name
        if adapter_id_or_name in self.name_to_id_map:
            adapter_id = self.name_to_id_map[adapter_id_or_name]
            del self.name_to_id_map[adapter_id_or_name]
        
        adapter = self.adapters.pop(adapter_id, None)
        if adapter:
            # Delete adapter info
            Logs.log(adapter_id, "Adapter deleted") 
            Events.notify(adapter_id, "Adapter deletion initiated")
            return True
        return False
    
    def contains(self, adapter_id_or_name: str) -> bool:
        return (adapter_id_or_name in self.adapters or 
                adapter_id_or_name in self.name_to_id_map)
    
# Deployments --------------------------------------------------------------------------------
class Deployments:
    def __init__(self):
        self.deployments: Dict[str, Deployment] = {}
        self.name_to_id_map: Dict[str, str] = {}  # Maps names to IDs for easier lookup

    def create(self, deployment: Deployment):
        self.deployments[deployment.id] = deployment
        self.name_to_id_map[deployment.name] = deployment.id
        Logs.log(deployment.id, "Deployment created")
        Events.notify(deployment.id, "Deployment creation initiated")
        return deployment.id

    def deploy(self, deployment_id_or_name: str) -> bool:
        """
        Start the deployment process for a created deployment
        """
        deployment_id = deployment_id_or_name
        if deployment_id_or_name in self.name_to_id_map:
            deployment_id = self.name_to_id_map[deployment_id_or_name]
        
        if deployment_id in self.deployments:
            deployment = self.deployments[deployment_id]
            deployment.status = "deploying"  # Update status
            Logs.log(deployment_id, f"Deploying...")
            Events.notify(deployment_id, "Deployment process started")
            return True
        return False

    def undeploy(self, deployment_id_or_name: str) -> bool:
        """
        Undeploy a deployed model
        """
        deployment_id = deployment_id_or_name
        if deployment_id_or_name in self.name_to_id_map:
            deployment_id = self.name_to_id_map[deployment_id_or_name]
            
        if deployment_id in self.deployments:
            deployment = self.deployments[deployment_id]
            deployment.status = "stopping"  # Update status
            Logs.log(deployment_id, "Undeploying")
            Events.notify(deployment_id, "Undeployment process started")
            return True
        return False

    def list(self) -> List[Deployment]:
        return list(self.deployments.values())
    
    def get(self, deployment_id_or_name: str) -> Optional[Deployment]:
        # Try to get by ID first
        if deployment_id_or_name in self.deployments:
            return self.deployments[deployment_id_or_name]
        # Then try by name
        elif deployment_id_or_name in self.name_to_id_map:
            return self.deployments[self.name_to_id_map[deployment_id_or_name]]
        return None
    
    def delete(self, deployment_id_or_name: str) -> bool:
        deployment_id = deployment_id_or_name
        if deployment_id_or_name in self.name_to_id_map:
            deployment_id = self.name_to_id_map[deployment_id_or_name]
            del self.name_to_id_map[deployment_id_or_name]
        
        deployment = self.deployments.pop(deployment_id, None)
        if deployment:
            # Delete deployment info
            Logs.log(deployment_id, "Deployment deleted")
            Events.notify(deployment_id, "Deployment deletion initiated")
            return True
        return False
        
    def contains(self, deployment_id_or_name: str) -> bool:
        return (deployment_id_or_name in self.deployments or 
                deployment_id_or_name in self.name_to_id_map)