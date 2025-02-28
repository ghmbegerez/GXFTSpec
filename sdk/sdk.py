from managers import FineTuningJobs, Models, Adapters, Datasets, Deployments
from entities import (
    FineTuningJob, FineTuningJobType, Model, Adapter, Dataset, Deployment,
    DatasetFormat, Hyperparameters, DeploymentStatus, DeploymentEnvironment
)
from typing import Dict, List, Optional, Any, Union
import os
from loguru import logger
from utils import generate_uuid

class SDK:
    """
    Main SDK for managing datasets, models, adapters, fine-tuning jobs and deployments.
    
    This SDK provides a unified interface to interact with all components
    of the model training and deployment system.
    """   
    def __init__(self, organization_id: str = None, project_id: str = None):
        """
        Initializes the SDK with the necessary resource managers.
        
        Args:
            organization_id (str, optional): Organization ID to use for all created resources.
            project_id (str, optional): Project ID to use for all created resources.
        """
        self.organization_id = organization_id or "default_org"
        self.project_id = project_id or "default_project"
        self.datasets = Datasets()
        self.models = Models()
        self.adapters = Adapters() 
        self.fine_tuning_jobs = FineTuningJobs()
        self.deployments = Deployments() 
        logger.info("SDK successfully initialized")

    # Datasets ---------------------------------------------------------------------------------------                        
    def list_datasets(self) -> List[Dataset]:
        """
        Gets a list of all available datasets.
        
        Returns:
            List[Dataset]: List of available Dataset objects.
        """
        return self.datasets.list()

    def create_dataset(self, 
                    name: str, 
                    path: str, 
                    format: Union[Dict[str,Any],DatasetFormat]) -> Dataset:
        """
        Creates a new dataset from a file or directory.
        
        Args:
            name (str): Unique name for the dataset.
            path (str): Path to the file or directory containing the data.
            format (Union[Dict[str, Any], DatasetFormat]): Format specification for the dataset.
            
        Returns:
            Dataset: The created Dataset object.
            
        Raises:
            ValueError: If a dataset with the same name already exists.
            FileNotFoundError: If the specified path doesn't exist.
        """        
        self._validate_resource_doesnt_exist(self.datasets, name, "Dataset")
        self._validate_path_exists(path)
        
        # Convert dict format to DatasetFormat if necessary
        if isinstance(format, dict):
            format = DatasetFormat(**format)
        
        dataset = Dataset(
            id=generate_uuid(),
            organization_id=self.organization_id,
            project_id=self.project_id,
            name=name,
            path=path,
            format=format
        )
        self.datasets.create(dataset)
        logger.success(f"Dataset '{name}' created successfully.")
        return dataset
    
    def get_dataset(self, name: str) -> Dataset:
        """
        Gets a dataset by its name.
        
        Args:
            name (str): Name of the dataset.
            
        Returns:
            Dataset: The corresponding Dataset object.
            
        Raises:
            ValueError: If the dataset doesn't exist.
        """
        return self._get_resource(self.datasets, name, "Dataset")

    def delete_dataset(self, name: str) -> None:
        """
        Deletes an existing dataset.
        
        Args:
            name (str): Name of the dataset to delete.
            
        Raises:
            ValueError: If the dataset doesn't exist.
        """        
        self._validate_resource_exists(self.datasets, name, "Dataset")
        self.datasets.delete(name)
        logger.success(f"Dataset '{name}' deleted successfully.")
        
    # Models -----------------------------------------------------------------------------------------
    def list_models(self) -> List[Model]:
        """
        Gets a list of all available models.
        
        Returns:
            List[Model]: List of available Model objects.
        """        
        return self.models.list()

    def create_model(self, 
                    name: str, 
                    path: str, 
                    dataset_id:str,
                    base_model: Optional[str] = None,
                    adapters: Optional[List[str]] = None,
                    hyperparameters: Optional[Union[Dict[str,Any],Hyperparameters]] = None) -> Model:
        """
        Creates a new model from a file or directory.
        
        Args:
            name (str): Unique name for the model.
            path (str): Path to the file or directory containing the model.
            dataset_id (str): ID or name of the dataset to use for training.
            base_model (Optional[str]): Name or ID of the base model to use.
            adapters (Optional[List[str]]): List of adapter names or IDs to apply.
            hyperparameters (Optional[Union[Dict[str, Any], Hyperparameters]]): Hyperparameters for the model.
            
        Returns:
            Model: The created Model object.
            
        Raises:
            ValueError: If a model with the same name already exists.
            FileNotFoundError: If the specified path doesn't exist.
        """        
        self._validate_resource_doesnt_exist(self.models, name, "Model")
        self._validate_path_exists(path)
        
        # Get dataset object to extract its ID
        dataset = self._get_resource(self.datasets, dataset_id, "Dataset")
        
        # Validate adapters if provided
        adapter_ids = []
        if adapters:
            for adapter_name in adapters:
                adapter = self._get_resource(self.adapters, adapter_name, "Adapter")
                adapter_ids.append(adapter.id)
        
        # Convert dict hyperparameters to Hyperparameters if necessary
        if isinstance(hyperparameters, dict):
            hyperparameters = Hyperparameters(**hyperparameters)
        elif hyperparameters is None:
            hyperparameters = Hyperparameters()
        
        model = Model(
            id=generate_uuid(),
            organization_id=self.organization_id,
            project_id=self.project_id,
            name=name,
            path=path,  # Using 'path' instead of 'model_path'
            dataset_id=dataset.id,
            base_model=base_model,
            adapters=adapter_ids if adapter_ids else None,
            hyperparameters=hyperparameters
        )
        self.models.create(model)
        logger.success(f"Model '{name}' created successfully.")
        return model

    def get_model(self, name: str) -> Model:
        """
        Gets a model by its name.
        
        Args:
            name (str): Name of the model.
            
        Returns:
            Model: The corresponding Model object.
            
        Raises:
            ValueError: If the model doesn't exist.
        """
        return self._get_resource(self.models, name, "Model")    
    
    def delete_model(self, name: str) -> None:
        """
        Deletes an existing model.
        
        Args:
            name (str): Name of the model to delete.
            
        Raises:
            ValueError: If the model doesn't exist.
        """
        self._validate_resource_exists(self.models, name, "Model")
        self.models.delete(name)
        logger.success(f"Model '{name}' deleted successfully.")
    
    # Adapters ---------------------------------------------------------------------------------------
    def list_adapters(self) -> List[Adapter]:
        """
        Gets a list of all available adapters.
        
        Returns:
            List[Adapter]: List of available Adapter objects.
        """
        return self.adapters.list()   

    def create_adapter(self,
                    name: str,
                    model_id: str,
                    dataset_id: str,
                    adapter_type: str,
                    path: str,
                    hyperparameters: Optional[Union[Dict[str, Any], Hyperparameters]] = None) -> Adapter:
        """
        Creates a new adapter.
        
        Args:
            name (str): Unique name for the adapter.
            model_id (str): ID or name of the model this adapter is for.
            dataset_id (str): ID or name of the dataset used to train this adapter.
            adapter_type (str): Type of adapter (e.g., "lora", "qlora").
            path (str): Path to store the adapter.
            hyperparameters (Optional[Union[Dict[str, Any], Hyperparameters]]): Adapter hyperparameters.
            
        Returns:
            Adapter: The created Adapter object.
            
        Raises:
            ValueError: If an adapter with the same name already exists.
        """
        self._validate_resource_doesnt_exist(self.adapters, name, "Adapter")
        
        # Get model object to extract its ID
        model = self._get_resource(self.models, model_id, "Model")
        
        # Get dataset object to extract its ID
        dataset = self._get_resource(self.datasets, dataset_id, "Dataset")
        
        # Convert dict hyperparameters to Hyperparameters if necessary
        if isinstance(hyperparameters, dict):
            hyperparameters = Hyperparameters(**hyperparameters)
        elif hyperparameters is None:
            hyperparameters = Hyperparameters()
            
        adapter = Adapter(
            id=generate_uuid(),
            organization_id=self.organization_id,
            project_id=self.project_id,
            name=name,
            base_model_id=model.id,  # Using 'base_model_id' instead of 'model_id'
            dataset_id=dataset.id,
            type=adapter_type,
            path=path,
            hyperparameters=hyperparameters
        )
        self.adapters.create(adapter)
        logger.success(f"Adapter '{name}' created successfully.")
        return adapter    
    
    def get_adapter(self, name: str) -> Adapter:
        """
        Gets an adapter by its name.
        
        Args:
            name (str): Name of the adapter.
            
        Returns:
            Adapter: The corresponding Adapter object.
            
        Raises:
            ValueError: If the adapter doesn't exist.
        """
        return self._get_resource(self.adapters, name, "Adapter")        
    
    def delete_adapter(self, name: str) -> None:
        """
        Deletes an existing adapter.
        
        Args:
            name (str): Name of the adapter to delete.
            
        Raises:
            ValueError: If the adapter doesn't exist.
        """
        self._validate_resource_exists(self.adapters, name, "Adapter")
        self.adapters.delete(name)
        logger.success(f"Adapter '{name}' deleted successfully.")
    
    # Deployments ------------------------------------------------------------------------------------
    def list_deployments(self) -> List[Deployment]:
        """
        Gets a list of all active deployments.
        
        Returns:
            List[Deployment]: List of active Deployment objects.
        """
        return self.deployments.list()
        
    def deploy(self, 
                model_name: str, 
                deployment_name: Optional[str] = None,
                adapters: Optional[List[str]] = None,
                merge: bool = False,
                environment: str = "development") -> Deployment:
        """
        Deploys a model with optional adapters.
        
        Args:
            model_name (str): Name of the model to deploy.
            deployment_name (Optional[str]): Custom name for the deployment.
            adapters (Optional[List[str]]): List of adapter names to apply.
            merge (bool): Whether to merge adapters with the model.
            environment (str): Deployment environment ("development", "staging", "production").            
            
        Returns:
            Deployment: The created Deployment object.
            
        Raises:
            ValueError: If the model doesn't exist or any adapter doesn't exist.
        """
        # Get model object to extract its ID
        model = self._get_resource(self.models, model_name, "Model")
        
        # Validate adapters
        adapter_ids = []
        if adapters:
            for adapter_name in adapters:
                adapter = self._get_resource(self.adapters, adapter_name, "Adapter")
                adapter_ids.append(adapter.id)
                
        # Create deployment name if not exists
        if not deployment_name:
            deployment_name = f"{model_name}-deployment"
        
        
        # Check if deployment with this name already exists
        self._validate_resource_doesnt_exist(self.deployments, deployment_name, "Deployment")

        # Create deployment
        deployment = Deployment(
            id=generate_uuid(),
            organization_id=self.organization_id,
            project_id=self.project_id,
            name=deployment_name,
            base_model_id=model.id,
            adapters_id=adapter_ids if adapter_ids else None,
            status=DeploymentStatus.PENDING,
            environment=DeploymentEnvironment(environment),
            merge=merge
        )
        
        self.deployments.create(deployment)
        # Trigger deployment process
        self.deployments.deploy(deployment_name)
        
        logger.success(f"Model '{model_name}' deployed successfully as '{deployment_name}'.")
        return deployment
    
    def get_deployment(self, name: str) -> Deployment:
        """
        Gets a deployment by its name.
        
        Args:
            name (str): Name of the deployment.
            
        Returns:
            Deployment: The corresponding Deployment object.
            
        Raises:
            ValueError: If the deployment doesn't exist.
        """
        return self._get_resource(self.deployments, name, "Deployment")
    
    def undeploy(self, deployment_name: str) -> None:
        """
        Deactivates and removes an existing deployment.
        
        Args:
            deployment_name (str): Name of the deployment to remove.
            
        Raises:
            ValueError: If the deployment doesn't exist.
        """        
        self._validate_resource_exists(self.deployments, deployment_name, "Deployment")
        self.deployments.undeploy(deployment_name)
        logger.success(f"Deployment '{deployment_name}' undeployed successfully.")
        
    # Fine-tuning jobs --------------------------------------------------------------------------------    
    def list_fine_tuning_jobs(self) -> List[FineTuningJob]:
        """
        Gets a list of all available fine-tuning
        jobs.

        Returns:
            List[FineTuningJob]: List of available FineTuningJob objects.
        """
        return self.fine_tuning_jobs.list()
    
    def create_fine_tuning_job(self, 
                            job_name: str,
                            base_model: str,
                            dataset: str,
                            job_type: str = "sft",
                            hyperparameters: Optional[Union[Dict[str, Any], Hyperparameters]] = None) -> FineTuningJob:        
        """
        Creates and prepares a fine-tuning job.
        
        Args:
            job_name (str): Unique name for the job.
            base_model (str): Name or ID of the base model to fine-tune.
            dataset (str): Name or ID of the dataset to use.
            job_type (str): Type of fine-tuning job ("sft", "lora", "qlora", etc.).
            hyperparameters (Optional[Union[Dict[str, Any], Hyperparameters]]): Job hyperparameters.
                        
        Returns:
            FineTuningJob: The created fine-tuning job object.
            
        Raises:
            ValueError: If a job with the same name already exists or if the base model or dataset don't exist.
        """       
        self._validate_resource_doesnt_exist(self.fine_tuning_jobs, job_name, "Fine tuning job")
        
        # Get model object to extract its ID/name
        base_model_obj = self._get_resource(self.models, base_model, "Base model")
        
        # Get dataset object to extract its ID
        dataset_obj = self._get_resource(self.datasets, dataset, "Dataset")
        
        # Convert dict hyperparameters to Hyperparameters if necessary
        if isinstance(hyperparameters, dict):
            hyperparameters = Hyperparameters(**hyperparameters)
        elif hyperparameters is None:
            hyperparameters = Hyperparameters()

        job = FineTuningJob(
            id=generate_uuid(),
            organization_id=self.organization_id,
            project_id=self.project_id,
            name=job_name,
            base_model=base_model_obj.name,  # Use name for better readability
            dataset_id=dataset_obj.id,
            job_type=FineTuningJobType(job_type),
            hyperparameters=hyperparameters
        )
        self.fine_tuning_jobs.create(job)   
        logger.success(f"Fine tuning job '{job_name}' created successfully.")     
        
        return job
    

    def get_fine_tuning_job(self, job_name: str) -> FineTuningJob:
        """
        Gets a fine-tuning job by its name.
        
        Args:
            job_name (str): Name of the job.
            
        Returns:
            FineTuningJob: The corresponding FineTuningJob object.
            
        Raises:
            ValueError: If the job doesn't exist.
        """        
        return self._get_resource(self.fine_tuning_jobs, job_name, "Fine tuning job")
        
    def cancel_fine_tuning_job(self, job_name: str) -> None:
        """
        Cancels a running or pending fine-tuning job.
        
        Args:
            job_name (str): Name of the job to cancel.
            
        Raises:
            ValueError: If the job doesn't exist.
        """
        job = self._get_resource(self.fine_tuning_jobs, job_name, "Fine tuning job")
        success = job.cancel()
        if success:
            self.fine_tuning_jobs.update(job)
            logger.success(f"Fine tuning job '{job_name}' cancelled successfully.")
        else:
            logger.warning(f"Fine tuning job '{job_name}' could not be cancelled. Current status: {job.status}")

    def pause_fine_tuning_job(self, job_name: str) -> None:
        """
        Pauses a running fine-tuning job.
        
        Args:
            job_name (str): Name of the job to pause.
            
        Raises:
            ValueError: If the job doesn't exist.
        """
        job = self._get_resource(self.fine_tuning_jobs, job_name, "Fine tuning job")
        success = job.pause()
        if success:
            self.fine_tuning_jobs.update(job)
            logger.success(f"Fine tuning job '{job_name}' paused successfully.")
        else:
            logger.warning(f"Fine tuning job '{job_name}' could not be paused. Current status: {job.status}")
            
    def resume_fine_tuning_job(self, job_name: str) -> None:
        """
        Resumes a paused fine-tuning job.
        
        Args:
            job_name (str): Name of the job to resume.
            
        Raises:
            ValueError: If the job doesn't exist.
        """
        job = self._get_resource(self.fine_tuning_jobs, job_name, "Fine tuning job")
        success = job.resume()
        if success:
            self.fine_tuning_jobs.update(job)
            logger.success(f"Fine tuning job '{job_name}' resumed successfully.")
        else:
            logger.warning(f"Fine tuning job '{job_name}' could not be resumed. Current status: {job.status}")

# Helper methods --------------------------------------------------------------------------------
    def _validate_resource_exists(self, manager, resource_name: str, resource_type: str) -> None:
        """Validates that a resource exists"""
        if not manager.contains(resource_name):
            logger.error(f"{resource_type} '{resource_name}' not found")
            raise ValueError(f"{resource_type} '{resource_name}' not found")
            
    def _validate_resource_doesnt_exist(self, manager, resource_name: str, resource_type: str) -> None:
        """Validates that a resource doesn't exist"""
        if manager.contains(resource_name):
            logger.error(f"{resource_type} '{resource_name}' already exists")
            raise ValueError(f"{resource_type} '{resource_name}' already exists")
            
    def _validate_path_exists(self, path: str) -> None:
        """Validates that a path exists"""
        if not os.path.exists(path):
            logger.error(f"Path not found: {path}")
            raise FileNotFoundError(f"Path not found: {path}")
            
    def _get_resource(self, manager, resource_name: str, resource_type: str) -> Any:
        """Gets a resource, raising a ValueError if it doesn't exist"""
        self._validate_resource_exists(manager, resource_name, resource_type)
        resource = manager.get(resource_name)
        if not resource:
            logger.error(f"{resource_type} '{resource_name}' not found")
            raise ValueError(f"{resource_type} '{resource_name}' not found")
        return resource


