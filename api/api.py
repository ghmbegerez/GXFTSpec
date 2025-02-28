from fastapi import FastAPI, HTTPException, Depends, Header, Path, Body, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from loguru import logger
import os


# Import the SDK class
from sdk import SDK
from sdk.entities import FineTuningJob, FineTuningJobState, DatasetFormat, DatasetFormatField, DatasetSchema

# Initialize the SDK
sdk_instance = SDK()

# Create FastAPI app
app = FastAPI(
    title="SDK API",
    description="API for managing datasets, models, adapters, fine-tuning jobs, and deployments.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication -----------------------------------------------------------------------------------
def authenticate_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Validate the API key.
    This is a simple placeholder. In a production environment, you would use a more secure approach.    
    """
    expected_api_key = os.getenv("API_KEY", "test-api-key")
    if not api_key or api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key

# Health check --------------------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

# Dataset ------------------------------------------------------------------------------------------
#   Methods:
#     list_datasets: None -> List[Dataset]
#     get_dataset: str -> Dataset
#     create_dataset: str, str, Optional[Dict[str, Any]] -> Dataset
#     delete_dataset: str -> None
#
class DatasetCreateRequest(BaseModel):
    """
    Request model for creating a dataset.    
    Attributes:
        name: str: Unique name for the dataset
        path: str: Path to the file or directory containing the data
        schema: Optional[str]: Dataset schema type (completion, chat, instruction, etc.)
        metadata: Optional[Dict[str, Any]]: Additional metadata
    """
    name: str = Field(..., description="Unique name for the dataset")
    path: str = Field(..., description="Path to the file or directory containing the data")
    schema: Optional[str] = Field("custom", description="Dataset schema type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DatasetResponse(BaseModel):
    """
    Response model for creating a dataset.
    Attributes:
        name: str: Unique name for the dataset
        path: str: Path to the file or directory containing the data
        created_at: datetime: Creation date
        metadata: Optional[Dict[str, Any]]: Additional metadata
    """    
    name: str = Field(..., description="Unique name for the dataset")
    path: str = Field(..., description="Path to the file or directory containing the data")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

@app.get("/datasets", response_model=List[DatasetResponse], tags=["Datasets"])
async def list_datasets(api_key: str = Depends(authenticate_api_key)):
    """List all available datasets"""
    try:
        datasets = sdk_instance.list_datasets()
        
        # Convert SDK Dataset objects to DatasetResponse objects
        result = []
        for dataset in datasets:
            result.append(
                DatasetResponse(
                    name=dataset.name,
                    path=dataset.path,
                    created_at=dataset.created_at,
                    metadata={"size": os.path.getsize(dataset.path) if os.path.isfile(dataset.path) else None}
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing datasets: {str(e)}"
        )
    
@app.get("/datasets/{name}", response_model=DatasetResponse, tags=["Datasets"])
async def get_dataset(name: str = Path(..., description="Name of the dataset"), 
                    api_key: str = Depends(authenticate_api_key)):
    """Get dataset by name"""
    try:
        dataset = sdk_instance.get_dataset(name)
        return DatasetResponse(
            name=dataset.name,
            path=dataset.path,
            created_at=dataset.created_at,
            metadata={"size": os.path.getsize(dataset.path) if os.path.isfile(dataset.path) else None}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error getting dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting dataset: {str(e)}"
        )   


@app.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED, tags=["Datasets"])
async def create_dataset(dataset: DatasetCreateRequest, api_key: str = Depends(authenticate_api_key)):
    """Create a new dataset"""
    try:
        # Create a simple format based on schema
        format = DatasetFormat(
            name=f"{dataset.name}_format",
            fields=[DatasetFormatField(name="text", data_type="string", required=True)],
            data_schema=DatasetSchema(dataset.schema)
        )
        
        new_dataset = sdk_instance.create_dataset(
            name=dataset.name, 
            path=dataset.path, 
            format=format
        )
        
        return DatasetResponse(
            name=new_dataset.name,
            path=new_dataset.path,
            created_at=new_dataset.created_at,
            metadata={"size": os.path.getsize(dataset.path) if os.path.isfile(dataset.path) else None}
        )
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating dataset: {str(e)}"
        )
    
@app.delete("/datasets/{name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Datasets"])
async def delete_dataset(name: str = Path(..., description="Name of the dataset"),
                        api_key: str = Depends(authenticate_api_key)):
    """Delete dataset by name"""
    try:
        sdk_instance.delete_dataset(name=name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting dataset: {str(e)}"
        )

# Models -------------------------------------------------------------------------------------------
#   Methods:
#     list_models: None -> List[Model]
#     get_model: str -> Model
#     create_model: str, str -> Model
#     delete_model: str -> None
#
class ModelCreate(BaseModel):
    """
    Request model for creating a model.
    
    Attributes:
        name (str): Unique name for the model
        path (str): Path to the file or directory containing the model
        dataset_id (str): ID or name of the dataset to use for training
        base_model (Optional[str]): Name of the base model to use
        adapters (Optional[List[str]]): List of adapter names to apply
        hyperparameters (Optional[Dict[str, Any]]): Hyperparameters for the model
    """
    name: str = Field(..., description="Unique name for the model")
    path: str = Field(..., description="Path to the file or directory containing the model")
    dataset_id: str = Field(..., description="ID or name of the dataset to use for training")
    base_model: Optional[str] = Field(None, description="Name of the base model to use")
    adapters: Optional[List[str]] = Field(None, description="List of adapter names to apply")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Hyperparameters for the model")

class ModelResponse(BaseModel):
    """
    Response model for model operations.
    
    Attributes:
        name (str): Unique name for the model
        path (str): Path to the file or directory containing the model
        created_at (datetime): Creation date
        metadata (Optional[Dict[str, Any]]): Additional metadata
    """
    name: str = Field(..., description="Unique name for the model")
    path: str = Field(..., description="Path to the file or directory containing the model")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

@app.get("/models", response_model=List[ModelResponse], tags=["Models"])
async def list_models(api_key: str = Depends(authenticate_api_key)):
    """List all available models"""
    try:
        models = sdk_instance.list_models()
        
        # Convert SDK Model objects to ModelResponse objects
        result = []
        for model in models:
            result.append(
                ModelResponse(
                    name=model.name,
                    path=model.path,
                    created_at=model.created_at,
                    metadata={"framework": "pytorch"}  # Example metadata
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )

@app.get("/models/{name}", response_model=ModelResponse, tags=["Models"])
async def get_model(name: str = Path(..., description="Name of the model"), 
                    api_key: str = Depends(authenticate_api_key)):
    """Get model by name"""
    try:
        model = sdk_instance.get_model(name=name)
        return ModelResponse(
            name=model.name,
            path=model.path,
            created_at=model.created_at,
            metadata={"framework": "pytorch"}  # Example metadata
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model: {str(e)}"
        )
    
@app.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED, tags=["Models"])
async def create_model(model: ModelCreate, api_key: str = Depends(authenticate_api_key)):
    """Create a new model"""
    try:
        new_model = sdk_instance.create_model(
            name=model.name, 
            path=model.path,
            dataset_id=model.dataset_id,
            base_model=model.base_model,
            adapters=model.adapters,
            hyperparameters=model.hyperparameters
        )
        
        return ModelResponse(
            name=new_model.name,
            path=new_model.path,
            created_at=new_model.created_at,
            metadata={"framework": "pytorch"}  # Example metadata
        )
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating model: {str(e)}"
        )

@app.delete("/models/{name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Models"])
async def delete_model(name: str = Path(..., description="Name of the model"),
                        api_key: str = Depends(authenticate_api_key)):
    """Delete model by name"""
    try:
        sdk_instance.delete_model(name=name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}"
        )

# Adapters -----------------------------------------------------------------------------------------
#   Methods:
#     list_adapters: None -> List[Adapter]
#     get_adapter: str -> Adapter
#     create_adapter: str, str, str, str, str, Optional -> Adapter
#     delete_adapter: str -> None
#
class AdapterCreate(BaseModel):
    """
    Request model for creating an adapter.
    
    Attributes:
        name (str): Unique name for the adapter
        model_id (str): ID of the model this adapter is for
        dataset_id (str): ID of the dataset used to train this adapter
        adapter_type (str): Type of adapter (e.g., "lora", "qlora")
        path (str): Path to store the adapter
        hyperparameters (Optional[Dict[str, Any]]): Adapter hyperparameters
    """
    name: str = Field(..., description="Unique name for the adapter")
    model_id: str = Field(..., description="ID or name of the model this adapter is for")
    dataset_id: str = Field(..., description="ID or name of the dataset used to train this adapter")
    adapter_type: str = Field(..., description="Type of adapter (e.g., 'lora', 'qlora')")
    path: str = Field(..., description="Path to store the adapter")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Adapter hyperparameters")

class AdapterResponse(BaseModel):
    """
    Response model for adapter operations.
    
    Attributes:
        name (str): Unique name for the adapter
        base_model (str): ID/name of the base model
        type (str): Type of adapter
        created_at (datetime): Creation date
        metadata (Optional[Dict[str, Any]]): Additional metadata
    """
    name: str = Field(..., description="Unique name for the adapter")
    base_model: str = Field(..., description="ID/name of the base model")
    type: str = Field(..., description="Type of adapter")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

@app.get("/adapters", response_model=List[AdapterResponse], tags=["Adapters"])
async def list_adapters(api_key: str = Depends(authenticate_api_key)):
    """List all available adapters"""
    try:
        adapters = sdk_instance.list_adapters()
        
        # Convert SDK Adapter objects to AdapterResponse objects
        result = []
        for adapter in adapters:
            # Get model name for better display
            model_name = adapter.base_model_id
            try:
                model = sdk_instance.models.get(adapter.base_model_id)
                if model:
                    model_name = model.name
            except:
                pass
                
            result.append(
                AdapterResponse(
                    name=adapter.name,
                    base_model=model_name,
                    type=adapter.type,
                    created_at=adapter.created_at,
                    metadata={"rank": 8}  # Example metadata
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing adapters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing adapters: {str(e)}"
        )

@app.get("/adapters/{name}", response_model=AdapterResponse, tags=["Adapters"])
async def get_adapter(name: str = Path(..., description="Name of the adapter"), 
                    api_key: str = Depends(authenticate_api_key)):
    """Get adapter by name"""
    try:
        adapter = sdk_instance.get_adapter(name=name)
        
        # Get model name for better display
        model_name = adapter.base_model_id
        try:
            model = sdk_instance.models.get(adapter.base_model_id)
            if model:
                model_name = model.name
        except:
            pass
            
        return AdapterResponse(
            name=adapter.name,
            base_model=model_name,
            type=adapter.type,
            created_at=adapter.created_at,
            metadata={"rank": 8}  # Example metadata
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting adapter: {str(e)}"
        )

@app.post("/adapters", response_model=AdapterResponse, status_code=status.HTTP_201_CREATED, tags=["Adapters"])
async def create_adapter(adapter: AdapterCreate, api_key: str = Depends(authenticate_api_key)):
    """Create a new adapter"""
    try:
        new_adapter = sdk_instance.create_adapter(
            name=adapter.name,
            model_id=adapter.model_id,
            dataset_id=adapter.dataset_id,
            adapter_type=adapter.adapter_type,
            path=adapter.path,
            hyperparameters=adapter.hyperparameters
        )
        
        # Get model name for better display
        model_name = new_adapter.base_model_id
        try:
            model = sdk_instance.models.get(new_adapter.base_model_id)
            if model:
                model_name = model.name
        except:
            pass
            
        return AdapterResponse(
            name=new_adapter.name,
            base_model=model_name,
            type=new_adapter.type,
            created_at=new_adapter.created_at,
            metadata={"rank": 8}  # Example metadata
        )
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating adapter: {str(e)}"
        )

@app.delete("/adapters/{name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Adapters"])
async def delete_adapter(name: str = Path(..., description="Name of the adapter"),
                        api_key: str = Depends(authenticate_api_key)):
    """Delete adapter by name"""
    try:
        sdk_instance.delete_adapter(name=name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting adapter: {str(e)}"
        )

# Fine-tuning jobs ---------------------------------------------------------------------------------  
class CreateFineTuningJobRequest(BaseModel):
    """
    Request model for creating a fine-tuning job.
    
    Attributes:
        job_name (str): Unique name for the job
        base_model (str): Name of the base model to fine-tune
        dataset (str): Name of the dataset to use
        job_type (str): Type of fine-tuning job ("sft", "lora", "qlora", etc.)
        hyperparameters (Optional[Dict[str, Any]]): Job hyperparameters
    """
    job_name: str = Field(..., description="Unique name for the job")
    base_model: str = Field(..., description="Name of the base model to fine-tune")
    dataset: str = Field(..., description="Name of the dataset to use")
    job_type: str = Field("sft", description="Type of fine-tuning job")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Job hyperparameters")

class FineTuningJobResponse(BaseModel):
    """
    Response model for fine-tuning job operations.
    
    Attributes:
        id (str): Job ID
        name (str): Job name
        base_model (str): Base model name
        status (str): Job status
        job_type (str): Type of job
        created_at (datetime): Creation date
        started_at (Optional[datetime]): Start date
        completed_at (Optional[datetime]): Completion date
    """
    id: str = Field(..., description="Job ID")
    name: str = Field(..., description="Job name")
    base_model: str = Field(..., description="Base model name")
    status: str = Field(..., description="Job status")
    job_type: str = Field(..., description="Type of job")
    created_at: datetime = Field(..., description="Creation date")
    started_at: Optional[datetime] = Field(None, description="Start date")
    completed_at: Optional[datetime] = Field(None, description="Completion date")

@app.post("/fine_tuning/jobs", response_model=FineTuningJobResponse, status_code=status.HTTP_201_CREATED, tags=["Fine-Tuning"])
async def create_fine_tuning_job(request: CreateFineTuningJobRequest, api_key: str = Depends(authenticate_api_key)):
    """Create a new fine-tuning job"""
    try:
        job = sdk_instance.create_fine_tuning_job(
            job_name=request.job_name,
            base_model=request.base_model,
            dataset=request.dataset,
            job_type=request.job_type,
            hyperparameters=request.hyperparameters
        )
        
        return FineTuningJobResponse(
            id=job.id,
            name=job.name,
            base_model=job.base_model,
            status=job.status,
            job_type=job.job_type,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating fine-tuning job: {str(e)}"
        )

@app.get("/fine_tuning/jobs", response_model=List[FineTuningJobResponse], tags=["Fine-Tuning"])
async def list_fine_tuning_jobs(limit: int = 20, after: Optional[str] = None, api_key: str = Depends(authenticate_api_key)):
    """List all fine-tuning jobs"""
    try:
        jobs = sdk_instance.list_fine_tuning_jobs()
        
        # Filter jobs if after parameter is provided
        if after:
            jobs = [job for job in jobs if job.id > after]
            
        # Apply limit
        jobs = jobs[:limit]
        
        # Convert to response model
        result = []
        for job in jobs:
            result.append(
                FineTuningJobResponse(
                    id=job.id,
                    name=job.name,
                    base_model=job.base_model,
                    status=job.status,
                    job_type=job.job_type,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing fine-tuning jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing fine-tuning jobs: {str(e)}"
        )

@app.get("/fine_tuning/jobs/{job_name}", response_model=FineTuningJobResponse, tags=["Fine-Tuning"])
async def get_fine_tuning_job(job_name: str = Path(..., description="Name of the fine-tuning job"), 
                            api_key: str = Depends(authenticate_api_key)):
    """Get a fine-tuning job by name"""
    try:
        job = sdk_instance.get_fine_tuning_job(job_name)
        
        return FineTuningJobResponse(
            id=job.id,
            name=job.name,
            base_model=job.base_model,
            status=job.status,
            job_type=job.job_type,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting fine-tuning job: {str(e)}"
        )

@app.delete("/fine_tuning/jobs/{job_name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Fine-Tuning"])
async def cancel_fine_tuning_job(job_name: str = Path(..., description="Name of the fine-tuning job to cancel"),
                                api_key: str = Depends(authenticate_api_key)):
    """Cancel a fine-tuning job"""
    try:
        sdk_instance.cancel_fine_tuning_job(job_name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error cancelling fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling fine-tuning job: {str(e)}"
        )

@app.post("/fine_tuning/jobs/{job_name}/pause", status_code=status.HTTP_204_NO_CONTENT, tags=["Fine-Tuning"])
async def pause_fine_tuning_job(job_name: str = Path(..., description="Name of the fine-tuning job to pause"),
                            api_key: str = Depends(authenticate_api_key)):
    """Pause a fine-tuning job"""
    try:
        sdk_instance.pause_fine_tuning_job(job_name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error pausing fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error pausing fine-tuning job: {str(e)}"
        )

@app.post("/fine_tuning/jobs/{job_name}/resume", status_code=status.HTTP_204_NO_CONTENT, tags=["Fine-Tuning"])
async def resume_fine_tuning_job(job_name: str = Path(..., description="Name of the fine-tuning job to resume"),
                                api_key: str = Depends(authenticate_api_key)):
    """Resume a fine-tuning job"""
    try:
        sdk_instance.resume_fine_tuning_job(job_name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error resuming fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resuming fine-tuning job: {str(e)}"
        )

# Deployment --------------------------------------------------------------------------------------- 
#  Methods:
#    list_deployments: None -> List[Deployment]    
#    deploy: str, Optional[List[str]], Optional[str] -> Deployment
#    get_deployment: str -> Deployment
#    undeploy: str -> None
#
class DeploymentCreate(BaseModel):
    """
    Request model for creating a deployment.
    Attributes:
        model_name: str: Name of the model to deploy
        adapters: Optional[List[str]]: List of adapter names to apply
        deployment_name: Optional[str]: Custom name for the deployment
        merge: bool: Whether to merge adapters with the model
        environment: str: Deployment environment ("development", "staging", "production")
    """
    model_name: str = Field(..., description="Name of the model to deploy")
    adapters: Optional[List[str]] = Field(None, description="List of adapter names to apply")
    deployment_name: Optional[str] = Field(None, description="Custom name for the deployment")
    merge: bool = Field(False, description="Whether to merge adapters with the model")
    environment: str = Field("development", description="Deployment environment")


class DeploymentResponse(BaseModel):
    """
    Response model for creating a deployment.
    Attributes:
        id: str: Deployment ID
        name: str: Unique name for the deployment
        model: str: Model ID or name
        adapters: Optional[List[str]]: List of adapter IDs or names
        status: str: Deployment status
        environment: str: Deployment environment
        endpoint: Optional[str]: API endpoint
        created_at: datetime: Creation date
        last_updated: datetime: Last update date
        metadata: Optional[Dict[str, Any]]: Additional metadata
    """
    id: str = Field(..., description="Deployment ID")
    name: str = Field(..., description="Unique name for the deployment")
    model: str = Field(..., description="Model ID or name")
    adapters: Optional[List[str]] = Field(None, description="List of adapter IDs or names")
    status: str = Field(..., description="Deployment status")
    environment: str = Field(..., description="Deployment environment")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation date")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

@app.get("/deployments", response_model=List[DeploymentResponse], tags=["Deployments"])
async def list_deployments(api_key: str = Depends(authenticate_api_key)):
    """List all active deployments"""
    try:
        deployments = sdk_instance.list_deployments()
        
        # Convert SDK Deployment objects to DeploymentResponse objects
        result = []
        for deployment in deployments:
            # Get model name for better display
            model_name = deployment.base_model_id
            try:
                model = sdk_instance.models.get(deployment.base_model_id)
                if model:
                    model_name = model.name
            except:
                pass
                
            # Get adapter names for better display
            adapter_names = []
            if hasattr(deployment, 'adapters_id') and deployment.adapters_id:
                for adapter_id in deployment.adapters_id:
                    try:
                        adapter = sdk_instance.adapters.get(adapter_id)
                        if adapter:
                            adapter_names.append(adapter.name)
                        else:
                            adapter_names.append(adapter_id)
                    except:
                        adapter_names.append(adapter_id)
            
            # Format status properly
            status = deployment.status
            if hasattr(deployment.status, 'value'):
                status = deployment.status.value
                
            # Format environment properly
            environment = deployment.environment
            if hasattr(deployment.environment, 'value'):
                environment = deployment.environment.value
                
            result.append(
                DeploymentResponse(
                    id=deployment.id,
                    name=deployment.name,
                    model=model_name,
                    adapters=adapter_names if adapter_names else None,
                    status=status,
                    environment=environment,
                    endpoint=f"https://api.example.com/v1/predictions/{deployment.name}",
                    created_at=deployment.created_at,
                    last_updated=deployment.last_updated if hasattr(deployment, 'last_updated') else deployment.created_at
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing deployments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing deployments: {str(e)}"
        )


@app.post("/deployments", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED, tags=["Deployments"])
async def create_deployment(deployment: DeploymentCreate, api_key: str = Depends(authenticate_api_key)):
    """Create a new deployment"""
    try:
        deployment_obj = sdk_instance.deploy(
            model_name=deployment.model_name,
            adapters=deployment.adapters,
            deployment_name=deployment.deployment_name,
            merge=deployment.merge,
            environment=deployment.environment
        )
        
        # Get model name for better display
        model_name = deployment_obj.base_model_id
        try:
            model = sdk_instance.models.get(deployment_obj.base_model_id)
            if model:
                model_name = model.name
        except:
            pass
                
        # Get adapter names for better display
        adapter_names = []
        if hasattr(deployment_obj, 'adapters_id') and deployment_obj.adapters_id:
            for adapter_id in deployment_obj.adapters_id:
                try:
                    adapter = sdk_instance.adapters.get(adapter_id)
                    if adapter:
                        adapter_names.append(adapter.name)
                    else:
                        adapter_names.append(adapter_id)
                except:
                    adapter_names.append(adapter_id)
        
        # Format status properly
        status = deployment_obj.status
        if hasattr(deployment_obj.status, 'value'):
            status = deployment_obj.status.value
            
        # Format environment properly
        environment = deployment_obj.environment
        if hasattr(deployment_obj.environment, 'value'):
            environment = deployment_obj.environment.value
        
        return DeploymentResponse(
            id=deployment_obj.id,
            name=deployment_obj.name,
            model=model_name,
            adapters=adapter_names if adapter_names else None,
            status=status,
            environment=environment,
            endpoint=f"https://api.example.com/v1/predictions/{deployment_obj.name}",
            created_at=deployment_obj.created_at,
            last_updated=deployment_obj.last_updated if hasattr(deployment_obj, 'last_updated') else deployment_obj.created_at
        )
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating deployment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating deployment: {str(e)}"
        )

@app.get("/deployments/{name}", response_model=DeploymentResponse, tags=["Deployments"])
async def get_deployment(name: str = Path(..., description="Name of the deployment"), 
                        api_key: str = Depends(authenticate_api_key)):
    """Get deployment by name"""
    try:
        deployment = sdk_instance.get_deployment(name=name)
        
        # Get model name for better display
        model_name = deployment.base_model_id
        try:
            model = sdk_instance.models.get(deployment.base_model_id)
            if model:
                model_name = model.name
        except:
            pass
                
        # Get adapter names for better display
        adapter_names = []
        if hasattr(deployment, 'adapters_id') and deployment.adapters_id:
            for adapter_id in deployment.adapters_id:
                try:
                    adapter = sdk_instance.adapters.get(adapter_id)
                    if adapter:
                        adapter_names.append(adapter.name)
                    else:
                        adapter_names.append(adapter_id)
                except:
                    adapter_names.append(adapter_id)
        
        # Format status properly
        status = deployment.status
        if hasattr(deployment.status, 'value'):
            status = deployment.status.value
            
        # Format environment properly
        environment = deployment.environment
        if hasattr(deployment.environment, 'value'):
            environment = deployment.environment.value
        
        return DeploymentResponse(
            id=deployment.id,
            name=deployment.name,
            model=model_name,
            adapters=adapter_names if adapter_names else None,
            status=status,
            environment=environment,
            endpoint=f"https://api.example.com/v1/predictions/{deployment.name}",
            created_at=deployment.created_at,
            last_updated=deployment.last_updated if hasattr(deployment, 'last_updated') else deployment.created_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error getting deployment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting deployment: {str(e)}"
        )


@app.delete("/deployments/{name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Deployments"])
async def delete_deployment(name: str = Path(..., description="Name of the deployment"),
                            api_key: str = Depends(authenticate_api_key)):
    """Deactivate and remove a deployment"""
    try:
        sdk_instance.undeploy(deployment_name=name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error undeploying deployment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error undeploying deployment: {str(e)}"
        )