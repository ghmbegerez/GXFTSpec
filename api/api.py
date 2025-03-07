from fastapi import FastAPI, HTTPException, Depends, Header, Path, Body, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from loguru import logger
import os


# Import the SDK class
from sdk.sdk import SDK
from sdk.entities import (
    FineTuningJob, FineTuningJobState, FineTuningMethod, AdapterType,
    Model, Adapter, Checkpoint, Metrics, Event, File, ModelType, ModelOrigin
)

# Initialize the SDK
sdk_instance = SDK()

# Create FastAPI app
app = FastAPI(
    title="Fine-Tuning SDK API",
    description="API for managing models, adapters, fine-tuning jobs, and hub operations.",
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

# Hub Models --------------------------------------------------------------------------------------
class HubModelResponse(BaseModel):
    """
    Response model for hub model operations.
    
    Attributes:
        id (str): Model ID
        name (str): Unique name for the model
        type (str): Type of model (base or fine-tuned)
        origin (str): Origin of model (local or remote)
        path (str): Path to the model files
        created_at (datetime): Creation date
        description (Optional[str]): Model description
        base_model_id (Optional[str]): ID of base model (for fine-tuned models)
        metadata (Optional[Dict[str, Any]]): Additional metadata
    """
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Unique name for the model")
    type: str = Field(..., description="Type of model (base or fine-tuned)")
    origin: str = Field(..., description="Origin of model (local or remote)")
    path: str = Field(..., description="Path to the model files")
    created_at: datetime = Field(..., description="Creation date")
    description: Optional[str] = Field(None, description="Model description")
    base_model_id: Optional[str] = Field(None, description="ID of base model (for fine-tuned models)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class HubModelCreate(BaseModel):
    """
    Request model for creating a hub model.
    
    Attributes:
        name (str): Unique name for the model
        model_type (str): Type of model (base or fine-tuned)
        path (str): Path to model files
        description (Optional[str]): Model description
        base_model_id (Optional[str]): Base model ID for fine-tuned models
    """
    name: str = Field(..., description="Unique name for the model")
    model_type: str = Field(..., description="Type of model (base or fine-tuned)")
    path: str = Field(..., description="Path to model files")
    description: Optional[str] = Field(None, description="Model description")
    base_model_id: Optional[str] = Field(None, description="Base model ID for fine-tuned models")

@app.get("/hub/models", response_model=List[HubModelResponse], tags=["Hub Models"])
async def list_hub_models(api_key: str = Depends(authenticate_api_key)):
    """List all models in the hub"""
    try:
        models = sdk_instance.list_hub_models()
        result = []
        for model_data in models:
            # Convert each model to the response model
            model = model_data.get("model", {})
            result.append(
                HubModelResponse(
                    id=model.get("id", ""),
                    name=model.get("name", ""),
                    type=model.get("type", ""),
                    origin=model.get("origin", ""),
                    path=model.get("path", ""),
                    created_at=model.get("created_at", datetime.utcnow()),
                    description=model.get("description", None),
                    base_model_id=model.get("base_model_id", None),
                    metadata={"framework": "pytorch"}  # Example metadata
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing hub models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing hub models: {str(e)}"
        )

@app.get("/hub/models/{name}", response_model=HubModelResponse, tags=["Hub Models"])
async def get_hub_model(name: str = Path(..., description="Name of the model"), 
                      api_key: str = Depends(authenticate_api_key)):
    """Get a model from the hub by name"""
    try:
        model = sdk_instance.get_hub_model(name)
        return HubModelResponse(
            id=model.id,
            name=model.name,
            type=model.type.value if hasattr(model.type, 'value') else model.type,
            origin=model.origin.value if hasattr(model.origin, 'value') else model.origin,
            path=model.path,
            created_at=model.created_at,
            description=model.description,
            base_model_id=model.base_model_id,
            metadata={"framework": "pytorch"}  # Example metadata
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting hub model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting hub model: {str(e)}"
        )

@app.post("/hub/models", response_model=HubModelResponse, status_code=status.HTTP_201_CREATED, tags=["Hub Models"])
async def create_hub_model(model: HubModelCreate, api_key: str = Depends(authenticate_api_key)):
    """Create a new model in the hub"""
    try:
        new_model = sdk_instance.create_hub_model(
            name=model.name,
            model_type=model.model_type,
            path=model.path,
            description=model.description,
            base_model_id=model.base_model_id
        )
        
        return HubModelResponse(
            id=new_model.id,
            name=new_model.name,
            type=new_model.type.value if hasattr(new_model.type, 'value') else new_model.type,
            origin=new_model.origin.value if hasattr(new_model.origin, 'value') else new_model.origin,
            path=new_model.path,
            created_at=new_model.created_at,
            description=new_model.description,
            base_model_id=new_model.base_model_id,
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
        logger.error(f"Error creating hub model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating hub model: {str(e)}"
        )

@app.delete("/hub/models/{name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Hub Models"])
async def delete_hub_model(name: str = Path(..., description="Name of the model"),
                          api_key: str = Depends(authenticate_api_key)):
    """Delete a model from the hub"""
    try:
        sdk_instance.delete_hub_model(model_id_or_name=name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting hub model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting hub model: {str(e)}"
        )

# Hub Adapters -------------------------------------------------------------------------------------
class HubAdapterResponse(BaseModel):
    """
    Response model for hub adapter operations.
    
    Attributes:
        id (str): Adapter ID
        name (str): Unique name for the adapter
        type (str): Type of adapter (lora, qlora)
        model_id (str): ID of the associated model
        dataset (str): Dataset used for training
        path (str): Path to adapter files
        created_at (datetime): Creation date
        description (Optional[str]): Adapter description
        metadata (Optional[Dict[str, Any]]): Additional metadata
    """
    id: str = Field(..., description="Adapter ID")
    name: str = Field(..., description="Unique name for the adapter")
    type: str = Field(..., description="Type of adapter")
    model_id: str = Field(..., description="ID of the associated model")
    dataset: str = Field(..., description="Dataset used for training")
    path: str = Field(..., description="Path to adapter files")
    created_at: datetime = Field(..., description="Creation date")
    description: Optional[str] = Field(None, description="Adapter description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class HubAdapterCreate(BaseModel):
    """
    Request model for creating a hub adapter.
    
    Attributes:
        name (str): Unique name for the adapter
        adapter_type (str): Type of adapter (lora, qlora)
        model_id (str): ID of the associated model
        dataset (str): Dataset used for training
        path (str): Path to adapter files
        description (Optional[str]): Adapter description
        hyperparameters (Optional[Dict[str, Any]]): Training hyperparameters
    """
    name: str = Field(..., description="Unique name for the adapter")
    adapter_type: str = Field(..., description="Type of adapter (lora, qlora)")
    model_id: str = Field(..., description="ID of the associated model")
    dataset: str = Field(..., description="Dataset used for training")
    path: str = Field(..., description="Path to adapter files")
    description: Optional[str] = Field(None, description="Adapter description")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Training hyperparameters")

@app.get("/hub/adapters", response_model=List[HubAdapterResponse], tags=["Hub Adapters"])
async def list_hub_adapters(api_key: str = Depends(authenticate_api_key)):
    """List all adapters in the hub"""
    try:
        adapters = sdk_instance.list_hub_adapters()
        result = []
        for adapter_data in adapters:
            # Convert each adapter to the response model
            adapter = adapter_data.get("adapter", {})
            result.append(
                HubAdapterResponse(
                    id=adapter.get("id", ""),
                    name=adapter.get("name", ""),
                    type=adapter.get("type", ""),
                    model_id=adapter.get("model_id", ""),
                    dataset=adapter.get("dataset", ""),
                    path=adapter.get("path", ""),
                    created_at=adapter.get("created_at", datetime.utcnow()),
                    description=adapter.get("description", None),
                    metadata={"rank": 8}  # Example metadata
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing hub adapters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing hub adapters: {str(e)}"
        )

@app.get("/hub/adapters/{name}", response_model=HubAdapterResponse, tags=["Hub Adapters"])
async def get_hub_adapter(name: str = Path(..., description="Name of the adapter"), 
                        api_key: str = Depends(authenticate_api_key)):
    """Get an adapter from the hub by name"""
    try:
        adapter = sdk_instance.get_hub_adapter(name)
        return HubAdapterResponse(
            id=adapter.id,
            name=adapter.name,
            type=adapter.adapter_type.value if hasattr(adapter.adapter_type, 'value') else adapter.adapter_type,
            model_id=adapter.model_id,
            dataset=adapter.dataset,
            path=adapter.path,
            created_at=adapter.created_at,
            description=adapter.description if hasattr(adapter, 'description') else None,
            metadata={"rank": 8}  # Example metadata
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting hub adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting hub adapter: {str(e)}"
        )

@app.post("/hub/adapters", response_model=HubAdapterResponse, status_code=status.HTTP_201_CREATED, tags=["Hub Adapters"])
async def create_hub_adapter(adapter: HubAdapterCreate, api_key: str = Depends(authenticate_api_key)):
    """Create a new adapter in the hub"""
    try:
        new_adapter = sdk_instance.create_hub_adapter(
            name=adapter.name,
            adapter_type=adapter.adapter_type,
            model_id=adapter.model_id,
            dataset=adapter.dataset,
            path=adapter.path,
            description=adapter.description,
            hyperparameters=adapter.hyperparameters
        )
        
        return HubAdapterResponse(
            id=new_adapter.id,
            name=new_adapter.name,
            type=new_adapter.adapter_type.value if hasattr(new_adapter.adapter_type, 'value') else new_adapter.adapter_type,
            model_id=new_adapter.model_id,
            dataset=new_adapter.dataset,
            path=new_adapter.path,
            created_at=new_adapter.created_at,
            description=new_adapter.description if hasattr(new_adapter, 'description') else None,
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
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating hub adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating hub adapter: {str(e)}"
        )

@app.delete("/hub/adapters/{name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Hub Adapters"])
async def delete_hub_adapter(name: str = Path(..., description="Name of the adapter"),
                           api_key: str = Depends(authenticate_api_key)):
    """Delete an adapter from the hub"""
    try:
        sdk_instance.delete_hub_adapter(adapter_id_or_name=name)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting hub adapter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting hub adapter: {str(e)}"
        )

# Fine-tuning jobs ---------------------------------------------------------------------------------  
class FineTuningJobCreate(BaseModel):
    """
    Request model for creating a fine-tuning job.
    
    Attributes:
        name (str): Unique name for the job
        base_model (str): Name of the base model to fine-tune
        dataset (str): Path to the training dataset
        val_dataset (Optional[str]): Path to the validation dataset
        job_type (str): Type of fine-tuning job ("sft", "lora", "qlora")
        output_model_name (Optional[str]): Name for the output model
        hyperparameters (Optional[Dict[str, Any]]): Job hyperparameters
    """
    name: str = Field(..., description="Unique name for the job")
    base_model: str = Field(..., description="Name or path of base model to fine-tune")
    dataset: str = Field(..., description="Path to training dataset")
    val_dataset: Optional[str] = Field(None, description="Optional path to validation dataset")
    job_type: str = Field("sft", description="Type of fine-tuning (sft, lora, qlora)")
    output_model_name: Optional[str] = Field(None, description="Optional name for output model")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Optional hyperparameters for training")

class FineTuningJobResponse(BaseModel):
    """
    Response model for fine-tuning job operations.
    
    Attributes:
        id (str): Job ID
        name (str): Job name
        base_model (str): Base model path or name
        dataset (str): Training dataset path
        val_dataset (Optional[str]): Validation dataset path
        status (str): Job status
        created_at (datetime): Creation date
        started_at (Optional[datetime]): Start date
        completed_at (Optional[datetime]): Completion date
        logs_path (str): Path to job logs
        output_model_id (Optional[str]): ID of the output model
    """
    id: str = Field(..., description="Job ID")
    name: str = Field(..., description="Job name")
    organization_id: str = Field(..., description="Organization ID")
    base_model: str = Field(..., description="Base model path or name")
    dataset: str = Field(..., description="Training dataset path")
    val_dataset: Optional[str] = Field(None, description="Validation dataset path")
    status: str = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Creation date")
    started_at: Optional[datetime] = Field(None, description="Start date")
    completed_at: Optional[datetime] = Field(None, description="Completion date")
    logs_path: str = Field(..., description="Path to job logs")
    output_model_id: Optional[str] = Field(None, description="ID of the output model")
    description: Optional[str] = Field(None, description="Job description")

@app.post("/fine_tuning/jobs", response_model=FineTuningJobResponse, status_code=status.HTTP_201_CREATED, tags=["Fine-Tuning"])
async def create_fine_tuning_job(request: FineTuningJobCreate, api_key: str = Depends(authenticate_api_key)):
    """Create a new fine-tuning job"""
    try:
        job = sdk_instance.create_fine_tuning_job(
            name=request.name,
            base_model=request.base_model,
            dataset=request.dataset,
            val_dataset=request.val_dataset,
            job_type=request.job_type,
            output_model_name=request.output_model_name,
            hyperparameters=request.hyperparameters
        )
        
        return FineTuningJobResponse(
            id=job.id,
            name=job.name,
            organization_id=job.organization_id,
            base_model=job.base_model,
            dataset=job.dataset,
            val_dataset=job.val_dataset,
            status=job.status.value if hasattr(job.status, 'value') else job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            logs_path=job.logs_path,
            output_model_id=job.output_model_id,
            description=job.description
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
        logger.error(f"Error creating fine-tuning job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating fine-tuning job: {str(e)}"
        )

@app.get("/fine_tuning/jobs", response_model=List[FineTuningJobResponse], tags=["Fine-Tuning"])
async def list_fine_tuning_jobs(limit: int = 20, after: Optional[str] = None, api_key: str = Depends(authenticate_api_key)):
    """List all fine-tuning jobs"""
    try:
        jobs_data = sdk_instance.list_fine_tuning_jobs()
        
        # Filter jobs if after parameter is provided
        if after:
            filtered_jobs = []
            for job_data in jobs_data:
                if job_data.get("id", "") > after:
                    filtered_jobs.append(job_data)
            jobs_data = filtered_jobs
            
        # Apply limit
        jobs_data = jobs_data[:limit]
        
        # Convert to response model
        result = []
        for job_data in jobs_data:
            job = job_data.get("job", {})
            
            # Format status properly
            status = job.get("status", "")
            if isinstance(status, dict) and "value" in status:
                status = status["value"]
                
            result.append(
                FineTuningJobResponse(
                    id=job.get("id", ""),
                    name=job.get("name", ""),
                    organization_id=job.get("organization_id", ""),
                    base_model=job.get("base_model", ""),
                    dataset=job.get("dataset", ""),
                    val_dataset=job.get("val_dataset", None),
                    status=status,
                    created_at=job.get("created_at", datetime.utcnow()),
                    started_at=job.get("started_at", None),
                    completed_at=job.get("completed_at", None),
                    logs_path=job.get("logs_path", ""),
                    output_model_id=job.get("output_model_id", None),
                    description=job.get("description", None)
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
            organization_id=job.organization_id,
            base_model=job.base_model,
            dataset=job.dataset,
            val_dataset=job.val_dataset,
            status=job.status.value if hasattr(job.status, 'value') else job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            logs_path=job.logs_path,
            output_model_id=job.output_model_id,
            description=job.description
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

# Checkpoints --------------------------------------------------------------------------------------
class CheckpointResponse(BaseModel):
    """
    Response model for checkpoint operations.
    
    Attributes:
        id (str): Checkpoint ID
        job_id (str): Associated job ID
        step_number (int): Training step number
        path (str): Path to checkpoint files
        created_at (datetime): Creation date
        metrics (Dict[str, float]): Training metrics
    """
    id: str = Field(..., description="Checkpoint ID")
    job_id: str = Field(..., description="Associated job ID")
    step_number: int = Field(..., description="Training step number")
    path: str = Field(..., description="Path to checkpoint files")
    created_at: datetime = Field(..., description="Creation date")
    metrics: Dict[str, float] = Field(..., description="Training metrics")

@app.get("/checkpoints", response_model=List[CheckpointResponse], tags=["Checkpoints"])
async def list_checkpoints(job_id: Optional[str] = None, api_key: str = Depends(authenticate_api_key)):
    """List all checkpoints, optionally filtered by job"""
    try:
        checkpoints = sdk_instance.list_checkpoints(job_id_or_name=job_id)
        
        result = []
        for checkpoint_data in checkpoints:
            # Convert each checkpoint to the response model
            checkpoint = checkpoint_data.get("checkpoint", {})
            metrics_data = checkpoint.get("metrics", {})
            
            # Format metrics
            metrics = {}
            if "train_loss" in metrics_data:
                metrics["train_loss"] = float(metrics_data["train_loss"])
            if "valid_loss" in metrics_data:
                metrics["valid_loss"] = float(metrics_data["valid_loss"])
                
            result.append(
                CheckpointResponse(
                    id=checkpoint.get("id", ""),
                    job_id=checkpoint.get("job_id", ""),
                    step_number=checkpoint.get("step_number", 0),
                    path=checkpoint.get("path", ""),
                    created_at=checkpoint.get("created_at", datetime.utcnow()),
                    metrics=metrics
                )
            )
        return result
    except Exception as e:
        logger.error(f"Error listing checkpoints: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing checkpoints: {str(e)}"
        )

@app.get("/checkpoints/{checkpoint_id}", response_model=CheckpointResponse, tags=["Checkpoints"])
async def get_checkpoint(checkpoint_id: str = Path(..., description="ID of the checkpoint"), 
                       api_key: str = Depends(authenticate_api_key)):
    """Get a checkpoint by ID"""
    try:
        checkpoint = sdk_instance.get_checkpoint(checkpoint_id)
        
        # Format metrics
        metrics = {}
        if hasattr(checkpoint.metrics, "train_loss") and checkpoint.metrics.train_loss is not None:
            metrics["train_loss"] = float(checkpoint.metrics.train_loss)
        if hasattr(checkpoint.metrics, "valid_loss") and checkpoint.metrics.valid_loss is not None:
            metrics["valid_loss"] = float(checkpoint.metrics.valid_loss)
            
        return CheckpointResponse(
            id=checkpoint.id,
            job_id=checkpoint.job_id,
            step_number=checkpoint.step_number,
            path=checkpoint.path,
            created_at=checkpoint.created_at,
            metrics=metrics
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting checkpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting checkpoint: {str(e)}"
        )

@app.get("/fine_tuning/jobs/{job_id}/checkpoints/latest", response_model=CheckpointResponse, tags=["Checkpoints"])
async def get_latest_checkpoint(job_id: str = Path(..., description="ID or name of the fine-tuning job"), 
                              api_key: str = Depends(authenticate_api_key)):
    """Get the latest checkpoint for a job"""
    try:
        checkpoint = sdk_instance.get_latest_checkpoint(job_id)
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No checkpoints found for job {job_id}"
            )
            
        # Format metrics
        metrics = {}
        if hasattr(checkpoint.metrics, "train_loss") and checkpoint.metrics.train_loss is not None:
            metrics["train_loss"] = float(checkpoint.metrics.train_loss)
        if hasattr(checkpoint.metrics, "valid_loss") and checkpoint.metrics.valid_loss is not None:
            metrics["valid_loss"] = float(checkpoint.metrics.valid_loss)
            
        return CheckpointResponse(
            id=checkpoint.id,
            job_id=checkpoint.job_id,
            step_number=checkpoint.step_number,
            path=checkpoint.path,
            created_at=checkpoint.created_at,
            metrics=metrics
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error getting latest checkpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting latest checkpoint: {str(e)}"
        )