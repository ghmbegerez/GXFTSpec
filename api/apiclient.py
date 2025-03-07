#!/usr/bin/env python3
"""
SDK API Client

A client for testing the SDK API endpoints.
"""

import os
import json
import argparse
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pprint import pprint


class SDKAPIClient:
    """Client for interacting with the SDK API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the SDK API client.
        
        Args:
            base_url: Base URL of the SDK API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("SDK_API_KEY", "test-api-key")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle API response and errors."""
        try:
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if response.content:
                print(f"Response: {response.json()}")
            return {}
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    # Health Check
    def health_check(self) -> Dict:
        """Check the API health status."""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        return self._handle_response(response)
    
    # Hub Models
    def list_hub_models(self) -> List[Dict]:
        """List all models in the hub."""
        response = requests.get(f"{self.base_url}/hub/models", headers=self.headers)
        return self._handle_response(response)
    
    def get_hub_model(self, name: str) -> Dict:
        """
        Get a model from the hub by name.
        
        Args:
            name: Name of the model
        """
        response = requests.get(f"{self.base_url}/hub/models/{name}", headers=self.headers)
        return self._handle_response(response)
    
    def create_hub_model(self, name: str, model_type: str, path: str, 
                        description: str = None, base_model_id: str = None) -> Dict:
        """
        Create a new model in the hub.
        
        Args:
            name: Unique name for the model
            model_type: Type of model (base or fine-tuned)
            path: Path to model files
            description: Optional description
            base_model_id: Base model ID for fine-tuned models
        """
        payload = {
            "name": name,
            "model_type": model_type,
            "path": path,
        }
        if description:
            payload["description"] = description
        if base_model_id:
            payload["base_model_id"] = base_model_id
            
        response = requests.post(
            f"{self.base_url}/hub/models", 
            headers=self.headers, 
            json=payload
        )
        return self._handle_response(response)
    
    def delete_hub_model(self, name: str) -> Dict:
        """
        Delete a model from the hub.
        
        Args:
            name: Name of the model
        """
        response = requests.delete(f"{self.base_url}/hub/models/{name}", headers=self.headers)
        return self._handle_response(response)
    
    # Hub Adapters
    def list_hub_adapters(self) -> List[Dict]:
        """List all adapters in the hub."""
        response = requests.get(f"{self.base_url}/hub/adapters", headers=self.headers)
        return self._handle_response(response)
    
    def get_hub_adapter(self, name: str) -> Dict:
        """
        Get an adapter from the hub by name.
        
        Args:
            name: Name of the adapter
        """
        response = requests.get(f"{self.base_url}/hub/adapters/{name}", headers=self.headers)
        return self._handle_response(response)
    
    def create_hub_adapter(self, name: str, adapter_type: str, model_id: str, 
                         dataset: str, path: str, description: str = None, 
                         hyperparameters: Dict = None) -> Dict:
        """
        Create a new adapter in the hub.
        
        Args:
            name: Unique name for the adapter
            adapter_type: Type of adapter (lora, qlora)
            model_id: ID of the associated model
            dataset: Dataset used for training
            path: Path to adapter files
            description: Optional adapter description
            hyperparameters: Optional training hyperparameters
        """
        payload = {
            "name": name,
            "adapter_type": adapter_type,
            "model_id": model_id,
            "dataset": dataset,
            "path": path
        }
        if description:
            payload["description"] = description
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
            
        response = requests.post(
            f"{self.base_url}/hub/adapters", 
            headers=self.headers, 
            json=payload
        )
        return self._handle_response(response)
    
    def delete_hub_adapter(self, name: str) -> Dict:
        """
        Delete an adapter from the hub.
        
        Args:
            name: Name of the adapter
        """
        response = requests.delete(f"{self.base_url}/hub/adapters/{name}", headers=self.headers)
        return self._handle_response(response)
    
    # Fine-Tuning Jobs
    def list_fine_tuning_jobs(self, limit: int = 20, after: str = None) -> List[Dict]:
        """
        List all fine-tuning jobs.
        
        Args:
            limit: Maximum number of jobs to return
            after: Return jobs after this job ID
        """
        params = {"limit": limit}
        if after:
            params["after"] = after
            
        response = requests.get(
            f"{self.base_url}/fine_tuning/jobs", 
            headers=self.headers,
            params=params
        )
        return self._handle_response(response)
    
    def get_fine_tuning_job(self, job_name: str) -> Dict:
        """
        Get a fine-tuning job by name.
        
        Args:
            job_name: Name of the fine-tuning job
        """
        response = requests.get(f"{self.base_url}/fine_tuning/jobs/{job_name}", headers=self.headers)
        return self._handle_response(response)
    
    def create_fine_tuning_job(self, name: str, base_model: str, dataset: str, 
                             val_dataset: str = None, job_type: str = "sft", 
                             output_model_name: str = None, hyperparameters: Dict = None) -> Dict:
        """
        Create a new fine-tuning job.
        
        Args:
            name: Unique name for the job
            base_model: Name or path of base model to fine-tune
            dataset: Path to training dataset
            val_dataset: Optional path to validation dataset
            job_type: Type of fine-tuning (sft, lora, qlora)
            output_model_name: Optional name for output model
            hyperparameters: Optional hyperparameters for training
        """
        payload = {
            "name": name,
            "base_model": base_model,
            "dataset": dataset,
            "job_type": job_type
        }
        if val_dataset:
            payload["val_dataset"] = val_dataset
        if output_model_name:
            payload["output_model_name"] = output_model_name
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
            
        response = requests.post(
            f"{self.base_url}/fine_tuning/jobs", 
            headers=self.headers, 
            json=payload
        )
        return self._handle_response(response)
    
    def cancel_fine_tuning_job(self, job_name: str) -> Dict:
        """
        Cancel a fine-tuning job.
        
        Args:
            job_name: Name of the fine-tuning job to cancel
        """
        response = requests.delete(f"{self.base_url}/fine_tuning/jobs/{job_name}", headers=self.headers)
        return self._handle_response(response)
    
    def pause_fine_tuning_job(self, job_name: str) -> Dict:
        """
        Pause a fine-tuning job.
        
        Args:
            job_name: Name of the fine-tuning job to pause
        """
        response = requests.post(f"{self.base_url}/fine_tuning/jobs/{job_name}/pause", headers=self.headers)
        return self._handle_response(response)
    
    def resume_fine_tuning_job(self, job_name: str) -> Dict:
        """
        Resume a fine-tuning job.
        
        Args:
            job_name: Name of the fine-tuning job to resume
        """
        response = requests.post(f"{self.base_url}/fine_tuning/jobs/{job_name}/resume", headers=self.headers)
        return self._handle_response(response)
    
    # Checkpoints
    def list_checkpoints(self, job_id: str = None) -> List[Dict]:
        """
        List all checkpoints, optionally filtered by job.
        
        Args:
            job_id: Optional job ID to filter by
        """
        params = {}
        if job_id:
            params["job_id"] = job_id
            
        response = requests.get(
            f"{self.base_url}/checkpoints", 
            headers=self.headers,
            params=params
        )
        return self._handle_response(response)
    
    def get_checkpoint(self, checkpoint_id: str) -> Dict:
        """
        Get a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of the checkpoint
        """
        response = requests.get(f"{self.base_url}/checkpoints/{checkpoint_id}", headers=self.headers)
        return self._handle_response(response)
    
    def get_latest_checkpoint(self, job_id: str) -> Dict:
        """
        Get the latest checkpoint for a job.
        
        Args:
            job_id: ID or name of the fine-tuning job
        """
        response = requests.get(f"{self.base_url}/fine_tuning/jobs/{job_id}/checkpoints/latest", 
                               headers=self.headers)
        return self._handle_response(response)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Fine-Tuning SDK API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the SDK API")
    parser.add_argument("--key", help="API key for authentication")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Health Check
    health_parser = subparsers.add_parser("health", help="Check API health")
    
    # Hub Models
    hub_models_parser = subparsers.add_parser("hub-models", help="Hub model operations")
    hub_models_subparsers = hub_models_parser.add_subparsers(dest="subcommand", help="Hub model subcommand")
    
    list_hub_models_parser = hub_models_subparsers.add_parser("list", help="List all models in the hub")
    
    get_hub_model_parser = hub_models_subparsers.add_parser("get", help="Get a model from the hub by name")
    get_hub_model_parser.add_argument("name", help="Model name")
    
    create_hub_model_parser = hub_models_subparsers.add_parser("create", help="Create a new model in the hub")
    create_hub_model_parser.add_argument("name", help="Model name")
    create_hub_model_parser.add_argument("model_type", choices=["base", "fine-tuned"], help="Type of model")
    create_hub_model_parser.add_argument("path", help="Path to model files")
    create_hub_model_parser.add_argument("--description", help="Model description")
    create_hub_model_parser.add_argument("--base-model-id", help="Base model ID for fine-tuned models")
    
    delete_hub_model_parser = hub_models_subparsers.add_parser("delete", help="Delete a model from the hub")
    delete_hub_model_parser.add_argument("name", help="Model name")
    
    # Hub Adapters
    hub_adapters_parser = subparsers.add_parser("hub-adapters", help="Hub adapter operations")
    hub_adapters_subparsers = hub_adapters_parser.add_subparsers(dest="subcommand", help="Hub adapter subcommand")
    
    list_hub_adapters_parser = hub_adapters_subparsers.add_parser("list", help="List all adapters in the hub")
    
    get_hub_adapter_parser = hub_adapters_subparsers.add_parser("get", help="Get an adapter from the hub by name")
    get_hub_adapter_parser.add_argument("name", help="Adapter name")
    
    create_hub_adapter_parser = hub_adapters_subparsers.add_parser("create", help="Create a new adapter in the hub")
    create_hub_adapter_parser.add_argument("name", help="Adapter name")
    create_hub_adapter_parser.add_argument("adapter_type", choices=["lora", "qlora"], help="Type of adapter")
    create_hub_adapter_parser.add_argument("model_id", help="ID of the associated model")
    create_hub_adapter_parser.add_argument("dataset", help="Dataset used for training")
    create_hub_adapter_parser.add_argument("path", help="Path to adapter files")
    create_hub_adapter_parser.add_argument("--description", help="Adapter description")
    
    delete_hub_adapter_parser = hub_adapters_subparsers.add_parser("delete", help="Delete an adapter from the hub")
    delete_hub_adapter_parser.add_argument("name", help="Adapter name")
    
    # Fine-Tuning Jobs
    ft_parser = subparsers.add_parser("fine-tuning", help="Fine-tuning operations")
    ft_subparsers = ft_parser.add_subparsers(dest="subcommand", help="Fine-tuning subcommand")
    
    list_ft_parser = ft_subparsers.add_parser("list", help="List all fine-tuning jobs")
    list_ft_parser.add_argument("--limit", type=int, default=20, help="Max jobs to return")
    list_ft_parser.add_argument("--after", help="Return jobs after this ID")
    
    get_ft_parser = ft_subparsers.add_parser("get", help="Get fine-tuning job by name")
    get_ft_parser.add_argument("name", help="Job name")
    
    create_ft_parser = ft_subparsers.add_parser("create", help="Create a new fine-tuning job")
    create_ft_parser.add_argument("name", help="Job name")
    create_ft_parser.add_argument("base_model", help="Base model name or path")
    create_ft_parser.add_argument("dataset", help="Path to training dataset")
    create_ft_parser.add_argument("--val-dataset", help="Path to validation dataset")
    create_ft_parser.add_argument("--job-type", choices=["sft", "lora", "qlora"], default="sft", help="Job type")
    create_ft_parser.add_argument("--output-model-name", help="Name for the output model")
    
    cancel_ft_parser = ft_subparsers.add_parser("cancel", help="Cancel a fine-tuning job")
    cancel_ft_parser.add_argument("name", help="Job name")
    
    pause_ft_parser = ft_subparsers.add_parser("pause", help="Pause a fine-tuning job")
    pause_ft_parser.add_argument("name", help="Job name")
    
    resume_ft_parser = ft_subparsers.add_parser("resume", help="Resume a fine-tuning job")
    resume_ft_parser.add_argument("name", help="Job name")
    
    # Checkpoints
    checkpoint_parser = subparsers.add_parser("checkpoints", help="Checkpoint operations")
    checkpoint_subparsers = checkpoint_parser.add_subparsers(dest="subcommand", help="Checkpoint subcommand")
    
    list_checkpoint_parser = checkpoint_subparsers.add_parser("list", help="List all checkpoints")
    list_checkpoint_parser.add_argument("--job-id", help="Filter by job ID")
    
    get_checkpoint_parser = checkpoint_subparsers.add_parser("get", help="Get checkpoint by ID")
    get_checkpoint_parser.add_argument("checkpoint_id", help="Checkpoint ID")
    
    get_latest_checkpoint_parser = checkpoint_subparsers.add_parser("latest", help="Get latest checkpoint for a job")
    get_latest_checkpoint_parser.add_argument("job_id", help="Job ID or name")
    
    args = parser.parse_args()
    
    # Create the client
    client = SDKAPIClient(base_url=args.url, api_key=args.key)
    
    if args.command == "health":
        pprint(client.health_check())
    
    elif args.command == "hub-models":
        if args.subcommand == "list":
            pprint(client.list_hub_models())
        elif args.subcommand == "get":
            pprint(client.get_hub_model(args.name))
        elif args.subcommand == "create":
            pprint(client.create_hub_model(
                args.name, args.model_type, args.path, 
                args.description, args.base_model_id
            ))
        elif args.subcommand == "delete":
            pprint(client.delete_hub_model(args.name))
    
    elif args.command == "hub-adapters":
        if args.subcommand == "list":
            pprint(client.list_hub_adapters())
        elif args.subcommand == "get":
            pprint(client.get_hub_adapter(args.name))
        elif args.subcommand == "create":
            pprint(client.create_hub_adapter(
                args.name, args.adapter_type, args.model_id,
                args.dataset, args.path, args.description
            ))
        elif args.subcommand == "delete":
            pprint(client.delete_hub_adapter(args.name))
    
    elif args.command == "fine-tuning":
        if args.subcommand == "list":
            pprint(client.list_fine_tuning_jobs(args.limit, args.after))
        elif args.subcommand == "get":
            pprint(client.get_fine_tuning_job(args.name))
        elif args.subcommand == "create":
            pprint(client.create_fine_tuning_job(
                args.name, args.base_model, args.dataset, 
                args.val_dataset, args.job_type, args.output_model_name
            ))
        elif args.subcommand == "cancel":
            pprint(client.cancel_fine_tuning_job(args.name))
        elif args.subcommand == "pause":
            pprint(client.pause_fine_tuning_job(args.name))
        elif args.subcommand == "resume":
            pprint(client.resume_fine_tuning_job(args.name))
    
    elif args.command == "checkpoints":
        if args.subcommand == "list":
            pprint(client.list_checkpoints(args.job_id))
        elif args.subcommand == "get":
            pprint(client.get_checkpoint(args.checkpoint_id))
        elif args.subcommand == "latest":
            pprint(client.get_latest_checkpoint(args.job_id))


if __name__ == "__main__":
    main()