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
    
    # Datasets
    def list_datasets(self) -> List[Dict]:
        """List all available datasets."""
        response = requests.get(f"{self.base_url}/datasets", headers=self.headers)
        return self._handle_response(response)
    
    def get_dataset(self, name: str) -> Dict:
        """
        Get dataset by name.
        
        Args:
            name: Name of the dataset
        """
        response = requests.get(f"{self.base_url}/datasets/{name}", headers=self.headers)
        return self._handle_response(response)
    
    def create_dataset(self, name: str, path: str, schema: str = "custom", metadata: Dict = None) -> Dict:
        """
        Create a new dataset.
        
        Args:
            name: Unique name for the dataset
            path: Path to the file or directory containing the data
            schema: Dataset schema type
            metadata: Additional metadata
        """
        payload = {
            "name": name,
            "path": path,
            "schema": schema
        }
        if metadata:
            payload["metadata"] = metadata
            
        response = requests.post(
            f"{self.base_url}/datasets", 
            headers=self.headers, 
            json=payload
        )
        return self._handle_response(response)
    
    def delete_dataset(self, name: str) -> Dict:
        """
        Delete dataset by name.
        
        Args:
            name: Name of the dataset
        """
        response = requests.delete(f"{self.base_url}/datasets/{name}", headers=self.headers)
        return self._handle_response(response)
    
    # Models
    def list_models(self) -> List[Dict]:
        """List all available models."""
        response = requests.get(f"{self.base_url}/models", headers=self.headers)
        return self._handle_response(response)
    
    def get_model(self, name: str) -> Dict:
        """
        Get model by name.
        
        Args:
            name: Name of the model
        """
        response = requests.get(f"{self.base_url}/models/{name}", headers=self.headers)
        return self._handle_response(response)
    
    def create_model(self, name: str, path: str, dataset_id: str, 
                    base_model: str = None, adapters: List[str] = None, 
                    hyperparameters: Dict = None) -> Dict:
        """
        Create a new model.
        
        Args:
            name: Unique name for the model
            path: Path to the file or directory containing the model
            dataset_id: ID or name of the dataset to use for training
            base_model: Name of the base model to use
            adapters: List of adapter names to apply
            hyperparameters: Hyperparameters for the model
        """
        payload = {
            "name": name,
            "path": path,
            "dataset_id": dataset_id
        }
        if base_model:
            payload["base_model"] = base_model
        if adapters:
            payload["adapters"] = adapters
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
            
        response = requests.post(
            f"{self.base_url}/models", 
            headers=self.headers, 
            json=payload
        )
        return self._handle_response(response)
    
    def delete_model(self, name: str) -> Dict:
        """
        Delete model by name.
        
        Args:
            name: Name of the model
        """
        response = requests.delete(f"{self.base_url}/models/{name}", headers=self.headers)
        return self._handle_response(response)
    
    # Adapters
    def list_adapters(self) -> List[Dict]:
        """List all available adapters."""
        response = requests.get(f"{self.base_url}/adapters", headers=self.headers)
        return self._handle_response(response)
    
    def get_adapter(self, name: str) -> Dict:
        """
        Get adapter by name.
        
        Args:
            name: Name of the adapter
        """
        response = requests.get(f"{self.base_url}/adapters/{name}", headers=self.headers)
        return self._handle_response(response)
    
    def create_adapter(self, name: str, model_id: str, dataset_id: str, 
                      adapter_type: str, path: str, hyperparameters: Dict = None) -> Dict:
        """
        Create a new adapter.
        
        Args:
            name: Unique name for the adapter
            model_id: ID or name of the model this adapter is for
            dataset_id: ID or name of the dataset used to train this adapter
            adapter_type: Type of adapter (e.g., "lora", "qlora")
            path: Path to store the adapter
            hyperparameters: Adapter hyperparameters
        """
        payload = {
            "name": name,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "adapter_type": adapter_type,
            "path": path
        }
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
            
        response = requests.post(
            f"{self.base_url}/adapters", 
            headers=self.headers, 
            json=payload
        )
        return self._handle_response(response)
    
    def delete_adapter(self, name: str) -> Dict:
        """
        Delete adapter by name.
        
        Args:
            name: Name of the adapter
        """
        response = requests.delete(f"{self.base_url}/adapters/{name}", headers=self.headers)
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
    
    def create_fine_tuning_job(self, job_name: str, base_model: str, dataset: str, 
                              job_type: str = "sft", hyperparameters: Dict = None) -> Dict:
        """
        Create a new fine-tuning job.
        
        Args:
            job_name: Unique name for the job
            base_model: Name of the base model to fine-tune
            dataset: Name of the dataset to use
            job_type: Type of fine-tuning job ("sft", "lora", "qlora", etc.)
            hyperparameters: Job hyperparameters
        """
        payload = {
            "job_name": job_name,
            "base_model": base_model,
            "dataset": dataset,
            "job_type": job_type
        }
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
    
    # Deployments
    def list_deployments(self) -> List[Dict]:
        """List all active deployments."""
        response = requests.get(f"{self.base_url}/deployments", headers=self.headers)
        return self._handle_response(response)
    
    def get_deployment(self, name: str) -> Dict:
        """
        Get deployment by name.
        
        Args:
            name: Name of the deployment
        """
        response = requests.get(f"{self.base_url}/deployments/{name}", headers=self.headers)
        return self._handle_response(response)
    
    def create_deployment(self, model_name: str, adapters: List[str] = None, 
                         deployment_name: str = None, merge: bool = False, 
                         environment: str = "development") -> Dict:
        """
        Create a new deployment.
        
        Args:
            model_name: Name of the model to deploy
            adapters: List of adapter names to apply
            deployment_name: Custom name for the deployment
            merge: Whether to merge adapters with the model
            environment: Deployment environment ("development", "staging", "production")
        """
        payload = {
            "model_name": model_name,
            "merge": merge,
            "environment": environment
        }
        if adapters:
            payload["adapters"] = adapters
        if deployment_name:
            payload["deployment_name"] = deployment_name
            
        response = requests.post(
            f"{self.base_url}/deployments", 
            headers=self.headers, 
            json=payload
        )
        return self._handle_response(response)
    
    def delete_deployment(self, name: str) -> Dict:
        """
        Deactivate and remove a deployment.
        
        Args:
            name: Name of the deployment
        """
        response = requests.delete(f"{self.base_url}/deployments/{name}", headers=self.headers)
        return self._handle_response(response)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="SDK API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the SDK API")
    parser.add_argument("--key", help="API key for authentication")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Health Check
    health_parser = subparsers.add_parser("health", help="Check API health")
    
    # Datasets
    dataset_parser = subparsers.add_parser("datasets", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="subcommand", help="Dataset subcommand")
    
    list_datasets_parser = dataset_subparsers.add_parser("list", help="List all datasets")
    
    get_dataset_parser = dataset_subparsers.add_parser("get", help="Get dataset by name")
    get_dataset_parser.add_argument("name", help="Dataset name")
    
    create_dataset_parser = dataset_subparsers.add_parser("create", help="Create a new dataset")
    create_dataset_parser.add_argument("name", help="Dataset name")
    create_dataset_parser.add_argument("path", help="Dataset path")
    create_dataset_parser.add_argument("--schema", default="custom", help="Dataset schema")
    
    delete_dataset_parser = dataset_subparsers.add_parser("delete", help="Delete dataset")
    delete_dataset_parser.add_argument("name", help="Dataset name")
    
    # Models
    model_parser = subparsers.add_parser("models", help="Model operations")
    model_subparsers = model_parser.add_subparsers(dest="subcommand", help="Model subcommand")
    
    list_models_parser = model_subparsers.add_parser("list", help="List all models")
    
    get_model_parser = model_subparsers.add_parser("get", help="Get model by name")
    get_model_parser.add_argument("name", help="Model name")
    
    create_model_parser = model_subparsers.add_parser("create", help="Create a new model")
    create_model_parser.add_argument("name", help="Model name")
    create_model_parser.add_argument("path", help="Model path")
    create_model_parser.add_argument("dataset_id", help="Dataset ID/name")
    create_model_parser.add_argument("--base_model", help="Base model name")
    create_model_parser.add_argument("--adapters", nargs="+", help="Adapter names")
    
    delete_model_parser = model_subparsers.add_parser("delete", help="Delete model")
    delete_model_parser.add_argument("name", help="Model name")
    
    # Adapters
    adapter_parser = subparsers.add_parser("adapters", help="Adapter operations")
    adapter_subparsers = adapter_parser.add_subparsers(dest="subcommand", help="Adapter subcommand")
    
    list_adapters_parser = adapter_subparsers.add_parser("list", help="List all adapters")
    
    get_adapter_parser = adapter_subparsers.add_parser("get", help="Get adapter by name")
    get_adapter_parser.add_argument("name", help="Adapter name")
    
    create_adapter_parser = adapter_subparsers.add_parser("create", help="Create a new adapter")
    create_adapter_parser.add_argument("name", help="Adapter name")
    create_adapter_parser.add_argument("model_id", help="Model ID/name")
    create_adapter_parser.add_argument("dataset_id", help="Dataset ID/name")
    create_adapter_parser.add_argument("adapter_type", help="Adapter type")
    create_adapter_parser.add_argument("path", help="Path to store adapter")
    
    delete_adapter_parser = adapter_subparsers.add_parser("delete", help="Delete adapter")
    delete_adapter_parser.add_argument("name", help="Adapter name")
    
    # Fine-Tuning Jobs
    ft_parser = subparsers.add_parser("finetuning", help="Fine-tuning operations")
    ft_subparsers = ft_parser.add_subparsers(dest="subcommand", help="Fine-tuning subcommand")
    
    list_ft_parser = ft_subparsers.add_parser("list", help="List all fine-tuning jobs")
    list_ft_parser.add_argument("--limit", type=int, default=20, help="Max jobs to return")
    list_ft_parser.add_argument("--after", help="Return jobs after this ID")
    
    get_ft_parser = ft_subparsers.add_parser("get", help="Get fine-tuning job by name")
    get_ft_parser.add_argument("job_name", help="Job name")
    
    create_ft_parser = ft_subparsers.add_parser("create", help="Create a new fine-tuning job")
    create_ft_parser.add_argument("job_name", help="Job name")
    create_ft_parser.add_argument("base_model", help="Base model name")
    create_ft_parser.add_argument("dataset", help="Dataset name")
    create_ft_parser.add_argument("--job_type", default="sft", help="Job type")
    
    cancel_ft_parser = ft_subparsers.add_parser("cancel", help="Cancel a fine-tuning job")
    cancel_ft_parser.add_argument("job_name", help="Job name")
    
    pause_ft_parser = ft_subparsers.add_parser("pause", help="Pause a fine-tuning job")
    pause_ft_parser.add_argument("job_name", help="Job name")
    
    resume_ft_parser = ft_subparsers.add_parser("resume", help="Resume a fine-tuning job")
    resume_ft_parser.add_argument("job_name", help="Job name")
    
    # Deployments
    deploy_parser = subparsers.add_parser("deployments", help="Deployment operations")
    deploy_subparsers = deploy_parser.add_subparsers(dest="subcommand", help="Deployment subcommand")
    
    list_deploy_parser = deploy_subparsers.add_parser("list", help="List all deployments")
    
    get_deploy_parser = deploy_subparsers.add_parser("get", help="Get deployment by name")
    get_deploy_parser.add_argument("name", help="Deployment name")
    
    create_deploy_parser = deploy_subparsers.add_parser("create", help="Create a new deployment")
    create_deploy_parser.add_argument("model_name", help="Model name")
    create_deploy_parser.add_argument("--adapters", nargs="+", help="Adapter names")
    create_deploy_parser.add_argument("--deployment_name", help="Custom deployment name")
    create_deploy_parser.add_argument("--merge", action="store_true", help="Merge adapters with model")
    create_deploy_parser.add_argument("--environment", default="development", 
                                    choices=["development", "staging", "production"], 
                                    help="Deployment environment")
    
    delete_deploy_parser = deploy_subparsers.add_parser("delete", help="Delete deployment")
    delete_deploy_parser.add_argument("name", help="Deployment name")
    
    args = parser.parse_args()
    
    # Create the client
    client = SDKAPIClient(base_url=args.url, api_key=args.key)
    
    if args.command == "health":
        pprint(client.health_check())
    
    elif args.command == "datasets":
        if args.subcommand == "list":
            pprint(client.list_datasets())
        elif args.subcommand == "get":
            pprint(client.get_dataset(args.name))
        elif args.subcommand == "create":
            pprint(client.create_dataset(args.name, args.path, args.schema))
        elif args.subcommand == "delete":
            pprint(client.delete_dataset(args.name))
    
    elif args.command == "models":
        if args.subcommand == "list":
            pprint(client.list_models())
        elif args.subcommand == "get":
            pprint(client.get_model(args.name))
        elif args.subcommand == "create":
            pprint(client.create_model(
                args.name, args.path, args.dataset_id, 
                args.base_model, args.adapters
            ))
        elif args.subcommand == "delete":
            pprint(client.delete_model(args.name))
    
    elif args.command == "adapters":
        if args.subcommand == "list":
            pprint(client.list_adapters())
        elif args.subcommand == "get":
            pprint(client.get_adapter(args.name))
        elif args.subcommand == "create":
            pprint(client.create_adapter(
                args.name, args.model_id, args.dataset_id,
                args.adapter_type, args.path
            ))
        elif args.subcommand == "delete":
            pprint(client.delete_adapter(args.name))
    
    elif args.command == "finetuning":
        if args.subcommand == "list":
            pprint(client.list_fine_tuning_jobs(args.limit, args.after))
        elif args.subcommand == "get":
            pprint(client.get_fine_tuning_job(args.job_name))
        elif args.subcommand == "create":
            pprint(client.create_fine_tuning_job(
                args.job_name, args.base_model, args.dataset, args.job_type
            ))
        elif args.subcommand == "cancel":
            pprint(client.cancel_fine_tuning_job(args.job_name))
        elif args.subcommand == "pause":
            pprint(client.pause_fine_tuning_job(args.job_name))
        elif args.subcommand == "resume":
            pprint(client.resume_fine_tuning_job(args.job_name))
    
    elif args.command == "deployments":
        if args.subcommand == "list":
            pprint(client.list_deployments())
        elif args.subcommand == "get":
            pprint(client.get_deployment(args.name))
        elif args.subcommand == "create":
            pprint(client.create_deployment(
                args.model_name, args.adapters, args.deployment_name, 
                args.merge, args.environment
            ))
        elif args.subcommand == "delete":
            pprint(client.delete_deployment(args.name))


if __name__ == "__main__":
    main()