#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifiedFineTune - A CLI utility to standardize fine-tuning across multiple frameworks.

This tool provides a unified interface for fine-tuning models using various frameworks:
- LlamaFactory
- PyTorch with PEFT
- Axolotl
- AutoTrain Advanced

Author: Claude
Date: February 21, 2025
"""
    
import os
import sys
import subprocess
import logging
from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from pydantic import BaseModel, Field, validator, root_validator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("unified-finetune")
console = Console()

# Constants
DEFAULT_OUTPUT_DIR = "./outputs"
FRAMEWORKS = ["llamafactory", "pytorch", "axolotl", "autotrain"]

# ===================== Models =====================

class Framework(str, Enum):
    LLAMAFACTORY = "llamafactory"
    PYTORCH = "pytorch"
    AXOLOTL = "axolotl"
    AUTOTRAIN = "autotrain"


class QuantizationMode(str, Enum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"


class FineTuningConfig(BaseModel):
    """Configuration for fine-tuning jobs across all frameworks."""
    job_name: str = Field(..., description="Name of the fine-tuning job")
    base_model: str = Field(..., description="Base model to fine-tune")
    dataset: str = Field(..., description="Dataset path or name")
    epochs: int = Field(3, description="Number of training epochs")
    lora_rank: int = Field(8, description="Rank of LoRA adapters")
    lora_alpha: float = Field(16.0, description="Alpha parameter for LoRA")
    lora_dropout: float = Field(0.05, description="Dropout probability for LoRA")
    quantized: Union[bool, QuantizationMode] = Field(False, description="Whether to use quantization")
    output_dir: str = Field(DEFAULT_OUTPUT_DIR, description="Directory to save outputs")
    framework: Framework = Field(..., description="Framework to use for fine-tuning")
    batch_size: int = Field(4, description="Batch size for training")
    learning_rate: float = Field(2e-5, description="Learning rate")
    max_length: int = Field(2048, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(1, description="Number of gradient accumulation steps")
    additional_args: Dict[str, Any] = Field(default_factory=dict, description="Additional framework-specific args")

    @validator('quantized', pre=True)
    def parse_quantization(cls, v):
        if isinstance(v, bool):
            return QuantizationMode.INT8 if v else QuantizationMode.NONE
        return v

    @validator('output_dir')
    def create_output_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v

    class Config:
        extra = "forbid"
        validate_assignment = True


class DeployConfig(BaseModel):
    """Configuration for model deployment."""
    model_id: str = Field(..., description="ID of the model to deploy")
    adapters: Optional[List[str]] = Field(None, description="List of adapter paths")
    output_dir: str = Field("./deployed_models", description="Directory to save deployed model")
    framework: Framework = Field(..., description="Framework used for training")

    @validator('output_dir')
    def create_output_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v


class JobInfo(BaseModel):
    """Information about a fine-tuning job."""
    job_id: str
    framework: Framework
    status: str
    created_at: str
    base_model: str
    dataset: str


# ===================== Command Generators =====================

class CommandGenerator:
    """Base class for generating commands for each framework."""
    
    @staticmethod
    def create_job(config: FineTuningConfig) -> List[str]:
        """Generate command to create a fine-tuning job."""
        raise NotImplementedError()
    
    @staticmethod
    def list_jobs(limit: int = 20, after: Optional[str] = None) -> List[str]:
        """Generate command to list fine-tuning jobs."""
        raise NotImplementedError()
    
    @staticmethod
    def get_job(job_id: str) -> List[str]:
        """Generate command to get details of a fine-tuning job."""
        raise NotImplementedError()
    
    @staticmethod
    def deploy_model(config: DeployConfig) -> List[str]:
        """Generate command to deploy a model."""
        raise NotImplementedError()


class LlamaFactoryCommandGenerator(CommandGenerator):
    """Command generator for LlamaFactory."""
    
    @staticmethod
    def create_job(config: FineTuningConfig) -> List[str]:
        cmd = [
            "python", "-m", "llamafactory.trainer",
            "--model_name_or_path", config.base_model,
            "--dataset_name", config.dataset,
            "--output_dir", f"{config.output_dir}/{config.job_name}",
            "--num_train_epochs", str(config.epochs),
            "--lora_rank", str(config.lora_rank),
            "--lora_alpha", str(config.lora_alpha),
            "--lora_dropout", str(config.lora_dropout),
            "--per_device_train_batch_size", str(config.batch_size),
            "--learning_rate", str(config.learning_rate),
            "--max_seq_length", str(config.max_length),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--do_train"
        ]
        
        if config.quantized != QuantizationMode.NONE:
            cmd.extend(["--quantization", config.quantized.value])
            
        # Add any additional args
        for k, v in config.additional_args.items():
            cmd.extend([f"--{k}", str(v)])
            
        return cmd
    
    @staticmethod
    def list_jobs(limit: int = 20, after: Optional[str] = None) -> List[str]:
        return ["ls", "-la", DEFAULT_OUTPUT_DIR]
    
    @staticmethod
    def get_job(job_id: str) -> List[str]:
        return ["ls", "-la", f"{DEFAULT_OUTPUT_DIR}/{job_id}"]
    
    @staticmethod
    def deploy_model(config: DeployConfig) -> List[str]:
        cmd = [
            "python", "-m", "llamafactory.deploy",
            "--model_name_or_path", f"{DEFAULT_OUTPUT_DIR}/{config.model_id}"
        ]
        
        if config.adapters:
            adapter_paths = ','.join(config.adapters)
            cmd.extend(["--adapter_name_or_path", adapter_paths])
            
        cmd.extend(["--export_dir", f"{config.output_dir}/{config.model_id}"])
        return cmd


class PyTorchCommandGenerator(CommandGenerator):
    """Command generator for PyTorch with PEFT."""
    
    @staticmethod
    def create_job(config: FineTuningConfig) -> List[str]:
        cmd = [
            "python", "-m", "torch.distributed.run",
            "--nproc_per_node=1", "train.py",
            "--model_name_or_path", config.base_model,
            "--train_file", config.dataset,
            "--output_dir", f"{config.output_dir}/{config.job_name}",
            "--num_train_epochs", str(config.epochs),
            "--per_device_train_batch_size", str(config.batch_size),
            "--learning_rate", str(config.learning_rate),
            "--max_seq_length", str(config.max_length),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--lora_rank", str(config.lora_rank),
            "--lora_alpha", str(config.lora_alpha),
            "--lora_dropout", str(config.lora_dropout)
        ]
        
        if config.quantized != QuantizationMode.NONE:
            cmd.append("--use_8bit_quantization")
            
        # Add any additional args
        for k, v in config.additional_args.items():
            cmd.extend([f"--{k}", str(v)])
            
        return cmd
    
    @staticmethod
    def list_jobs(limit: int = 20, after: Optional[str] = None) -> List[str]:
        return ["find", DEFAULT_OUTPUT_DIR, "-type", "d", "-name", "checkpoint-*", "|", "sort"]
    
    @staticmethod
    def get_job(job_id: str) -> List[str]:
        return ["ls", "-la", f"{DEFAULT_OUTPUT_DIR}/{job_id}"]
    
    @staticmethod
    def deploy_model(config: DeployConfig) -> List[str]:
        cmd = [
            "python", "merge_adapter.py",
            "--base_model_path", config.model_id
        ]
        
        if config.adapters:
            adapter_paths = ','.join(config.adapters)
            cmd.extend(["--adapter_path", adapter_paths])
            
        cmd.extend(["--output_path", f"{config.output_dir}/{config.model_id}_merged"])
        return cmd


class AxolotlCommandGenerator(CommandGenerator):
    """Command generator for Axolotl."""
    
    @staticmethod
    def create_job(config: FineTuningConfig) -> List[str]:
        cmd = [
            "accelerate", "launch", "-m", "axolotl.cli.train",
            "--config", f"configs/{config.job_name}.yml",
            "--model", config.base_model,
            "--datasets", config.dataset,
            "--epochs", str(config.epochs),
            "--lora_r", str(config.lora_rank),
            "--lora_alpha", str(config.lora_alpha),
            "--lora_dropout", str(config.lora_dropout),
            "--batch_size", str(config.batch_size),
            "--lr", str(config.learning_rate),
            "--sequence_len", str(config.max_length),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps)
        ]
        
        if config.quantized != QuantizationMode.NONE:
            cmd.extend(["--quantization", config.quantized.value])
            
        # Add any additional args
        for k, v in config.additional_args.items():
            cmd.extend([f"--{k}", str(v)])
            
        return cmd
    
    @staticmethod
    def list_jobs(limit: int = 20, after: Optional[str] = None) -> List[str]:
        return ["ls", "-la", f"{DEFAULT_OUTPUT_DIR}/"]
    
    @staticmethod
    def get_job(job_id: str) -> List[str]:
        return ["ls", "-la", f"{DEFAULT_OUTPUT_DIR}/{job_id}"]
    
    @staticmethod
    def deploy_model(config: DeployConfig) -> List[str]:
        cmd = [
            "python", "-m", "axolotl.cli.merge_lora",
            "--base_model", config.model_id
        ]
        
        if config.adapters:
            adapter_paths = ','.join(config.adapters)
            cmd.extend(["--lora_adapter", adapter_paths])
            
        cmd.extend(["--output_dir", f"{config.output_dir}/{config.model_id}"])
        return cmd


class AutoTrainCommandGenerator(CommandGenerator):
    """Command generator for AutoTrain Advanced."""
    
    @staticmethod
    def create_job(config: FineTuningConfig) -> List[str]:
        cmd = [
            "autotrain", "finetune",
            "--name", config.job_name,
            "--model", config.base_model,
            "--train-data", config.dataset,
            "--epochs", str(config.epochs),
            "--lora-r", str(config.lora_rank),
            "--lora-alpha", str(config.lora_alpha),
            "--lora-dropout", str(config.lora_dropout),
            "--batch-size", str(config.batch_size),
            "--learning-rate", str(config.learning_rate),
            "--max-seq-length", str(config.max_length),
            "--gradient-accumulation", str(config.gradient_accumulation_steps),
            "--project-name", config.job_name
        ]
        
        if config.quantized != QuantizationMode.NONE:
            cmd.extend(["--quantized", config.quantized.value])
            
        # Add any additional args
        for k, v in config.additional_args.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
            
        return cmd
    
    @staticmethod
    def list_jobs(limit: int = 20, after: Optional[str] = None) -> List[str]:
        cmd = ["autotrain", "projects", "list", "--limit", str(limit)]
        if after:
            cmd.extend(["--after", after])
        return cmd
    
    @staticmethod
    def get_job(job_id: str) -> List[str]:
        return ["autotrain", "projects", "get", "--name", job_id]
    
    @staticmethod
    def deploy_model(config: DeployConfig) -> List[str]:
        cmd = [
            "autotrain", "export",
            "--model", config.model_id
        ]
        
        if config.adapters:
            adapter_paths = ','.join(config.adapters)
            cmd.extend(["--adapter", adapter_paths])
            
        cmd.extend(["--output-dir", f"{config.output_dir}/{config.model_id}"])
        return cmd


# Factory to get the appropriate command generator
def get_command_generator(framework: Framework) -> CommandGenerator:
    """Factory function to get the appropriate command generator."""
    mapping = {
        Framework.LLAMAFACTORY: LlamaFactoryCommandGenerator,
        Framework.PYTORCH: PyTorchCommandGenerator,
        Framework.AXOLOTL: AxolotlCommandGenerator,
        Framework.AUTOTRAIN: AutoTrainCommandGenerator
    }
    return mapping[framework]


# ===================== CLI App =====================

app = typer.Typer(help="Unified Fine-Tuning CLI for multiple ML frameworks")


@app.command("create")
def create_job(
    job_name: str = typer.Option(..., help="Name of the fine-tuning job"),
    base_model: str = typer.Option(..., help="Base model to fine-tune"),
    dataset: str = typer.Option(..., help="Dataset path or name"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    lora_rank: int = typer.Option(8, help="Rank of LoRA adapters"),
    lora_alpha: float = typer.Option(16.0, help="Alpha parameter for LoRA"),
    lora_dropout: float = typer.Option(0.05, help="Dropout probability for LoRA"),
    quantized: str = typer.Option("none", help="Quantization mode: none, int8, int4"),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory to save outputs"),
    framework: Framework = typer.Option(..., help="Framework to use for fine-tuning"),
    batch_size: int = typer.Option(4, help="Batch size for training"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
    max_length: int = typer.Option(2048, help="Maximum sequence length"),
    gradient_accumulation_steps: int = typer.Option(1, help="Number of gradient accumulation steps"),
    dry_run: bool = typer.Option(False, help="Print command without executing"),
    additional_args: Optional[List[str]] = typer.Option(None, help="Additional framework-specific args in format key=value")
):
    """Create a new fine-tuning job."""
    # Parse additional args
    additional_args_dict = {}
    if additional_args:
        for arg in additional_args:
            try:
                key, value = arg.split("=", 1)
                additional_args_dict[key] = value
            except ValueError:
                log.warning(f"Ignoring malformed additional argument: {arg}")
    
    # Create config
    config = FineTuningConfig(
        job_name=job_name,
        base_model=base_model,
        dataset=dataset,
        epochs=epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantized=quantized,
        output_dir=output_dir,
        framework=framework,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        additional_args=additional_args_dict
    )
    
    # Get command
    cmd_gen = get_command_generator(framework)
    cmd = cmd_gen.create_job(config)
    
    # Execute or print
    console.print(f"[bold green]Running command:[/bold green] {' '.join(cmd)}")
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
            console.print(f"[bold green]Successfully created fine-tuning job '{job_name}'[/bold green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error creating fine-tuning job: {e}[/bold red]")
            sys.exit(1)


@app.command("list")
def list_jobs(
    framework: Framework = typer.Option(..., help="Framework to use"),
    limit: int = typer.Option(20, help="Maximum number of jobs to list"),
    after: Optional[str] = typer.Option(None, help="List jobs after this job ID"),
    dry_run: bool = typer.Option(False, help="Print command without executing")
):
    """List fine-tuning jobs."""
    cmd_gen = get_command_generator(framework)
    cmd = cmd_gen.list_jobs(limit, after)
    
    console.print(f"[bold green]Running command:[/bold green] {' '.join(cmd)}")
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error listing fine-tuning jobs: {e}[/bold red]")
            sys.exit(1)


@app.command("get")
def get_job(
    job_id: str = typer.Option(..., help="ID of the job to get"),
    framework: Framework = typer.Option(..., help="Framework to use"),
    dry_run: bool = typer.Option(False, help="Print command without executing")
):
    """Get details of a specific fine-tuning job."""
    cmd_gen = get_command_generator(framework)
    cmd = cmd_gen.get_job(job_id)
    
    console.print(f"[bold green]Running command:[/bold green] {' '.join(cmd)}")
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error getting fine-tuning job: {e}[/bold red]")
            sys.exit(1)


@app.command("deploy")
def deploy_model(
    model_id: str = typer.Option(..., help="ID of the model to deploy"),
    framework: Framework = typer.Option(..., help="Framework to use"),
    adapters: Optional[List[str]] = typer.Option(None, help="List of adapter paths"),
    output_dir: str = typer.Option("./deployed_models", help="Directory to save deployed model"),
    dry_run: bool = typer.Option(False, help="Print command without executing")
):
    """Deploy a fine-tuned model."""
    config = DeployConfig(
        model_id=model_id,
        adapters=adapters,
        output_dir=output_dir,
        framework=framework
    )
    
    cmd_gen = get_command_generator(framework)
    cmd = cmd_gen.deploy_model(config)
    
    console.print(f"[bold green]Running command:[/bold green] {' '.join(cmd)}")
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
            console.print(f"[bold green]Successfully deployed model '{model_id}'[/bold green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error deploying model: {e}[/bold red]")
            sys.exit(1)


@app.command("info")
def show_info():
    """Show information about supported frameworks and commands."""
    table = Table(title="Supported Frameworks")
    table.add_column("Framework", style="cyan")
    table.add_column("Description", style="green")
    
    table.add_row("llamafactory", "LlamaFactory: Framework for fine-tuning LLMs")
    table.add_row("pytorch", "PyTorch with PEFT: Standard PyTorch with parameter-efficient fine-tuning")
    table.add_row("axolotl", "Axolotl: Fine-tuning LLMs with a focus on performance and efficiency")
    table.add_row("autotrain", "AutoTrain Advanced: Hugging Face's AutoTrain for easy fine-tuning")
    
    console.print(table)
    
    console.print("\n[bold]Available Commands:[/bold]")
    console.print("  [cyan]create[/cyan]: Create a new fine-tuning job")
    console.print("  [cyan]list[/cyan]: List existing fine-tuning jobs")
    console.print("  [cyan]get[/cyan]: Get details of a specific fine-tuning job")
    console.print("  [cyan]deploy[/cyan]: Deploy a fine-tuned model")
    console.print("  [cyan]info[/cyan]: Show this information\n")
    
    console.print("[bold]Example:[/bold]")
    console.print("  unified-finetune create --job-name my-job --base-model llama2-7b --dataset my-dataset --framework llamafactory")


if __name__ == "__main__":
    app()