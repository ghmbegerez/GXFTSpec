#!/usr/bin/env python3
from typing import Optional, List, Dict, Any
from pathlib import Path
import argparse
import sys
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from sdk import SDK
from entities import (
    ModelType, AdapterType, FineTuningMethod, Hyperparameters
)

# Initialize console for rich output
console = Console()

def setup_argparse():
    """Set up the command line argument parser"""
    parser = argparse.ArgumentParser(
        description="GXFT - Command line interface for model fine-tuning and hub management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # Hub Models commands
    model_parser = subparsers.add_parser("models", help="Hub model management commands")
    model_subparsers = model_parser.add_subparsers(dest="subcommand", help="Model subcommand")
    model_subparsers.required = True
    
    # List models
    model_list_parser = model_subparsers.add_parser("list", help="List all models in the hub")
    
    # Get model info
    model_get_parser = model_subparsers.add_parser("get", help="Get information about a model")
    model_get_parser.add_argument("model", help="Model name or ID")
    
    # Delete model
    model_delete_parser = model_subparsers.add_parser("delete", help="Delete a model from the hub")
    model_delete_parser.add_argument("model", help="Model name or ID to delete")
    
    # Hub Adapters commands
    adapter_parser = subparsers.add_parser("adapters", help="Hub adapter management commands")
    adapter_subparsers = adapter_parser.add_subparsers(dest="subcommand", help="Adapter subcommand")
    adapter_subparsers.required = True
    
    # List adapters
    adapter_list_parser = adapter_subparsers.add_parser("list", help="List all adapters in the hub")
    
    # Get adapter info
    adapter_get_parser = adapter_subparsers.add_parser("get", help="Get information about an adapter")
    adapter_get_parser.add_argument("adapter", help="Adapter name or ID")
    
    # Delete adapter
    adapter_delete_parser = adapter_subparsers.add_parser("delete", help="Delete an adapter from the hub")
    adapter_delete_parser.add_argument("adapter", help="Adapter name or ID to delete")
    
    # Fine-tuning Jobs commands
    job_parser = subparsers.add_parser("jobs", help="Fine-tuning job management commands")
    job_subparsers = job_parser.add_subparsers(dest="subcommand", help="Job subcommand")
    job_subparsers.required = True
    
    # List jobs
    job_list_parser = job_subparsers.add_parser("list", help="List all fine-tuning jobs")
    
    # Get job info
    job_get_parser = job_subparsers.add_parser("get", help="Get information about a fine-tuning job")
    job_get_parser.add_argument("job", help="Job name or ID")
    
    # Create job
    job_create_parser = job_subparsers.add_parser("create", help="Create a new fine-tuning job")
    job_create_parser.add_argument("name", help="Job name")
    job_create_parser.add_argument("--base-model", required=True, help="Base model name or path")
    job_create_parser.add_argument("--dataset", required=True, help="Path to training dataset")
    job_create_parser.add_argument("--val-dataset", help="Path to validation dataset (optional)")
    job_create_parser.add_argument("--type", choices=["sft", "lora", "qlora"], default="sft", help="Fine-tuning method")
    job_create_parser.add_argument("--output-name", help="Name for output model/adapter")
    job_create_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    job_create_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    job_create_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    # Additional LoRA parameters
    job_create_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank parameter")
    job_create_parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha parameter")
    job_create_parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    
    # Control jobs
    job_start_parser = job_subparsers.add_parser("start", help="Start a fine-tuning job")
    job_start_parser.add_argument("job", help="Job name or ID to start")
    
    job_pause_parser = job_subparsers.add_parser("pause", help="Pause a running fine-tuning job")
    job_pause_parser.add_argument("job", help="Job name or ID to pause")
    
    job_resume_parser = job_subparsers.add_parser("resume", help="Resume a paused fine-tuning job")
    job_resume_parser.add_argument("job", help="Job name or ID to resume")
    
    job_cancel_parser = job_subparsers.add_parser("cancel", help="Cancel a fine-tuning job")
    job_cancel_parser.add_argument("job", help="Job name or ID to cancel")
    
    job_delete_parser = job_subparsers.add_parser("delete", help="Delete a fine-tuning job")
    job_delete_parser.add_argument("job", help="Job name or ID to delete")
    
    # Checkpoint commands
    checkpoint_parser = subparsers.add_parser("checkpoints", help="Checkpoint management commands")
    checkpoint_subparsers = checkpoint_parser.add_subparsers(dest="subcommand", help="Checkpoints subcommand")
    checkpoint_subparsers.required = True
    
    # List checkpoints
    checkpoint_list_parser = checkpoint_subparsers.add_parser("list", help="List checkpoints")
    checkpoint_list_parser.add_argument("--job", help="Filter by job name or ID (optional)")
    
    # Get checkpoint
    checkpoint_get_parser = checkpoint_subparsers.add_parser("get", help="Get information about a checkpoint")
    checkpoint_get_parser.add_argument("checkpoint", help="Checkpoint ID")
    
    # Get latest checkpoint
    checkpoint_latest_parser = checkpoint_subparsers.add_parser("latest", help="Get the latest checkpoint for a job")
    checkpoint_latest_parser.add_argument("job", help="Job name or ID")
    
    # Create checkpoint
    checkpoint_create_parser = checkpoint_subparsers.add_parser("create", help="Create a checkpoint")
    checkpoint_create_parser.add_argument("job", help="Job name or ID")
    checkpoint_create_parser.add_argument("step", type=int, help="Step number")
    checkpoint_create_parser.add_argument("path", type=Path, help="Path to checkpoint files")
    checkpoint_create_parser.add_argument("--train-loss", type=float, help="Training loss")
    checkpoint_create_parser.add_argument("--valid-loss", type=float, help="Validation loss")
    
    # Utility commands
    register_parser = subparsers.add_parser("register-results", help="Register fine-tuning results in the hub")
    register_parser.add_argument("job", help="Job name or ID")
    register_parser.add_argument("--model-name", help="Name for the output model")
    register_parser.add_argument("--adapter-name", help="Name for the output adapter")
    
    return parser

def format_job_status(status: str) -> str:
    """Format job status with colors"""
    color_map = {
        "pending": "yellow",
        "validating": "yellow",
        "preparing": "yellow",
        "queued": "yellow",
        "running": "blue",
        "succeeded": "green",
        "failed": "red",
        "canceled": "magenta",
        "paused": "cyan"
    }
    
    color = color_map.get(status.lower(), "white")
    return f"[{color}]{status}[/{color}]"

def handle_models_command(args, sdk: SDK):
    """Handle models related commands"""
    
    if args.subcommand == "list":
        models = sdk.list_hub_models()
        
        if not models:
            rprint("[yellow]No models registered in hub[/yellow]")
            return 0
            
        table = Table(title="Hub Models")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Origin", style="magenta")
        table.add_column("Created", style="blue")
        
        for model in models:
            table.add_row(
                model["id"],
                model["name"],
                model["type"],
                model["origin"],
                model["created_at"].strftime("%Y-%m-%d %H:%M")
            )
            
        console.print(table)
        
    elif args.subcommand == "get":
        try:
            model = sdk.get_hub_model(args.model)
            
            # Display model details
            console.print(Panel.fit(
                f"[bold cyan]Model: {model.name}[/bold cyan]\n"
                f"ID: {model.id}\n"
                f"Type: {model.type}\n"
                f"Origin: {model.origin}\n"
                f"Path: {model.path}\n"
                f"Created: {model.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"Description: {model.description or 'N/A'}\n"
                f"Base Model: {model.base_model_id or 'N/A'}"
            ))
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "delete":
        try:
            sdk.delete_hub_model(args.model)
            rprint(f"[yellow]Model deleted:[/yellow] {args.model}")
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    return 0

def handle_adapters_command(args, sdk: SDK):
    """Handle adapters related commands"""
    
    if args.subcommand == "list":
        adapters = sdk.list_hub_adapters()
        
        if not adapters:
            rprint("[yellow]No adapters registered in hub[/yellow]")
            return 0
            
        table = Table(title="Hub Adapters")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Model ID", style="magenta")
        table.add_column("Created", style="blue")
        
        for adapter in adapters:
            table.add_row(
                adapter["id"],
                adapter["name"],
                adapter["adapter_type"],
                adapter["model_id"],
                adapter["created_at"].strftime("%Y-%m-%d %H:%M")
            )
            
        console.print(table)
        
    elif args.subcommand == "get":
        try:
            adapter = sdk.get_hub_adapter(args.adapter)
            
            # Display adapter details
            console.print(Panel.fit(
                f"[bold cyan]Adapter: {adapter.name}[/bold cyan]\n"
                f"ID: {adapter.id}\n"
                f"Type: {adapter.adapter_type}\n"
                f"Model ID: {adapter.model_id}\n"
                f"Path: {adapter.path}\n"
                f"Dataset: {adapter.dataset}\n"
                f"Created: {adapter.created_at.strftime('%Y-%m-%d %H:%M')}"
            ))
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "delete":
        try:
            sdk.delete_hub_adapter(args.adapter)
            rprint(f"[yellow]Adapter deleted:[/yellow] {args.adapter}")
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    return 0

def handle_jobs_command(args, sdk: SDK):
    """Handle fine-tuning jobs related commands"""
    
    if args.subcommand == "list":
        jobs = sdk.list_fine_tuning_jobs()
        
        if not jobs:
            rprint("[yellow]No fine-tuning jobs found[/yellow]")
            return 0
            
        table = Table(title="Fine-tuning Jobs")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Base Model", style="green")
        table.add_column("Status", style="magenta")
        table.add_column("Progress", style="yellow")
        
        for job in jobs:
            progress = f"{job['progress']:.1f}%" if job['progress'] is not None else "N/A"
            
            table.add_row(
                job["id"],
                job["name"],
                job["method"],
                job["base_model"].split('/')[-1] if '/' in job["base_model"] else job["base_model"],
                format_job_status(job["status"]),
                progress
            )
            
        console.print(table)
        
    elif args.subcommand == "get":
        try:
            job = sdk.get_fine_tuning_job(args.job)
            
            # Format job details
            details = [
                f"[bold cyan]Fine-tuning Job: {job.name}[/bold cyan]",
                f"ID: {job.id}",
                f"Status: {format_job_status(job.status)}",
                f"Method: {job.hyperparameters.method}",
                f"Base Model: {job.base_model}",
                f"Dataset: {job.dataset}",
                f"Progress: {job.progress:.1f}%",
                f"Created: {job.created_at.strftime('%Y-%m-%d %H:%M')}"
            ]
            
            if job.started_at:
                details.append(f"Started: {job.started_at.strftime('%Y-%m-%d %H:%M')}")
                
            if job.completed_at:
                details.append(f"Completed: {job.completed_at.strftime('%Y-%m-%d %H:%M')}")
                
            if job.duration:
                details.append(f"Duration: {job.duration:.1f} seconds")
                
            if job.error_message:
                details.append(f"Error: [red]{job.error_message}[/red]")
                
            if job.checkpoints:
                details.append(f"Checkpoints: {len(job.checkpoints)}")
                
            if job.output_model_id:
                details.append(f"Output Model: {job.output_model_id}")
                
            if job.output_adapter_id:
                details.append(f"Output Adapter: {job.output_adapter_id}")
                
            console.print(Panel.fit("\n".join(details)))
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "create":
        try:
            # Prepare hyperparameters
            hyperparameters = {
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size
            }
            
            # Add LoRA specific parameters if applicable
            if args.type in ["lora", "qlora"]:
                hyperparameters.update({
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout
                })
                
                # Add QLoRA specific parameters
                if args.type == "qlora":
                    hyperparameters["quantization_bits"] = 8
            
            job = sdk.create_fine_tuning_job(
                name=args.name,
                base_model=args.base_model,
                dataset=args.dataset,
                val_dataset=args.val_dataset,
                job_type=args.type,
                output_model_name=args.output_name,
                hyperparameters=hyperparameters
            )
            
            # Show job details in a table
            table = Table(title=f"Fine-tuning Job Created: {job.name}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Job ID", job.id)
            table.add_row("Method", job.hyperparameters.method)
            table.add_row("Base Model", job.base_model)
            table.add_row("Dataset", job.dataset)
            
            if job.val_dataset:
                table.add_row("Validation Dataset", job.val_dataset)
                
            table.add_row("Epochs", str(job.hyperparameters.epochs))
            table.add_row("Learning Rate", str(job.hyperparameters.learning_rate))
            table.add_row("Batch Size", str(job.hyperparameters.batch_size))
            
            if job.hyperparameters.method in ["lora", "qlora"]:
                table.add_row("LoRA Rank", str(job.hyperparameters.lora_rank))
                table.add_row("LoRA Alpha", str(job.hyperparameters.lora_alpha))
                table.add_row("LoRA Dropout", str(job.hyperparameters.lora_dropout))
                
            console.print(table)
            
            # Show next steps
            console.print("\n[bold green]Job created successfully![/bold green]")
            console.print("To start the job, run:")
            console.print(f"    [blue]clisdk.py jobs start {job.name}[/blue]")
            
        except (ValueError, FileNotFoundError) as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "start":
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Starting job..."),
                transient=True
            ) as progress:
                progress.add_task("start", total=1)
                sdk.start_fine_tuning_job(args.job)
                
            rprint(f"[green]Job started:[/green] {args.job}")
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "pause":
        try:
            sdk.pause_fine_tuning_job(args.job)
            rprint(f"[yellow]Job paused:[/yellow] {args.job}")
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "resume":
        try:
            sdk.resume_fine_tuning_job(args.job)
            rprint(f"[green]Job resumed:[/green] {args.job}")
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "cancel":
        try:
            sdk.cancel_fine_tuning_job(args.job)
            rprint(f"[yellow]Job canceled:[/yellow] {args.job}")
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "delete":
        try:
            sdk.delete_fine_tuning_job(args.job)
            rprint(f"[yellow]Job deleted:[/yellow] {args.job}")
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    return 0

def handle_checkpoints_command(args, sdk: SDK):
    """Handle checkpoints related commands"""
    
    if args.subcommand == "list":
        try:
            checkpoints = sdk.list_checkpoints(args.job)
            
            if not checkpoints:
                if args.job:
                    rprint(f"[yellow]No checkpoints found for job:[/yellow] {args.job}")
                else:
                    rprint("[yellow]No checkpoints found[/yellow]")
                return 0
                
            table = Table(title="Checkpoints")
            table.add_column("ID", style="dim")
            table.add_column("Job ID", style="cyan")
            table.add_column("Step", style="green")
            table.add_column("Train Loss", style="magenta")
            table.add_column("Valid Loss", style="blue")
            table.add_column("Created", style="yellow")
            
            for checkpoint in checkpoints:
                table.add_row(
                    checkpoint["id"],
                    checkpoint["job_id"],
                    str(checkpoint["step_number"]),
                    f"{checkpoint['metrics']['train_loss']:.4f}" if checkpoint['metrics']['train_loss'] is not None else "N/A",
                    f"{checkpoint['metrics']['valid_loss']:.4f}" if checkpoint['metrics']['valid_loss'] is not None else "N/A",
                    checkpoint["created_at"].strftime("%Y-%m-%d %H:%M")
                )
                
            console.print(table)
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "get":
        try:
            checkpoint = sdk.get_checkpoint(args.checkpoint)
            
            # Display checkpoint details
            console.print(Panel.fit(
                f"[bold cyan]Checkpoint: {checkpoint.id}[/bold cyan]\n"
                f"Job ID: {checkpoint.job_id}\n"
                f"Step Number: {checkpoint.step_number}\n"
                f"Path: {checkpoint.path}\n"
                f"Train Loss: {checkpoint.metrics.train_loss or 'N/A'}\n"
                f"Valid Loss: {checkpoint.metrics.valid_loss or 'N/A'}\n"
                f"Created: {checkpoint.created_at.strftime('%Y-%m-%d %H:%M')}"
            ))
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "latest":
        try:
            checkpoint = sdk.get_latest_checkpoint(args.job)
            
            if not checkpoint:
                rprint(f"[yellow]No checkpoints found for job:[/yellow] {args.job}")
                return 0
                
            # Display checkpoint details
            console.print(Panel.fit(
                f"[bold cyan]Latest Checkpoint for Job {args.job}[/bold cyan]\n"
                f"Checkpoint ID: {checkpoint.id}\n"
                f"Step Number: {checkpoint.step_number}\n"
                f"Path: {checkpoint.path}\n"
                f"Train Loss: {checkpoint.metrics.train_loss or 'N/A'}\n"
                f"Valid Loss: {checkpoint.metrics.valid_loss or 'N/A'}\n"
                f"Created: {checkpoint.created_at.strftime('%Y-%m-%d %H:%M')}"
            ))
            
        except ValueError as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    elif args.subcommand == "create":
        try:
            # Prepare metrics if provided
            metrics = None
            if args.train_loss is not None or args.valid_loss is not None:
                metrics = {
                    "train_loss": args.train_loss,
                    "valid_loss": args.valid_loss
                }
                
            checkpoint = sdk.create_checkpoint(
                job_id_or_name=args.job,
                step_number=args.step,
                path=str(args.path),
                metrics=metrics
            )
            
            rprint(f"[green]Checkpoint created:[/green] {checkpoint.id} at step {checkpoint.step_number}")
            
        except (ValueError, FileNotFoundError) as e:
            rprint(f"[red]Error:[/red] {str(e)}")
            return 1
            
    return 0

def handle_register_results(args, sdk: SDK):
    """Handle registering fine-tuning results in the hub"""
    
    try:
        results = sdk.register_fine_tuning_results(
            job_id_or_name=args.job,
            model_name=args.model_name,
            adapter_name=args.adapter_name
        )
        
        if not results:
            rprint(f"[yellow]No results were registered for job:[/yellow] {args.job}")
            return 0
            
        table = Table(title=f"Registered Results for Job {args.job}")
        table.add_column("Type", style="cyan")
        table.add_column("ID", style="green")
        
        if "model_id" in results:
            table.add_row("Model", results["model_id"])
            
        if "adapter_id" in results:
            table.add_row("Adapter", results["adapter_id"])
            
        console.print(table)
        rprint("[green]Results registered successfully![/green]")
        
    except ValueError as e:
        rprint(f"[red]Error:[/red] {str(e)}")
        return 1
        
    return 0

def main():
    """Main CLI entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Initialize SDK
    try:
        sdk = SDK()
    except Exception as e:
        rprint(f"[red]Failed to initialize SDK:[/red] {str(e)}")
        return 1
    
    try:
        # Handle commands based on the selected subcommand
        if args.command == "models":
            return handle_models_command(args, sdk)
            
        elif args.command == "adapters":
            return handle_adapters_command(args, sdk)
            
        elif args.command == "jobs":
            return handle_jobs_command(args, sdk)
            
        elif args.command == "checkpoints":
            return handle_checkpoints_command(args, sdk)
            
        elif args.command == "register-results":
            return handle_register_results(args, sdk)
            
    except Exception as e:
        rprint(f"[red]Error:[/red] {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())