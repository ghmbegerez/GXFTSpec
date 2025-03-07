#!/usr/bin/env python3
from typing import Optional, Dict, Any
from pathlib import Path
import argparse
import sys
import yaml
import json
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from finetuner import FineTuner, FineTunerConfig, DatasetConfig, ModelConfig, TrainingConfig

console = Console()

def setup_argparse():
    """Setup the command line argument parser"""
    parser = argparse.ArgumentParser(
        description="GX FineTuner CLI - Command line interface for direct model fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # Run fine-tuning with a config file
    train_parser = subparsers.add_parser("train", help="Run fine-tuning with a configuration file")
    train_parser.add_argument("config", type=Path, help="Path to a JSON or YAML configuration file")
    train_parser.add_argument("--job-id", help="Custom job ID for this run (default: auto-generated)")
    train_parser.add_argument("--base-dir", help="Base directory for all fine-tuning data (default: from config or 'fine_tuning_data')")
    
    # Create a new config file
    create_config_parser = subparsers.add_parser("create-config", help="Create a new configuration file template")
    create_config_parser.add_argument("output", type=Path, help="Path to save the configuration file")
    create_config_parser.add_argument("--model", default="gpt2", help="Base model name or path")
    create_config_parser.add_argument("--dataset", default="fine_tuning_data/alpaca.jsonl", help="Path to dataset")
    create_config_parser.add_argument("--method", choices=["sft", "lora", "qlora"], default="lora", help="Fine-tuning method")
    create_config_parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Configuration file format")
    create_config_parser.add_argument("--base-dir", default="fine_tuning_data", help="Base directory for all fine-tuning data")
    
    # Evaluate a fine-tuned model
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a fine-tuned model")
    evaluate_parser.add_argument("model_path", type=Path, help="Path to the fine-tuned model")
    evaluate_parser.add_argument("--dataset", required=True, help="Path to evaluation dataset")
    evaluate_parser.add_argument("--job-id", help="Custom job ID for this evaluation")
    evaluate_parser.add_argument("--base-dir", help="Base directory for evaluation data")
    
    # List jobs command
    list_parser = subparsers.add_parser("list-jobs", help="List all fine-tuning jobs")
    list_parser.add_argument("--base-dir", default="fine_tuning_data", help="Base directory to search for jobs")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of jobs to show")
    
    # Job info command
    job_info_parser = subparsers.add_parser("job-info", help="Get information about a specific job")
    job_info_parser.add_argument("job_id", help="Job ID to get information about")
    job_info_parser.add_argument("--base-dir", default="fine_tuning_data", help="Base directory where jobs are stored")
    
    return parser

def run_fine_tuning(args):
    """Run fine-tuning using the FineTuner class with proper directory isolation"""
    try:
        config_file_path = str(args.config)
        
        # Load config file to potentially modify it
        with open(config_file_path, 'r') as f:
            if config_file_path.endswith('.yaml') or config_file_path.endswith('.yml'):
                import yaml
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Override base_dir if specified
        if args.base_dir:
            config_dict['base_dir'] = args.base_dir
            
        # Initialize the fine tuner with the modified config
        rprint(f"[bold]Loading configuration from:[/bold] {config_file_path}")
        fine_tuner = FineTuner(config_dict, job_id=args.job_id)
        
        # Show the configuration and job info
        console.print(Panel.fit(
            f"[bold cyan]Fine-tuning Job Info[/bold cyan]\n"
            f"Job ID: {fine_tuner.job_id}\n"
            f"Base Directory: {fine_tuner.config.base_dir}\n"
            f"Job Directory: {fine_tuner.job_dir}\n"
            f"Dataset: {fine_tuner.config.dataset.path}\n"
            f"Model: {fine_tuner.config.model.model_name_path}\n"
            f"Method: {fine_tuner.config.training.method}\n"
            f"Epochs: {fine_tuner.config.training.num_train_epochs}\n"
            f"Learning Rate: {fine_tuner.config.training.learning_rate}\n"
            f"Output Directory: {fine_tuner.output_dir}"
        ))
        
        # Run training with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Fine-tuning in progress..."),
            transient=False
        ) as progress:
            task = progress.add_task("Training", total=100)
            
            # Start training
            rprint("[green]Starting fine-tuning...[/green]")
            results = fine_tuner.train()
            
            # Update progress (since we can't get real-time updates easily)
            progress.update(task, completed=100)
        
        # Save model and show results
        save_path = fine_tuner.save_model()
        
        rprint(f"[green]Fine-tuning completed successfully![/green]")
        rprint(f"Model saved to: {save_path}")
        
        # Create a summary file for later reference
        summary = {
            "job_id": fine_tuner.job_id,
            "base_dir": fine_tuner.config.base_dir,
            "job_dir": fine_tuner.job_dir,
            "dataset": fine_tuner.config.dataset.path,
            "model": fine_tuner.config.model.model_name_path,
            "method": fine_tuner.config.training.method,
            "output_model_path": save_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Add training metrics
        if hasattr(results, 'metrics'):
            summary["training_loss"] = results.training_loss
            summary["training_steps"] = results.global_step
            
            console.print(Panel.fit(
                f"[bold cyan]Training Results[/bold cyan]\n"
                f"Loss: {results.training_loss:.4f}\n"
                f"Steps: {results.global_step}"
            ))
        
        # Evaluate the model if validation dataset is available
        if fine_tuner.val_dataset:
            rprint("[bold]Running evaluation...[/bold]")
            eval_results = fine_tuner.evaluate()
            if eval_results:
                summary["eval_loss"] = eval_results['eval_loss']
                
                console.print(Panel.fit(
                    f"[bold cyan]Evaluation Results[/bold cyan]\n"
                    f"Loss: {eval_results['eval_loss']:.4f}"
                ))
        
        # Save the job summary
        summary_path = os.path.join(fine_tuner.job_dir, "job_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        rprint(f"Job summary saved to: {summary_path}")
        rprint(f"To get info about this job later, run: finetunercli job-info {fine_tuner.job_id}")
        
        return 0
        
    except Exception as e:
        rprint(f"[red]Error during fine-tuning:[/red] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def create_config_template(args):
    """Create a configuration file template"""
    # Create a default config with the specified parameters
    config = {
        "base_dir": args.base_dir,
        "dataset": {
            "format": "alpaca",
            "path": args.dataset,
            "train_split": "train",
            "val_split": "validation"
        },
        "model": {
            "model_name_path": args.model,
            "max_length": 512,
            "padding_side": "right",
            "device_map": "auto"
        },
        "training": {
            "method": args.method,
            "output_dir": "output",  # This will be overridden by the job-specific directory
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "evaluation_strategy": "epoch",
            "fp16": True,
            "optim": "adamw_torch"
        }
    }
    
    # Add method-specific parameters
    if args.method == "lora" or args.method == "qlora":
        config["training"]["lora"] = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["c_attn"] if "gpt" in args.model.lower() else ["q", "k", "v"]
        }
    
    if args.method == "qlora":
        config["training"]["qlora"] = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    
    # Write the config to a file
    try:
        output_path = args.output
        if args.format == "yaml":
            # If not specified with yaml extension, add it
            if not str(output_path).endswith(('.yaml', '.yml')):
                output_path = Path(f"{str(output_path)}.yaml")
            
            with open(output_path, "w") as f:
                yaml.dump(config, f, sort_keys=False, default_flow_style=False)
        else:
            # If not specified with json extension, add it
            if not str(output_path).endswith('.json'):
                output_path = Path(f"{str(output_path)}.json")
                
            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)
        
        rprint(f"[green]Configuration template created at:[/green] {output_path}")
        rprint("Edit this file to customize your fine-tuning parameters.")
        rprint("To run fine-tuning with this config:")
        rprint(f"[blue]python -m sdk.finetunercli train {output_path}[/blue]")
        
        return 0
    except Exception as e:
        rprint(f"[red]Error creating configuration file:[/red] {str(e)}")
        return 1

def evaluate_model(args):
    """Evaluate a fine-tuned model on a dataset with isolated directory"""
    try:
        model_path = str(args.model_path)
        dataset_path = args.dataset
        base_dir = args.base_dir or "fine_tuning_data"
        job_id = args.job_id or f"eval_{int(datetime.datetime.now().timestamp())}"
        
        rprint(f"[bold]Evaluating model:[/bold] {model_path}")
        rprint(f"[bold]Using dataset:[/bold] {dataset_path}")
        rprint(f"[bold]Job ID:[/bold] {job_id}")
        
        # Create a config for evaluation with proper directory isolation
        config = {
            "base_dir": base_dir,
            "dataset": {
                "format": "alpaca",
                "path": dataset_path,
                "train_split": "train"
            },
            "model": {
                "model_name_path": model_path,
                "max_length": 512,
                "padding_side": "right",
                "device_map": "auto"
            },
            "training": {
                "method": "sft",  # Doesn't matter for evaluation
                "output_dir": "output",
                "per_device_eval_batch_size": 4
            }
        }
        
        # Initialize fine-tuner with the specified job ID for isolation
        fine_tuner = FineTuner(config, job_id=job_id)
        
        # Display job directory
        rprint(f"[bold]Evaluation directory:[/bold] {fine_tuner.job_dir}")
        
        # Run evaluation
        results = fine_tuner.evaluate()
        
        # Save evaluation results
        eval_results_path = os.path.join(fine_tuner.job_dir, "eval_results.json")
        with open(eval_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display results
        if results:
            table = Table(title="Evaluation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in results.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
            
            console.print(table)
            rprint(f"Evaluation results saved to: {eval_results_path}")
        else:
            rprint("[yellow]No evaluation results available.[/yellow]")
        
        # Save summary
        summary = {
            "job_id": fine_tuner.job_id,
            "base_dir": fine_tuner.config.base_dir,
            "job_dir": fine_tuner.job_dir,
            "model_path": model_path,
            "dataset": dataset_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "completed",
            "type": "evaluation"
        }
        
        # Add evaluation results
        if results:
            for key, value in results.items():
                summary[key] = value
        
        # Save the job summary
        summary_path = os.path.join(fine_tuner.job_dir, "job_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return 0
    except Exception as e:
        rprint(f"[red]Error during evaluation:[/red] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def list_jobs(args):
    """List all fine-tuning jobs in the base directory"""
    try:
        base_dir = args.base_dir
        limit = args.limit
        
        # Check if the base directory exists
        if not os.path.exists(base_dir):
            rprint(f"[yellow]Base directory not found:[/yellow] {base_dir}")
            return 1
        
        # Find all job directories
        job_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                summary_path = os.path.join(item_path, "job_summary.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                            job_dirs.append((item, summary))
                    except:
                        # If summary can't be loaded, still include the job but with minimal info
                        job_dirs.append((item, {"job_id": item, "status": "unknown"}))
        
        # Sort by timestamp if available, otherwise by job ID
        job_dirs.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)
        
        # Apply limit
        job_dirs = job_dirs[:limit]
        
        if not job_dirs:
            rprint(f"[yellow]No jobs found in:[/yellow] {base_dir}")
            return 0
        
        # Display the jobs
        table = Table(title=f"Fine-tuning Jobs in {base_dir}")
        table.add_column("Job ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Model", style="blue")
        table.add_column("Method", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="dim")
        
        for job_id, summary in job_dirs:
            job_type = summary.get("type", "fine-tuning")
            model = summary.get("model", "N/A")
            method = summary.get("method", "N/A")
            status = summary.get("status", "unknown")
            timestamp = summary.get("timestamp", "N/A")
            
            # Format timestamp for display
            if timestamp != "N/A":
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
                
            table.add_row(job_id, job_type, model, method, status, timestamp)
        
        console.print(table)
        return 0
    except Exception as e:
        rprint(f"[red]Error listing jobs:[/red] {str(e)}")
        return 1

def job_info(args):
    """Display detailed information about a specific job"""
    try:
        job_id = args.job_id
        base_dir = args.base_dir
        
        job_dir = os.path.join(base_dir, job_id)
        if not os.path.exists(job_dir):
            rprint(f"[red]Job directory not found:[/red] {job_dir}")
            return 1
        
        # Look for job summary
        summary_path = os.path.join(job_dir, "job_summary.json")
        if not os.path.exists(summary_path):
            rprint(f"[yellow]Job summary not found:[/yellow] {summary_path}")
            rprint(f"This doesn't appear to be a valid job directory.")
            return 1
        
        # Load the summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Display detailed job information
        job_type = summary.get("type", "fine-tuning")
        
        # Create a panel with job info
        info_lines = []
        info_lines.append(f"[bold cyan]Job ID:[/bold cyan] {job_id}")
        info_lines.append(f"[bold cyan]Type:[/bold cyan] {job_type}")
        info_lines.append(f"[bold cyan]Status:[/bold cyan] {summary.get('status', 'unknown')}")
        info_lines.append(f"[bold cyan]Directory:[/bold cyan] {job_dir}")
        
        if "timestamp" in summary:
            info_lines.append(f"[bold cyan]Created:[/bold cyan] {summary['timestamp']}")
        
        if job_type == "fine-tuning":
            info_lines.append(f"[bold cyan]Dataset:[/bold cyan] {summary.get('dataset', 'N/A')}")
            info_lines.append(f"[bold cyan]Model:[/bold cyan] {summary.get('model', 'N/A')}")
            info_lines.append(f"[bold cyan]Method:[/bold cyan] {summary.get('method', 'N/A')}")
            info_lines.append(f"[bold cyan]Output Model:[/bold cyan] {summary.get('output_model_path', 'N/A')}")
            
            if "training_loss" in summary:
                info_lines.append(f"[bold cyan]Training Loss:[/bold cyan] {summary['training_loss']:.4f}")
            
            if "eval_loss" in summary:
                info_lines.append(f"[bold cyan]Evaluation Loss:[/bold cyan] {summary['eval_loss']:.4f}")
                
        elif job_type == "evaluation":
            info_lines.append(f"[bold cyan]Model Path:[/bold cyan] {summary.get('model_path', 'N/A')}")
            info_lines.append(f"[bold cyan]Dataset:[/bold cyan] {summary.get('dataset', 'N/A')}")
            
            # Show evaluation metrics
            for key, value in summary.items():
                if key.startswith("eval_") and key != "eval_loss":
                    if isinstance(value, float):
                        info_lines.append(f"[bold cyan]{key}:[/bold cyan] {value:.4f}")
                    else:
                        info_lines.append(f"[bold cyan]{key}:[/bold cyan] {value}")
        
        console.print(Panel("\n".join(info_lines), title=f"{job_type.title()} Job Information"))
        
        # List files in the job directory
        file_table = Table(title="Job Files")
        file_table.add_column("File", style="cyan")
        file_table.add_column("Size", style="yellow")
        file_table.add_column("Modified", style="dim")
        
        for root, dirs, files in os.walk(job_dir):
            rel_path = os.path.relpath(root, job_dir)
            if rel_path == ".":
                prefix = ""
            else:
                prefix = rel_path + "/"
                
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                mtime = os.path.getmtime(file_path)
                
                # Format size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/(1024*1024):.1f} MB"
                
                # Format time
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                
                file_table.add_row(prefix + file, size_str, mtime_str)
        
        console.print(file_table)
        
        return 0
    except Exception as e:
        rprint(f"[red]Error getting job info:[/red] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main CLI entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        if args.command == "train":
            return run_fine_tuning(args)
        
        elif args.command == "create-config":
            return create_config_template(args)
        
        elif args.command == "evaluate":
            return evaluate_model(args)
            
        elif args.command == "list-jobs":
            return list_jobs(args)
            
        elif args.command == "job-info":
            return job_info(args)
        
    except Exception as e:
        rprint(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())