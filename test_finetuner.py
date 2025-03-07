#!/usr/bin/env python3
"""
Test script for fine-tuning using the FineTuner class and job queue system.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.jobqueuesystem import JobQueue, JobConfig, Priority, FineTunerJob, run_finetuner_job

console = Console()

# Create a test config file
config = {
    "dataset": {
        "format": "alpaca",
        "path": "fine_tuning_data/alpaca.jsonl",
        "train_split": "train",
        # No validation split for this example
    },
    "model": {
        "model_name_path": "gpt2",  # Using a smaller causal model for testing
        "max_length": 512,
        "padding_side": "right",
        "device_map": "auto"
    },
    "training": {
        "method": "lora",  # Using LoRA for efficient fine-tuning
        "output_dir": "./test_results",
        "num_train_epochs": 1,  # Just 1 epoch for testing
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "evaluation_strategy": "no",  # No evaluation for this test
        "fp16": False,  # Disable fp16 for compatibility
        "optim": "adamw_torch",
        "gradient_checkpointing": False,
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["c_attn"]  # Target modules for GPT-2 model
        }
    }
}

# Save the test configuration
test_config_path = "test_config.json"
with open(test_config_path, "w") as f:
    json.dump(config, f, indent=2)

async def main():
    console.print(f"[green]Test configuration saved to:[/green] {test_config_path}")
    
    # Start job queue server (normally this would be a separate process)
    server_queue = JobQueue(db_path="finetune_jobs.db", is_server=True)
    
    # Create client
    client_queue = JobQueue(db_path="finetune_jobs.db", is_server=False)
    
    # Submit fine-tuning job
    job_id = f"ft_job_{int(asyncio.get_event_loop().time())}"
    
    # Submit the job
    job = await client_queue.submit(
        run_finetuner_job,
        test_config_path,
        job_id,
        name=f"fine_tuning_{job_id}",
        config=JobConfig(priority=Priority.HIGH)
    )
    
    console.print(f"[green]Job submitted successfully![/green]")
    console.print(f"Job ID: {job.id}")
    console.print(f"Name: {job.name}")
    console.print(f"Status: {job.status}")
    
    # Monitor job progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Fine-tuning in progress..."),
        transient=False
    ) as progress:
        task = progress.add_task("Training", total=None)
        
        try:
            job = await client_queue.wait_for_job(job.id, timeout=None)  # No timeout for fine-tuning
            progress.update(task, completed=True)
            
            if job.status == "completed":
                console.print(f"[green]Fine-tuning completed successfully![/green]")
                if isinstance(job.result, dict):
                    # Print summary
                    console.print("\n[bold]Job Results:[/bold]")
                    for key, value in job.result.items():
                        if isinstance(value, float):
                            console.print(f"{key}: {value:.4f}")
                        else:
                            console.print(f"{key}: {value}")
            else:
                console.print(f"[red]Fine-tuning failed: {job.error}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error monitoring job: {str(e)}[/red]")
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())