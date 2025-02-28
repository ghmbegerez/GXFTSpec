from typing import Optional, List
from pathlib import Path
import typer
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from sdk import SDK
from entities import DatasetFormat, DatasetFormatField, DatasetSchema, Hyperparameters

# Initialize Typer apps
app = typer.Typer(help="CLI for processing datasets, models and fine-tuning jobs")
console = Console()

# Create sub-commands
dataset_app = typer.Typer(help="Datasets management commands")
model_app = typer.Typer(help="Models management commands")
job_app = typer.Typer(help="Fine-tuning jobs management commands")
adapter_app = typer.Typer(help="Adapters management commands")
deployment_app = typer.Typer(help="Deployments management commands")

# Add sub-commands to main app
app.add_typer(dataset_app, name="datasets")
app.add_typer(model_app, name="models")
app.add_typer(job_app, name="jobs")
app.add_typer(adapter_app, name="adapters")
app.add_typer(deployment_app, name="deployments")

# Initialize SDK
sdk = SDK()

# Dataset commands
@dataset_app.command(name="list")
def list_datasets():
    """List all registered datasets"""
    datasets = sdk.list_datasets()
    if not datasets:
        rprint("[yellow]No datasets registered[/yellow]")
        return

    table = Table(title="Registered Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Format", style="magenta")

    for dataset in datasets:
        format_name = dataset.format.name if dataset.format else "Unknown"
        table.add_row(dataset.name, dataset.path, format_name)

    console.print(table)

@dataset_app.command(name="create")
def create_dataset(
    name: str = typer.Argument(..., help="Dataset name"),
    path: Path = typer.Argument(..., help="Dataset file path"),
    schema: str = typer.Option("custom", help="Dataset schema (completion, chat, instruction, preference, custom)")
):
    """Create a new dataset from local files"""
    # Create a simple format based on the schema
    format = DatasetFormat(
        name=f"{name}_format",
        fields=[DatasetFormatField(name="text", data_type="string", required=True)],
        data_schema=DatasetSchema(schema)
    )
    
    dataset = sdk.create_dataset(name, str(path), format)
    rprint(f"[green]Dataset created:[/green] {dataset.name}")

@dataset_app.command(name="delete")
def delete_dataset(name: str = typer.Argument(..., help="Dataset name")):
    """Delete a dataset"""
    sdk.delete_dataset(name)
    rprint(f"[yellow]Dataset deleted:[/yellow] {name}")

# Model commands
@model_app.command(name="list")
def list_models():
    """List all registered models"""
    models = sdk.list_models()
    if not models:
        rprint("[yellow]No models registered[/yellow]")
        return

    table = Table(title="Registered Models")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="green") 
    table.add_column("Dataset", style="magenta")
    table.add_column("Base Model", style="blue")

    for model in models:
        dataset_name = "N/A"
        dataset = sdk.datasets.get(model.dataset_id)
        if dataset:
            dataset_name = dataset.name
            
        base_model = model.base_model if model.base_model else "N/A"
        table.add_row(model.name, model.path, dataset_name, base_model)

    console.print(table)

@model_app.command(name="create")
def create_model(
    name: str = typer.Argument(..., help="Model name"),
    path: Path = typer.Argument(..., help="Model file path"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b", help="Base model name")
):
    """Create a new model from local files"""
    # Create default hyperparameters
    hyperparameters = Hyperparameters()
    
    model = sdk.create_model(
        name=name,
        path=str(path),
        dataset_id=dataset,
        base_model=base_model,
        hyperparameters=hyperparameters
    )
    rprint(f"[green]Model created:[/green] {model.name}")

# Fine-tuning job commands
@job_app.command(name="create")
def create_fine_tune_job(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    base_model: str = typer.Option(..., "--base-model", "-b", help="Base model name"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    job_type: str = typer.Option("sft", "--type", "-t", help="Job type (sft, lora, qlora)"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    learning_rate: float = typer.Option(5e-5, "--lr", help="Learning rate")
):
    """Create a new fine-tuning job"""
    hyperparameters = Hyperparameters(
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    job = sdk.create_fine_tuning_job(
        job_name=name,
        base_model=base_model,
        dataset=dataset,
        job_type=job_type,
        hyperparameters=hyperparameters
    )
    
    table = Table(title=f"Fine-tuning Job Created: {job.name}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Job Type", job.job_type)
    table.add_row("Base Model", job.base_model)
    table.add_row("Dataset", job.dataset_id)
    table.add_row("Epochs", str(job.hyperparameters.epochs))
    table.add_row("Learning Rate", str(job.hyperparameters.learning_rate))
    
    console.print(table)

@job_app.command(name="list")
def list_fine_tuning_jobs():
    """List all fine-tuning jobs"""
    jobs = sdk.list_fine_tuning_jobs()
    if not jobs:
        rprint("[yellow]No fine-tuning jobs registered[/yellow]")
        return

    table = Table(title="Registered Fine-tuning Jobs")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Base Model", style="green")
    table.add_column("Status", style="magenta")

    for job in jobs:
        table.add_row(job.name, job.job_type, job.base_model, job.status)

    console.print(table)

@job_app.command(name="cancel")
def cancel_fine_tuning_job(name: str = typer.Argument(..., help="Job name")):
    """Cancel a fine-tuning job"""
    sdk.cancel_fine_tuning_job(name)
    rprint(f"[yellow]Job cancelled:[/yellow] {name}")

# Adapter commands
@adapter_app.command(name="create")
def create_adapter(
    name: str = typer.Argument(..., help="Adapter name"),
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    adapter_type: str = typer.Option("lora", "--type", "-t", help="Adapter type (lora, qlora)"),
    path: Path = typer.Option(..., "--path", "-p", help="Path to store adapter")
):
    """Create a new adapter"""
    adapter = sdk.create_adapter(
        name=name,
        model_id=model,
        dataset_id=dataset,
        adapter_type=adapter_type,
        path=str(path)
    )
    rprint(f"[green]Adapter created:[/green] {adapter.name}")

@adapter_app.command(name="list")
def list_adapters():
    """List all registered adapters"""
    adapters = sdk.list_adapters()
    if not adapters:
        rprint("[yellow]No adapters registered[/yellow]")
        return

    table = Table(title="Registered Adapters")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Model", style="green")
    table.add_column("Path", style="magenta")

    for adapter in adapters:
        model_name = "Unknown"
        model = sdk.models.get(adapter.base_model_id)
        if model:
            model_name = model.name
            
        table.add_row(adapter.name, adapter.type, model_name, adapter.path)

    console.print(table)

# Deployment commands
@deployment_app.command(name="deploy")
def deploy_model(
    model_name: str = typer.Argument(..., help="Model name to deploy"),
    deployment_name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom deployment name"),
    adapters: Optional[List[str]] = typer.Option(None, "--adapter", "-a", help="Adapter name to apply"),
    environment: str = typer.Option("development", "--env", "-e", 
                                    help="Environment (development, staging, production)")
):
    """Deploy a model with optional adapters"""
    deployment = sdk.deploy(
        model_name=model_name,
        deployment_name=deployment_name,
        adapters=adapters,
        environment=environment
    )
    
    adapters_str = f"with adapters {adapters}" if adapters else "without adapters"
    rprint(f"[green]Model deployed:[/green] {deployment.name} {adapters_str} in {environment} environment")

@deployment_app.command(name="list")
def list_deployments():
    """List all model deployments"""
    deployments = sdk.list_deployments()
    if not deployments:
        rprint("[yellow]No deployments registered[/yellow]")
        return

    table = Table(title="Registered Deployments")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="blue")
    table.add_column("Environment", style="magenta")

    for deployment in deployments:
        model_name = "Unknown"
        model = sdk.models.get(deployment.base_model_id)
        if model:
            model_name = model.name
            
        table.add_row(deployment.name, model_name, deployment.status, deployment.environment)

    console.print(table)

@deployment_app.command(name="undeploy")
def undeploy_model(name: str = typer.Argument(..., help="Deployment name")):
    """Undeploy a model"""
    sdk.undeploy(name)
    rprint(f"[yellow]Deployment undeployed:[/yellow] {name}")

def main():
    try:
        app()
    except Exception as e:
        rprint(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    main()