from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal, List, Union, Any
import torch
import os
from pathlib import Path
import json
import numpy as np
from dataclasses import asdict
import typer
import yaml
from loguru import logger

# Imports for model training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType,
    PeftModel
)
from datasets import load_dataset, Dataset
from trl import (
    PPOTrainer, 
    PPOConfig, 
    AutoModelForCausalLMWithValueHead, 
    DPOTrainer, 
    DPOConfig,
    SFTTrainer
)

# Set up application
app = typer.Typer(help="Fine-tune language models with various methods")

"""
FineTuneConfig class
This class defines the configuration for the fine-tuning process.
"""
class FineTuneConfig(BaseModel):
    # Base configuration
    base_model: str = Field(..., description="Base model name or path")
    dataset_path: str = Field(..., description="Path to dataset file or HF dataset name")
    output_dir: str = Field(default="./fine_tuned_model", description="Output directory")
    
    # Training parameters
    method: Literal["lora", "qlora", "rlhf", "dpo", "sft"] = Field(
        default="lora", 
        description="Fine-tuning method: lora, qlora, rlhf (PPO), dpo, or sft (Supervised Fine-Tuning)"
    )
    epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    
    # LoRA parameters
    lora_rank: int = Field(default=8, ge=1, description="LoRA Rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA Alpha")
    lora_dropout: float = Field(default=0.1, ge=0, le=1, description="LoRA Dropout")
    target_modules: Optional[List[str]] = Field(
        default=None, 
        description="LoRA target modules (None for auto-detection)"
    )
    
    # QLoRA parameters
    quantization_bits: int = Field(default=4, ge=2, le=8, description="Quantization bits for QLoRA (2, 4 or 8)")
    
    # Dataset parameters
    max_seq_length: int = Field(default=512, ge=1, description="Maximum sequence length")
    dataset_format: Literal["json", "csv", "parquet", "arrow", "huggingface"] = Field(
        default="json", 
        description="Format of the dataset"
    )
    
    # Column names
    text_column: str = Field(default="text", description="Column name for text in standard fine-tuning")
    prompt_column: str = Field(default="prompt", description="Column name for prompts in SFT/RLHF/DPO")
    completion_column: str = Field(default="completion", description="Column name for completions in SFT")
    chosen_column: str = Field(default="chosen", description="Column name for chosen responses in DPO")
    rejected_column: str = Field(default="rejected", description="Column name for rejected responses in DPO")
    
    # Split name
    train_split: str = Field(default="train", description="Dataset split for training")
    
    # Evaluation parameters
    evaluation_strategy: str = Field(default="no", description="Evaluation strategy: no, steps, epoch")
    eval_steps: int = Field(default=100, ge=1, description="Steps between evaluations if using 'steps'")
    
    # RLHF parameters
    reward_model_name: Optional[str] = Field(
        default=None, 
        description="Reward model name or path for RLHF"
    )
    kl_penalty: float = Field(default=0.2, ge=0, description="KL divergence penalty for PPO")
    
    # DPO parameters
    dpo_beta: float = Field(default=0.1, ge=0, description="Beta parameter for DPO")
    
    # Optimization
    use_gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")
    use_mixed_precision: bool = Field(default=True, description="Use mixed precision training")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    
    def model_init_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for model initialization based on method"""
        kwargs = {
            "device_map": "auto",
        }
        
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            
        if self.method == "qlora":
            kwargs["load_in_bits"] = self.quantization_bits
            
        return kwargs


class FineTuner:
    """Class to handle fine-tuning of language models with various methods."""
    
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.reward_model = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.config.output_dir, "config.json"), "w") as f:
            f.write(self.config.json(indent=2))
            
        logger.info(f"Initialized fine-tuner with method: {self.config.method}")
        logger.info(f"Using device: {self.device}")

    def detect_target_modules(self) -> List[str]:
        """Auto-detect appropriate target modules for LoRA based on model architecture."""
        model_type = self.model.config.model_type.lower()
        model_name = self.config.base_model.lower()
        
        # Map of model types to their target modules
        target_modules_map = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "phi": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
            "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "mpt": ["Wqkv", "out_proj", "fc1", "fc2"],
            "bert": ["query", "key", "value", "output.dense"],
            "gpt": ["c_attn", "c_proj", "c_fc"],
            "opt": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
            "t5": ["q", "k", "v", "o", "wi", "wo"]
        }
        
        # Check if model matches any known architecture
        for key in target_modules_map:
            if key in model_type or key in model_name:
                modules = target_modules_map[key]
                logger.info(f"Auto-detected target modules for {key}: {modules}")
                return modules
                
        # Fallback to common modules
        logger.warning(f"No specific target modules found for {model_type}. Using default modules.")
        return ["q_proj", "v_proj", "k_proj", "o_proj"]

    def load_model(self) -> None:
        """Load model and tokenizer based on configuration."""
        try:
            logger.info(f"Loading model: {self.config.base_model}")
            
            # Get model initialization kwargs
            model_kwargs = self.config.model_init_kwargs()
            
            # Load model based on fine-tuning method
            if self.config.method == "rlhf":
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    self.config.base_model,
                    **model_kwargs
                )
                
                # Load reward model if provided
                if self.config.reward_model_name:
                    logger.info(f"Loading reward model: {self.config.reward_model_name}")
                    self.reward_model = AutoModelForCausalLM.from_pretrained(
                        self.config.reward_model_name,
                        **model_kwargs
                    )
            else:
                # Standard causal LM for other methods
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    **model_kwargs
                )
            
            # Apply quantization for QLoRA
            if self.config.method == "qlora":
                logger.info(f"Preparing for {self.config.quantization_bits}-bit training")
                self.model = prepare_model_for_kbit_training(self.model)

            # Apply LoRA for methods that use it
            if self.config.method in ["lora", "qlora", "rlhf", "sft"]:
                # Detect target modules if not specified
                target_modules = self.config.target_modules
                if target_modules is None:
                    target_modules = self.detect_target_modules()

                # LoRA configuration
                lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=target_modules,
                    bias="none"
                )
                
                logger.info(f"Applying LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
                if not isinstance(self.model, PeftModel):
                    self.model = get_peft_model(self.model, lora_config)
                    
            # Enable gradient checkpointing if requested
            if self.config.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                logger.info("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable()
                
            # Load tokenizer
            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            
            # Ensure tokenizer has necessary tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Log parameter info
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded with {trainable_params:,} trainable / {total_params:,} total parameters")
            logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Error loading model: {str(e)}")

    def load_dataset(self) -> None:
        """Load and prepare dataset based on configuration."""
        try:
            logger.info(f"Loading dataset: {self.config.dataset_path}")
            
            # Load dataset based on format
            if self.config.dataset_format == "huggingface":
                self.dataset = load_dataset(self.config.dataset_path)
            elif self.config.dataset_format == "json":
                self.dataset = load_dataset("json", data_files=self.config.dataset_path)
            elif self.config.dataset_format == "csv":
                self.dataset = load_dataset("csv", data_files=self.config.dataset_path)
            elif self.config.dataset_format == "parquet":
                self.dataset = load_dataset("parquet", data_files=self.config.dataset_path)
            elif self.config.dataset_format == "arrow":
                self.dataset = load_dataset("arrow", data_files=self.config.dataset_path)
            else:
                raise ValueError(f"Unsupported dataset format: {self.config.dataset_format}")
            
            # Verify required columns exist
            self._verify_dataset_columns()
            
            # Use first available split if train_split not found
            if self.config.train_split not in self.dataset and len(self.dataset) > 0:
                logger.warning(f"Split '{self.config.train_split}' not found. Using '{list(self.dataset.keys())[0]}' instead.")
                self.config.train_split = list(self.dataset.keys())[0]
            
            # Process dataset based on method
            if self.config.method == "sft":
                logger.info("Preparing dataset for Supervised Fine-Tuning")
                # SFT processing handled by SFTTrainer
                pass
            elif self.config.method in ["lora", "qlora"]:
                logger.info("Preparing dataset for standard fine-tuning")
                self._process_standard_dataset()
            elif self.config.method == "rlhf":
                logger.info("Preparing dataset for RLHF training")
                # RLHF processing handled by PPOTrainer
                pass
            elif self.config.method == "dpo":
                logger.info("Preparing dataset for DPO training")
                # DPO processing handled by DPOTrainer
                pass
                
            logger.info(f"Dataset loaded with {len(self.dataset[self.config.train_split])} training examples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Error loading dataset: {str(e)}")

    def _verify_dataset_columns(self) -> None:
        """Verify dataset has required columns for selected method."""
        for split in self.dataset:
            columns = self.dataset[split].column_names
            
            if self.config.method == "sft":
                required_cols = [self.config.prompt_column, self.config.completion_column]
                missing = [col for col in required_cols if col not in columns]
                if missing:
                    raise ValueError(f"Dataset split '{split}' missing required columns for SFT: {missing}")
            elif self.config.method in ["lora", "qlora"]:
                if self.config.text_column not in columns:
                    raise ValueError(f"Dataset split '{split}' missing required column: {self.config.text_column}")
            elif self.config.method == "rlhf":
                if self.config.prompt_column not in columns:
                    raise ValueError(f"Dataset split '{split}' missing required column: {self.config.prompt_column}")
            elif self.config.method == "dpo":
                required_cols = [self.config.prompt_column, self.config.chosen_column, self.config.rejected_column]
                missing = [col for col in required_cols if col not in columns]
                if missing:
                    raise ValueError(f"Dataset split '{split}' missing required columns for DPO: {missing}")

    def _process_standard_dataset(self) -> None:
        """Process dataset for standard fine-tuning (LoRA, QLoRA)."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.config.text_column],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
        
        # Process dataset
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in self.dataset[self.config.train_split].column_names if col != self.config.text_column]
        )
        
        # Set format for PyTorch
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def train(self) -> Dict[str, float]:
        """Run training using the selected method."""
        if not self.model or not self.tokenizer or not self.dataset:
            raise RuntimeError("Model and dataset must be loaded first")

        try:
            method_map = {
                "lora": self._train_standard,
                "qlora": self._train_standard,
                "sft": self._train_sft,
                "rlhf": self._train_rlhf,
                "dpo": self._train_dpo
            }
            
            if self.config.method not in method_map:
                raise ValueError(f"Unsupported training method: {self.config.method}")
                
            # Call appropriate training method
            logger.info(f"Starting {self.config.method.upper()} training")
            return method_map[self.config.method]()
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Error during training: {str(e)}")

    def _train_standard(self) -> Dict[str, float]:
        """Train using standard methods (LoRA, QLoRA)."""
        # Configure training arguments
        fp16 = self.config.use_mixed_precision and torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            fp16=fp16,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps if self.config.evaluation_strategy == "steps" else None,
            remove_unused_columns=False,
            load_best_model_at_end=self.config.evaluation_strategy != "no",
            report_to="none",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_total_limit=2,
        )

        # Get datasets
        train_data = self.dataset[self.config.train_split]
        eval_dataset = self.dataset["validation"] if "validation" in self.dataset else None
            
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Train
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.config.output_dir}")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        # Return metrics
        metrics = {
            "train_loss": float(train_result.training_loss),
            "steps": train_result.global_step
        }
        
        # Add evaluation metrics if available
        if eval_dataset and trainer.state.log_history:
            eval_metrics = [log for log in trainer.state.log_history if "eval_loss" in log]
            if eval_metrics:
                metrics["eval_loss"] = eval_metrics[-1]["eval_loss"]
                
        # Save metrics
        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

    def _train_sft(self) -> Dict[str, float]:
        """Train using Supervised Fine-Tuning (SFT)."""
        # Configure training arguments
        fp16 = self.config.use_mixed_precision and torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            fp16=fp16,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps if self.config.evaluation_strategy == "steps" else None,
            load_best_model_at_end=self.config.evaluation_strategy != "no",
            report_to="none",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_total_limit=2,
        )

        # Get datasets
        train_data = self.dataset[self.config.train_split]
        eval_dataset = self.dataset["validation"] if "validation" in self.dataset else None
        
        # Format datasets for SFT
        def format_dataset(examples):
            prompts = examples[self.config.prompt_column]
            completions = examples[self.config.completion_column]
            
            formatted_prompts = []
            for prompt, completion in zip(prompts, completions):
                formatted_prompts.append(f"{prompt}{completion}")
                
            return {"text": formatted_prompts}
            
        # Create SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            formatting_func=format_dataset,
            max_seq_length=self.config.max_seq_length,
            packing=False  # Set to True for more efficient training if desired
        )

        # Train
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)

        # Return metrics
        metrics = {
            "train_loss": float(train_result.training_loss),
            "steps": train_result.global_step
        }
        
        # Add evaluation metrics if available
        if eval_dataset and trainer.state.log_history:
            eval_metrics = [log for log in trainer.state.log_history if "eval_loss" in log]
            if eval_metrics:
                metrics["eval_loss"] = eval_metrics[-1]["eval_loss"]
                
        # Save metrics
        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

    def _train_rlhf(self) -> Dict[str, float]:
        """Train using Reinforcement Learning from Human Feedback (RLHF) with PPO."""
        if not self.config.reward_model_name and not self.reward_model:
            raise ValueError("Reward model must be specified for RLHF training")
            
        # Get the training dataset
        train_data = self.dataset[self.config.train_split]
        
        # Get prompts from the dataset
        prompts = train_data[self.config.prompt_column]
        
        # Configure PPO
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=max(1, self.config.batch_size // 4),  # Sensible default
            ppo_epochs=4,
            kl_penalty=self.config.kl_penalty,
            log_with=None
        )
        
        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            reward_model=self.reward_model
        )
        
        # Training loop
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        total_samples = len(prompts)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        metrics_history = []
        
        for epoch in range(epochs):
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_prompts = prompts[start_idx:end_idx]
                
                # Generate responses
                query_tensors = [
                    ppo_trainer.tokenizer.encode(prompt, return_tensors="pt").to(ppo_trainer.model.device) 
                    for prompt in batch_prompts
                ]
                responses = [
                    ppo_trainer.generate(query_tensor, max_length=self.config.max_seq_length) 
                    for query_tensor in query_tensors
                ]
                
                # Compute rewards
                rewards = []
                for response in responses:
                    response_text = ppo_trainer.tokenizer.decode(response)
                    input_ids = ppo_trainer.tokenizer.encode(response_text, return_tensors="pt").to(ppo_trainer.model.device)
                    with torch.no_grad():
                        outputs = self.reward_model(input_ids)
                        reward = outputs.logits[0][-1].item()  # Use last token logit as reward
                    rewards.append(torch.tensor(reward))
                
                # Run PPO step
                stats = ppo_trainer.step(query_tensors, responses, rewards)
                metrics_history.append(stats)
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{num_batches}, Reward: {torch.mean(torch.stack(rewards)).item():.4f}")
        
        # Save the model
        logger.info(f"Saving model to {self.config.output_dir}")
        ppo_trainer.model.save_pretrained(self.config.output_dir)
        ppo_trainer.tokenizer.save_pretrained(self.config.output_dir)
        
        # Calculate and return metrics
        last_metrics = metrics_history[-1] if metrics_history else {}
        avg_metrics = {key: np.mean([m.get(key, 0) for m in metrics_history if key in m]) for key in last_metrics.keys()}
        metrics = {
            **avg_metrics,
            "epochs": epochs,
            "steps": epochs * num_batches
        }
        
        # Save metrics
        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

    def _train_dpo(self) -> Dict[str, float]:
        """Train using Direct Preference Optimization (DPO)."""
        # Configure DPO training
        dpo_config = DPOConfig(
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            beta=self.config.dpo_beta,
            output_dir=self.config.output_dir,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            fp16=self.config.use_mixed_precision and torch.cuda.is_available(),
            save_strategy="epoch",
            evaluation_strategy=self.config.evaluation_strategy,
            report_to="none"
        )
        
        # Get datasets
        train_data = self.dataset[self.config.train_split]
        eval_dataset = self.dataset["validation"] if "validation" in self.dataset else None
        
        # Create a reference model from the same checkpoint
        reference_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **self.config.model_init_kwargs()
        )
        
        # Create DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=reference_model,
            args=dpo_config,
            train_dataset=train_data,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            beta=self.config.dpo_beta,
            prompt_column=self.config.prompt_column,
            chosen_column=self.config.chosen_column,
            rejected_column=self.config.rejected_column
        )
        
        # Train
        train_result = dpo_trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.config.output_dir}")
        dpo_trainer.model.save_pretrained(self.config.output_dir)
        dpo_trainer.tokenizer.save_pretrained(self.config.output_dir)
        
        # Calculate and return metrics
        metrics = {
            "train_loss": float(train_result.training_loss) if hasattr(train_result, 'training_loss') else 0.0,
            "steps": train_result.global_step
        }
        
        # Add evaluation metrics if available
        if eval_dataset and dpo_trainer.state.log_history:
            eval_metrics = [log for log in dpo_trainer.state.log_history if "eval_loss" in log]
            if eval_metrics:
                metrics["eval_loss"] = eval_metrics[-1]["eval_loss"]
                
        # Save metrics
        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7, 
                top_k: int = 50, top_p: float = 0.95, do_sample: bool = True) -> str:
        """Generate text using the fine-tuned model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model must be loaded first")
        
        try:
            logger.info(f"Generating text with temperature={temperature}, top_k={top_k}, top_p={top_p}")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Configure generation parameters
            generation_config = {
                "max_length": max_length,
                "num_return_sequences": 1,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            # Generate
            outputs = self.model.generate(**inputs, **generation_config)
            
            # Decode and return
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Error generating text: {str(e)}")
            
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the model on a dataset and return metrics."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model must be loaded first")
            
        try:
            logger.info("Starting model evaluation")
            # Use provided eval dataset or try to use validation split
            if eval_dataset is None:
                if "validation" in self.dataset:
                    eval_dataset = self.dataset["validation"]
                elif "test" in self.dataset:
                    eval_dataset = self.dataset["test"]
                else:
                    raise ValueError("No evaluation dataset provided or found in loaded dataset")
            
            # Configure evaluation arguments
            eval_args = TrainingArguments(
                output_dir=os.path.join(self.config.output_dir, "eval"),
                per_device_eval_batch_size=self.config.batch_size,
                report_to="none"
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer
            )
            
            # Evaluate
            metrics = trainer.evaluate()
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Save metrics
            with open(os.path.join(self.config.output_dir, "eval_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(f"Error evaluating model: {str(e)}")
            
    def calculate_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """Calculate perplexity on a set of texts."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model must be loaded first")
            
        try:
            logger.info(f"Calculating perplexity on {len(texts)} samples")
            perplexities = []
            
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Calculate loss
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    # Convert to perplexity
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
            
            # Calculate statistics
            results = {
                "mean": float(np.mean(perplexities)),
                "min": float(np.min(perplexities)),
                "max": float(np.max(perplexities)),
                "std": float(np.std(perplexities))
            }
            
            logger.info(f"Perplexity results: {results}")
            return results
                
        except Exception as e:
            logger.error(f"Perplexity calculation failed: {e}")
            raise RuntimeError(f"Error calculating perplexity: {str(e)}")
    
    def export_model(self, export_format: Literal["gguf", "onnx", "safetensors"] = "safetensors") -> str:
        """Export the model to a specified format."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model must be loaded first")
            
        export_dir = os.path.join(self.config.output_dir, f"export_{export_format}")
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            logger.info(f"Exporting model to {export_format} format")
            
            if export_format == "safetensors":
                self.model.save_pretrained(export_dir, safe_serialization=True)
                self.tokenizer.save_pretrained(export_dir)
                
            elif export_format == "onnx":
                try:
                    from transformers.onnx import export
                    from pathlib import Path
                    
                    onnx_path = Path(export_dir)
                    export(
                        preprocessor=self.tokenizer,
                        model=self.model,
                        output=onnx_path,
                        opset=12
                    )
                except ImportError:
                    logger.error("ONNX export requires transformers.onnx module")
                    raise ImportError("ONNX export requires the 'transformers.onnx' module")
                    
            elif export_format == "gguf":
                # Export to PyTorch format first
                self.model.save_pretrained(export_dir)
                self.tokenizer.save_pretrained(export_dir)
                
                # Save conversion instructions
                conversion_instructions = (
                    "To convert to GGUF format, use llama-cpp-python's conversion script:\n"
                    "python -m llama_cpp.convert_to_gguf /path/to/model --outfile /path/to/output.gguf"
                )
                
                with open(os.path.join(export_dir, "conversion_instructions.txt"), "w") as f:
                    f.write(conversion_instructions)
                    
                logger.info("GGUF export: saved model in PyTorch format with conversion instructions")
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
            logger.info(f"Model exported successfully to {export_dir}")
            return export_dir
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise RuntimeError(f"Error exporting model: {str(e)}")


# Command line interface
@app.command()
def train(
    config_path: str = typer.Argument(..., help="Path to configuration YAML file"),
    export_format: Optional[str] = typer.Option(
        None, help="Export format after training: gguf, onnx, or safetensors"
    )
):
    """Train a language model using the specified configuration."""
    try:
        # Configure logger
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("finetuner.log", rotation="10 MB", level="DEBUG")
        
        logger.info(f"Loading configuration from {config_path}")
        
        # Load configuration
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
        
        # Create config object
        config = FineTuneConfig(**config_data)
        
        # Create fine-tuner
        fine_tuner = FineTuner(config)
        
        # Load model
        fine_tuner.load_model()
        
        # Load dataset
        fine_tuner.load_dataset()
        
        # Train model
        metrics = fine_tuner.train()
        logger.info(f"Training completed with metrics: {metrics}")
        
        # Export if requested
        if export_format:
            if export_format not in ["gguf", "onnx", "safetensors"]:
                logger.warning(f"Unsupported export format: {export_format}. Using safetensors instead.")
                export_format = "safetensors"
                
            export_path = fine_tuner.export_model(export_format)
            logger.info(f"Model exported to {export_path}")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    dataset_path: str = typer.Argument(..., help="Path to evaluation dataset"),
    dataset_format: str = typer.Option("json", help="Dataset format: json, csv, parquet, arrow, huggingface"),
    column: str = typer.Option("text", help="Column name containing text to evaluate"),
    batch_size: int = typer.Option(4, help="Batch size for evaluation")
):
    """Evaluate a fine-tuned model on a dataset."""
    try:
        # Configure logger
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("finetuner_eval.log", rotation="10 MB", level="DEBUG")
        
        logger.info(f"Evaluating model at {model_path} on dataset {dataset_path}")
        
        # Create minimal config for evaluation
        config = FineTuneConfig(
            base_model=model_path,
            dataset_path=dataset_path,
            dataset_format=dataset_format,
            text_column=column,
            batch_size=batch_size,
            output_dir="./evaluation_results"
        )
        
        # Create fine-tuner
        fine_tuner = FineTuner(config)
        
        # Load model
        fine_tuner.load_model()
        
        # Load dataset
        fine_tuner.load_dataset()
        
        # Run evaluation
        metrics = fine_tuner.evaluate()
        logger.info(f"Evaluation completed with metrics: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def generate(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    prompt: str = typer.Argument(..., help="Prompt for text generation"),
    max_length: int = typer.Option(100, help="Maximum length of generated text"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_k: int = typer.Option(50, help="Top-k sampling parameter"),
    top_p: float = typer.Option(0.95, help="Top-p sampling parameter"),
    do_sample: bool = typer.Option(True, help="Whether to use sampling or greedy decoding")
):
    """Generate text using a fine-tuned model."""
    try:
        # Configure logger
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        
        logger.info(f"Generating text with model at {model_path}")
        
        # Create minimal config for generation
        config = FineTuneConfig(
            base_model=model_path,
            dataset_path="",  # Not needed for generation
            output_dir="./generation_results"
        )
        
        # Create fine-tuner
        fine_tuner = FineTuner(config)
        
        # Load model
        fine_tuner.load_model()
        
        # Generate text
        generated_text = fine_tuner.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
        
        # Print generated text
        print("\nGenerated text:")
        print("--------------")
        print(generated_text)
        print("--------------")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    import sys
    app()"