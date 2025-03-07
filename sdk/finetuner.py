import os
import json
import datetime
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from loguru import logger

# Convert this lines to a function
def setup_logging(organization_id: str, project_id: str, job_id: str):
    logger.remove()
    os.makedirs(f"{organization_id}/fine_tuning_data/output/logs", exist_ok=True)   
    logger.add("f{organization_id}/fine_tuning_data/output/logs/finetuning_{job_id}.log", rotation="1 day", retention="1 week", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(f"{organization_id}/fine_tuning_data/output/logs/finetuning_errors{job_id}.log", level="ERROR", rotation="1 day", retention="1 week", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")  


class DatasetConfig(BaseModel):
    format: Literal["alpaca", "openai"] = "alpaca"
    path: str
    train_split: str = "train"
    val_split: Optional[str] = "validation"
    text_column: Optional[str] = None
    input_column: Optional[str] = None
    output_column: Optional[str] = None


class ModelConfig(BaseModel):
    model_name_path: str
    tokenizer_name_path: Optional[str] = None
    max_length: int = 2048
    padding_side: str = "right"
    device_map: str = "auto"
    

class LoRAConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    

class QLoRAConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    
    class Config:
        arbitrary_types_allowed = True
    

class TrainingConfig(BaseModel):
    method: Literal["sft", "lora", "qlora"] = "sft"
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    evaluation_strategy: str = "epoch"
    fp16: bool = True
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = False
    lora: Optional[LoRAConfig] = None
    qlora: Optional[QLoRAConfig] = None
    
    @field_validator('lora')
    def validate_lora(cls, v, info):
        if info.data.get('method') == 'lora' and v is None:
            return LoRAConfig()
        return v
    
    @field_validator('qlora')
    def validate_qlora(cls, v, info):
        if info.data.get('method') == 'qlora' and v is None:
            return QLoRAConfig()
        return v


class FineTunerConfig(BaseModel):
    input_dir: str = "fine_tuning_data/input"  # Directory where input data is located
    output_dir: str = "fine_tuning_data/output"  # Directory where output will be generated
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


class FineTuner:
    def __init__(self, config: Union[FineTunerConfig, Dict[str, Any], str], job_id: str = None):
        if isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
                print(config_dict)
            self.config = FineTunerConfig(**config_dict)
        elif isinstance(config, dict):
            self.config = FineTunerConfig(**config)
        else:
            self.config = config
            
        # Set up job ID or create a unique one
        self.job_id = job_id or f"job_{int(datetime.datetime.now().timestamp())}"
        
        # Create necessary directories for isolation
        self.setup_directories()
        
        self.setup_tokenizer()
        self.setup_model()
        self.setup_dataset()
        self.setup_trainer()
        
    def setup_directories(self):
        """Set up all necessary directories for this fine-tuning job."""
        # Create input and output directories if they don't exist
        os.makedirs(self.config.input_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create job directory inside output directory
        self.job_dir = os.path.join(self.config.output_dir, self.job_id)
        os.makedirs(self.job_dir, exist_ok=True)
        
        # Create subdirectories for this job
        self.logs_dir = os.path.join(self.job_dir, "logs")
        self.checkpoints_dir = os.path.join(self.job_dir, "checkpoints")
        self.output_dir = os.path.join(self.job_dir, "output")
        self.cache_dir = os.path.join(self.job_dir, "cache")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Update the output directory in the config
        self.config.training.output_dir = self.output_dir
        
        # Configure logging for this job
        log_file = os.path.join(self.logs_dir, "finetuning.log")
        job_log_id = logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
        logger.info(f"Job ID: {self.job_id}")
        logger.info(f"Input directory: {self.config.input_dir}")
        logger.info(f"Job directory: {self.job_dir}")
        logger.info(f"Logs directory: {self.logs_dir}")
        logger.info(f"Checkpoints directory: {self.checkpoints_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def setup_tokenizer(self):
        tokenizer_name = self.config.model.tokenizer_name_path or self.config.model.model_name_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_side=self.config.model.padding_side,
            cache_dir=self.cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_model(self):
        method = self.config.training.method
        model_path = self.config.model.model_name_path
        logger.info(f"Loading model: {model_path} with method: {method}")
        
        # Common model loading parameters
        model_kwargs = {
            "device_map": self.config.model.device_map,
            "cache_dir": self.cache_dir
        }
        
        if method == "qlora":
            # Convert string to torch dtype
            compute_dtype = torch.float16
            if self.config.training.qlora.bnb_4bit_compute_dtype == "float32":
                compute_dtype = torch.float32
            elif self.config.training.qlora.bnb_4bit_compute_dtype == "bfloat16":
                compute_dtype = torch.bfloat16
                
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.training.qlora.load_in_4bit,
                bnb_4bit_use_double_quant=self.config.training.qlora.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.training.qlora.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                **model_kwargs
            )
                
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=self.config.training.lora.r,
                lora_alpha=self.config.training.lora.lora_alpha,
                lora_dropout=self.config.training.lora.lora_dropout,
                bias=self.config.training.lora.bias,
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.config.training.lora.target_modules
            )
            self.model = get_peft_model(self.model, lora_config)
            
        elif method == "lora":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            lora_config = LoraConfig(
                r=self.config.training.lora.r,
                lora_alpha=self.config.training.lora.lora_alpha,
                lora_dropout=self.config.training.lora.lora_dropout,
                bias=self.config.training.lora.bias,
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.config.training.lora.target_modules
            )
            self.model = get_peft_model(self.model, lora_config)
            
        else:  # sft
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
        logger.info(f"Model loaded successfully")
    
    def setup_dataset(self):
        # Resolve dataset path based on whether it's relative or absolute
        self.dataset_path = self.config.dataset.path
        if not os.path.isabs(self.dataset_path) and not self.dataset_path.startswith("hf://"):
            # If it's a relative path, assume it's relative to the input directory
            self.dataset_path = os.path.join(self.config.input_dir, self.dataset_path)
        
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Create dataset cache directory
        dataset_cache_dir = os.path.join(self.cache_dir, "datasets")
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        # Check if the path is a local file
        if os.path.isfile(self.dataset_path):
            try:
                # For local jsonl files
                dataset = load_dataset(
                    "json", 
                    data_files=self.dataset_path, 
                    cache_dir=dataset_cache_dir
                )
                # Convert to train/validation split
                dataset = dataset["train"].train_test_split(test_size=0.1)
                train_dataset = dataset["train"]
                val_dataset = dataset["test"] if self.config.dataset.val_split else None
            except Exception as e:
                logger.error(f"Error loading local dataset: {e}")
                raise
        else:
            # For datasets from the hub or other sources
            try:
                dataset = load_dataset(
                    self.dataset_path if self.dataset_path.startswith("hf://") else self.config.dataset.path,
                    cache_dir=dataset_cache_dir
                )
                train_dataset = dataset[self.config.dataset.train_split]
                val_dataset = dataset[self.config.dataset.val_split] if self.config.dataset.val_split else None
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                raise
        
        # Save dataset information
        dataset_info = {
            "source_path": self.dataset_path,
            "original_path": self.config.dataset.path,
            "format": self.config.dataset.format,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset) if val_dataset else 0,
            "train_split": self.config.dataset.train_split,
            "val_split": self.config.dataset.val_split
        }
        
        # Save dataset info to job directory
        dataset_info_path = os.path.join(self.job_dir, "dataset_info.json")
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        format_type = self.config.dataset.format
        logger.info(f"Formatting dataset with format: {format_type}")
        
        if format_type == "alpaca":
            text_column = self.config.dataset.text_column or "text"
            input_column = self.config.dataset.input_column or "instruction"
            output_column = self.config.dataset.output_column or "output"
            
            def format_alpaca(example):
                # For causal models, combine inputs and outputs
                if example.get(input_column):
                    text = f"### Instruction:\n{example[input_column]}\n\n### Response:\n{example[output_column]}"
                else:
                    text = example.get(text_column, "")
                return {"text": text}
            
            self.train_dataset = train_dataset.map(format_alpaca)
            self.val_dataset = val_dataset.map(format_alpaca) if val_dataset else None
            
        elif format_type == "openai":
            input_column = self.config.dataset.input_column or "messages"
            
            def format_openai(example):
                messages = example[input_column]
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>\n{content}\n"
                return {"text": text}
            
            self.train_dataset = train_dataset.map(format_openai)
            self.val_dataset = val_dataset.map(format_openai) if val_dataset else None
        
        logger.info(f"Dataset formatting complete. Train size: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Validation size: {len(self.val_dataset)}")
        
        # Save a few examples for reference
        examples_file = os.path.join(self.job_dir, "dataset_examples.txt")
        with open(examples_file, 'w') as f:
            f.write("Dataset Examples:\n\n")
            for i, example in enumerate(self.train_dataset[:5]):
                f.write(f"Example {i+1}:\n")
                f.write(f"{example['text']}\n\n")
        
        def tokenize(example):
            return self.tokenizer(
                example["text"],
                max_length=self.config.model.max_length,
                padding="max_length",
                truncation=True
            )
        
        logger.info(f"Tokenizing dataset...")
        self.train_dataset = self.train_dataset.map(tokenize, batched=True)
        self.val_dataset = self.val_dataset.map(tokenize, batched=True) if self.val_dataset else None
        logger.info(f"Tokenization complete")
    
    def setup_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            save_strategy=self.config.training.save_strategy,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy=self.config.training.evaluation_strategy,
            fp16=self.config.training.fp16,
            optim=self.config.training.optim,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            report_to="tensorboard"
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator
        )
    
    def train(self):
        logger.info(f"Starting training process...")
        try:
            results = self.trainer.train()
            logger.info(f"Training completed. Steps: {results.global_step}, Loss: {results.training_loss:.4f}")
            return results
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self):
        if self.val_dataset is None:
            logger.warning("No validation dataset available. Skipping evaluation.")
            return None
        
        logger.info(f"Starting evaluation...")
        try:
            results = self.trainer.evaluate()
            logger.info(f"Evaluation results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_model(self, path=None):
        """Save the final model to the specified path or the default output directory."""
        # Default path is in the job output directory
        if not path:
            save_path = os.path.join(self.output_dir, "final_model")
        else:
            # If absolute path, use it; otherwise, make it relative to job dir
            save_path = path if os.path.isabs(path) else os.path.join(self.job_dir, path)
            
        logger.info(f"Saving model to {save_path}...")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save the model
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save configuration used for this training
            config_path = os.path.join(save_path, "training_config.json")
            with open(config_path, 'w') as f:
                config_dict = self.config.model_dump()
                json.dump(config_dict, f, indent=2)
                
            # Save model info
            model_info = {
                "job_id": self.job_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "base_model": self.config.model.model_name_path,
                "training_method": self.config.training.method,
                "dataset": self.dataset_path if hasattr(self, 'dataset_path') else self.config.dataset.path,
                "path": save_path
            }
            
            model_info_path = os.path.join(save_path, "model_info.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Model, tokenizer, and configuration saved successfully")
            return save_path
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise