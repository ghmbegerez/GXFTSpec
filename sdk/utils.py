# Generate universal identifier
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from loguru import logger


def generate_uuid():
    return str(uuid.uuid4())

BASE_DIR = Path("fine_tuning_data")
BASE_DIR.mkdir(exist_ok=True)

# Internal modules
class Files:
    @staticmethod    
    def save_file(name: str, path: str, content: Any) -> str:
        # Change 'type' to 'name' since 'type' is undefined
        file_path = BASE_DIR / f"{name}_{Path(path).stem}.json"
        with open(file_path, "w") as f:
            json.dump(content, f)
        return str(file_path)  # Return the path as a string
                
    def load_file(path: str) -> Any:
        path = BASE_DIR / path
        with open(path, "r") as f:
            return json.load(f)
        
class Logs:
    @staticmethod
    def log(job_id: str, message: str, level: str = "info"):
        logger_opt = getattr(logger, level.lower(), logger.info)
        logger_opt(f"Job {job_id}: {message}")

class Metrics:
    @staticmethod
    def compute(train_loss: float, valid_loss: float = None) -> Dict[str, float]:
        return {"train_loss": train_loss, "valid_loss": valid_loss or 0.0}

class Monitoring:
    @staticmethod
    def track_resources(job_id: str) -> Dict[str, float]:
        return {"cpu_usage": 10.0, "memory_usage": 512.0}  # Simulación

class Events:
    @staticmethod
    def notify(job_id: str, message: str, level: str = "info") -> Dict[str, str]:
        Logs.log(job_id, f"Event: {message}", level)
        return {"id": f"evt_{job_id}_{int(datetime.now().timestamp())}", "level": level, "message": message}

