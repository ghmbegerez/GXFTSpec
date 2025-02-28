import uvicorn
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()
# Configure logger
logger.add(
    "logs/api.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def main():
    """
    Main entry point for the API application.
    Configures and starts the API server.
    """
    # Get configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "False").lower() == "true"
    env = os.getenv("ENVIRONMENT", "development")
    
    # Log startup information
    logger.info(f"✅ Starting SDK API in {env} environment")
    logger.info(f"✅ Server running at http://{host}:{port}")
    logger.info(f"✅ Documentation available at http://{host}:{port}/docs")
    
    # Start the API server
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()