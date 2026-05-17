"""
Configuration for CyberShield v2 Backend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
CYBERSHIELD_SRC = PROJECT_ROOT / "src"


class Settings:
    """Application settings."""
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    # ML Model Configuration
    CHECKPOINT_PATH: str = os.getenv(
        "CHECKPOINT_PATH",
        str(PROJECT_ROOT / "experiments/multifamily_generalization/strict_smoke/best_transformer.pth")
    )
    SCALER_PATH: str = os.getenv(
        "SCALER_PATH",
        str(PROJECT_ROOT / "experiments/baseline/scaler.joblib")
    )
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # Data Configuration
    DATA_DIR: str = os.getenv(
        "DATA_DIR",
        str(PROJECT_ROOT / "data/processed")
    )
    CYBERSHIELD_SRC: str = str(CYBERSHIELD_SRC)
    
    # Mutation Configuration
    MUTATION_BASE_SEED: int = int(os.getenv("MUTATION_BASE_SEED", "42"))
    MUTATION_BATCH_SIZE: int = int(os.getenv("MUTATION_BATCH_SIZE", "256"))
    
    # Artifacts Configuration
    ARTIFACTS_DIR: str = str(ARTIFACTS_DIR)
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# Create singleton settings instance
settings = Settings()

# Create artifacts directory if it doesn't exist
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
