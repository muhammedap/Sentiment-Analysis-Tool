"""Health check and system information API routes."""

import logging
from datetime import datetime
import psutil
import torch

from fastapi import APIRouter, HTTPException

from ..models import HealthResponse
from ...utils.config import settings
from ... import __version__

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Perform health check and return system status.
    
    Returns:
        Health check results
    """
    try:
        # Import here to avoid circular imports
        from ..routes.sentiment import analyzer
        
        # Check if analyzer is loaded
        model_loaded = analyzer is not None
        
        # Get supported languages
        if analyzer:
            supported_languages = list(analyzer.language_detector.get_supported_languages().keys())
            system_info = analyzer.get_system_info()
        else:
            supported_languages = []
            system_info = {}
        
        # Add system resource information
        system_info.update({
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        })
        
        # Determine overall status
        status = "healthy" if model_loaded else "degraded"
        
        response = HealthResponse(
            status=status,
            version=__version__,
            model_loaded=model_loaded,
            supported_languages=supported_languages,
            system_info=system_info,
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Multilingual Sentiment Analysis Tool",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }
