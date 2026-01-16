"""FastAPI application for multilingual sentiment analysis."""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routes import sentiment, language, health
from .models import ErrorResponse
from ..utils.config import settings
from .. import __version__

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up Multilingual Sentiment Analysis API...")
    
    try:
        # Initialize the sentiment analyzer
        sentiment.initialize_analyzer()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multilingual Sentiment Analysis API...")


# Create FastAPI application
app = FastAPI(
    title="Multilingual Sentiment Analysis API",
    description="""
    A comprehensive sentiment analysis API that supports multiple languages with automatic 
    language detection and translation fallback for unsupported languages.
    
    ## Features
    
    * **Multilingual Support**: Analyze sentiment in 10+ languages
    * **Language Detection**: Automatic language identification
    * **Translation Fallback**: Automatic translation for unsupported languages
    * **Batch Processing**: Analyze multiple texts efficiently
    * **High Performance**: Optimized for speed and accuracy
    
    ## Supported Languages
    
    English, Spanish, French, German, Chinese, Portuguese, Italian, Russian, Japanese, Arabic
    
    ## Models Used
    
    * **Sentiment Analysis**: XLM-RoBERTa or mBERT transformer models
    * **Language Detection**: LangDetect library
    * **Translation**: Google Translate API
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error: {exc}")
    
    error_response = ErrorResponse(
        error="validation_error",
        message="Request validation failed",
        details={
            "errors": exc.errors(),
            "body": exc.body
        },
        timestamp=str(request.state.timestamp) if hasattr(request.state, 'timestamp') else None
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.dict()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    
    error_response = ErrorResponse(
        error="http_error",
        message=exc.detail,
        timestamp=str(request.state.timestamp) if hasattr(request.state, 'timestamp') else None
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error="internal_error",
        message="An internal server error occurred",
        details={"type": type(exc).__name__} if settings.debug else None,
        timestamp=str(request.state.timestamp) if hasattr(request.state, 'timestamp') else None
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Middleware to add request timestamp
@app.middleware("http")
async def add_timestamp_middleware(request: Request, call_next):
    """Add timestamp to request state."""
    from datetime import datetime
    request.state.timestamp = datetime.utcnow().isoformat()
    response = await call_next(request)
    return response


# Include routers
app.include_router(health.router)
app.include_router(sentiment.router)
app.include_router(language.router)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers,
        log_level=settings.log_level.lower()
    )
