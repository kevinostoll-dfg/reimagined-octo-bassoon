"""
FastAPI main application for the Hybrid Search Graph Agent.

This API provides an endpoint to query the functional agent workflow
that uses Graph RAG tools.
"""

import sys
import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env files BEFORE checking them
from dotenv import load_dotenv
load_dotenv()  # Load .env.local and .env files

# Ensure root directory is in python path to import 'agent'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Set up colored logging with file output
import logging.handlers
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Log file with timestamp
log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Custom colored formatter
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        # Restore original levelname for file logging
        record.levelname = levelname
        
        return formatted

# Create formatters
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_formatter = ColoredFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers
root_logger.handlers = []

# File handler
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_file}")

# Ensure API key is set from environment variables (now .env is loaded)
if not os.getenv("COMPLETIONS_ROUTER_API_KEY"):
    logger.error("COMPLETIONS_ROUTER_API_KEY environment variable is not set. Please set it in .env.local or environment.")
    raise ValueError("COMPLETIONS_ROUTER_API_KEY environment variable is required")

# Import agent modules
from graph_agent import run_agent_query, run_agent_query_async
from agent.config import fetch_models_from_api, set_available_models, get_available_models, get_current_model, set_model_by_id
from prompts.weekly_market_analysis_prompt import (
    get_weekly_market_analysis_system_prompt,
    get_weekly_market_analysis_user_prompt
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting up application...")
    
    # Fetch models from Models API
    logger.info("Fetching models from Models API...")
    models = fetch_models_from_api()
    
    if models:
        set_available_models(models)
        logger.info(f"✅ Loaded {len(models)} models from Models API")
        
        # Force model to qwen/qwen3-max (with force=True to disable failover)
        if set_model_by_id("qwen/qwen3-max", force=True):
            logger.info("✅ Set and FORCED model to qwen/qwen3-max (failover disabled)")
        else:
            logger.warning("⚠️  qwen/qwen3-max not found, using default model")
            current_model = get_current_model()
            if current_model:
                logger.info(f"Current active model: {current_model.get('id')} ({current_model.get('name')})")
    else:
        logger.warning("⚠️  No models fetched from API. Using fallback configuration.")
        logger.warning("The application will continue but may not have access to models.")
    
    # Initialize agent on startup
    try:
        logger.info("Initializing agent on startup...")
        from graph_agent import get_agent
        get_agent()  # Initialize the agent
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.warning(f"Agent initialization on startup failed: {e}. It will be initialized on first query.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Hybrid Search Graph Agent API",
    description="API for querying the functional agent workflow with Graph RAG tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for user queries."""
    query: str
    conversation_id: Optional[str] = None  # Optional for conversation tracking


class QueryResponse(BaseModel):
    """Response model for agent responses."""
    response: str
    query: str
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str

class ModelInfoResponse(BaseModel):
    """Model information response."""
    current_model: Optional[dict]
    available_models: list
    total_models: int


class WeeklyMarketAnalysisRequest(BaseModel):
    """Request model for weekly market analysis."""
    symbol: str
    report_type: Optional[str] = "total_us_market"  # Options: total_us_market, large_cap, mid_cap, small_cap, tech_mag7, ai_sector, by_sector
    sector: Optional[str] = None  # Required if report_type is "by_sector"
    as_of_date: Optional[str] = None  # Optional: YYYY-MM-DD format, defaults to today
    focus_themes: Optional[str] = None  # Optional themes to emphasize


class WeeklyMarketAnalysisResponse(BaseModel):
    """Response model for weekly market analysis."""
    symbol: str
    report: str
    report_type: str
    as_of_date: Optional[str] = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for health check."""
    return HealthResponse(
        status="healthy",
        message="Hybrid Search Graph Agent API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Try to initialize the agent to verify it's working
        from graph_agent import get_agent
        agent = get_agent()
        return HealthResponse(
            status="healthy",
            message="API is operational and agent is initialized"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            message=f"API is running but agent initialization failed: {str(e)}"
        )

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about available models and current model."""
    try:
        current_model = get_current_model()
        available_models = get_available_models()
        return ModelInfoResponse(
            current_model=current_model,
            available_models=available_models,
            total_models=len(available_models)
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return ModelInfoResponse(
            current_model=None,
            available_models=[],
            total_models=0
        )


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Main endpoint to process user queries through the functional agent workflow.
    
    The agent will:
    1. Analyze the query
    2. Use graph_search tool to query the database
    3. Return the final response
    
    Args:
        request: QueryRequest containing the user query and optional conversation_id
        
    Returns:
        QueryResponse with the agent's response
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        logger.info(f"Received query: {request.query}")
        if request.conversation_id:
            logger.info(f"Conversation ID: {request.conversation_id}")
        
        # Use the functional workflow from graph_agent.py
        response = run_agent_query(request.query)
        
        logger.info(f"Query processed successfully. Response length: {len(response)}")
        
        return QueryResponse(
            response=response,
            query=request.query,
            conversation_id=request.conversation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/weekly-market-analysis", response_model=WeeklyMarketAnalysisResponse)
async def weekly_market_analysis(request: WeeklyMarketAnalysisRequest):
    """
    Generate a comprehensive weekly US equity market analysis report for a specific symbol.
    
    This endpoint uses the weekly market analysis prompt as a custom system prompt,
    allowing the agent to generate professional-grade weekly market reports (4,000-5,000 words)
    using its available tools (milvus_search, graph_search, tavily_search, FMP tools).
    
    The agent will automatically:
    - Use datetime tools to get current trading week dates
    - Query Milvus and Graph databases for relevant data
    - Use FMP tools for financial data about the symbol
    - Generate a comprehensive report following the weekly market analysis structure
    
    Args:
        request: WeeklyMarketAnalysisRequest containing:
            - symbol: Stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT") - REQUIRED
            - report_type: Optional report type (default: "total_us_market")
              Options: total_us_market, large_cap, mid_cap, small_cap, tech_mag7, ai_sector, by_sector
            - sector: Optional sector name for "by_sector" report type (e.g., "Technology", "Financials")
            - as_of_date: Optional analysis date in YYYY-MM-DD format (defaults to today)
            - focus_themes: Optional themes to emphasize (e.g., "AI earnings upside", "soft landing")
        
    Returns:
        WeeklyMarketAnalysisResponse with the generated report (4,000-5,000 words)
    """
    try:
        # Validate symbol
        if not request.symbol or not request.symbol.strip():
            raise HTTPException(
                status_code=400,
                detail="Symbol cannot be empty"
            )
        
        symbol = request.symbol.strip().upper()
        logger.info(f"Generating weekly market analysis for symbol: {symbol}, report_type: {request.report_type}")
        
        # Generate the weekly market analysis prompts with the symbol
        try:
            # Get minimal system prompt (tool descriptions + basic ReAct format)
            system_prompt = get_weekly_market_analysis_system_prompt()
            
            # Get enriched user prompt (all weekly market analysis instructions)
            user_prompt = get_weekly_market_analysis_user_prompt(
                report_type=request.report_type or "total_us_market",
                sector=request.sector,
                as_of_date=request.as_of_date,
                focus_themes=request.focus_themes,
                specific_tickers=symbol  # Pass symbol as specific_tickers
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request parameters: {str(e)}"
            )
        
        logger.info(f"Querying agent with weekly market analysis prompts for {symbol}...")
        
        # Use the async agent function directly (we're already in an async context)
        # system_prompt is minimal (tool desc + ReAct format)
        # user_prompt is enriched (all weekly market analysis instructions)
        report = await run_agent_query_async(
            query=user_prompt,
            return_tool_usage=False,
            custom_system_prompt=system_prompt
        )
        
        logger.info(f"Weekly market analysis generated successfully for {symbol}. Report length: {len(report)}")
        
        return WeeklyMarketAnalysisResponse(
            symbol=symbol,
            report=report,
            report_type=request.report_type or "total_us_market",
            as_of_date=request.as_of_date
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating weekly market analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info("API endpoints:")
    logger.info("  GET  / - Health check")
    logger.info("  GET  /health - Detailed health check")
    logger.info("  GET  /model-info - Model information")
    logger.info("  POST /query - Query the agent")
    logger.info("  POST /weekly-market-analysis - Generate weekly market analysis report")
    logger.info("  GET  /docs - API documentation (Swagger UI)")
    logger.info("  GET  /redoc - Alternative API documentation (ReDoc)")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )

