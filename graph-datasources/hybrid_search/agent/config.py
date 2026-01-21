
import os
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging
LOGGING_LEVEL = logging.INFO

# Dragonfly (Redis) Configuration
DRAGONFLY_HOST = os.getenv("DRAGONFLY_HOST", "dragonfly.blacksmith.deerfieldgreen.com")
DRAGONFLY_PORT = int(os.getenv("DRAGONFLY_PORT", "6379"))
DRAGONFLY_PASSWORD = os.getenv("DRAGONFLY_PASSWORD")
DRAGONFLY_DB = int(os.getenv("DRAGONFLY_DB", "0"))
DRAGONFLY_SSL = os.getenv("DRAGONFLY_SSL", "false").lower() == "true"

# API Configuration
COMPLETIONS_ROUTER_API_KEY = os.getenv("COMPLETIONS_ROUTER_API_KEY")
COMPLETIONS_ROUTER_URL = os.getenv("COMPLETIONS_ROUTER_URL", "https://api.blacksmith.deerfieldgreen.com/v3/completions/chat/completions")
MODELS_API_KEY = os.getenv("MODELS_API_KEY")
MODELS_API_URL = os.getenv("MODELS_API_URL", "https://api.blacksmith.deerfieldgreen.com/v3/completions/models")
DEFAULT_TIMEOUT = 180  # 3 minutes for long reports
MAX_RETRIES = 5

# Model Configuration - Now populated from Models API
AVAILABLE_MODELS = []  # Will be populated from Models API on startup
CURRENT_MODEL_INDEX = 0  # Index of current active model
MODEL_FAILURE_COUNT = 0  # Consecutive failure count
MAX_CONSECUTIVE_FAILURES = 3  # Failover threshold
FORCE_MODEL_ID = None  # If set, always use this model and don't failover (e.g., "qwen/qwen3-max")

logger = logging.getLogger(__name__)

def transform_models(raw_models_data: dict) -> list:
    """
    Transform Models API response to internal format.
    
    Args:
        raw_models_data: Response from Models API with 'data' key
        
    Returns:
        List of transformed model dictionaries
    """
    models = []
    raw_models = raw_models_data.get("data", [])
    
    for raw_model in raw_models:
        model_id = raw_model.get("id")
        if not model_id:
            logger.warning(f"Skipping model without ID: {raw_model}")
            continue
            
        transformed_model = {
            "id": model_id,
            "name": raw_model.get("title") or raw_model.get("display_name") or model_id,
            "description": raw_model.get("description", ""),
            "context_size": raw_model.get("context_size", 131072),
            "parameters": {
                "max_tokens": raw_model.get("max_output_tokens", 20000),
                "temperature": 0.7
            },
            "type": raw_model.get("model_type", "chat")
        }
        models.append(transformed_model)
        logger.info(f"Transformed model: {transformed_model['id']} -> {transformed_model['name']}")
    
    return models

def set_available_models(models: list) -> None:
    """
    Set the available models list.
    
    Args:
        models: List of model dictionaries from transform_models()
    """
    global AVAILABLE_MODELS, CURRENT_MODEL_INDEX, MODEL_FAILURE_COUNT
    AVAILABLE_MODELS = models
    CURRENT_MODEL_INDEX = 0  # Reset to first model
    MODEL_FAILURE_COUNT = 0  # Reset failure count
    logger.info(f"Set {len(models)} available models")

def get_available_models() -> list:
    """Return list of all available models."""
    return AVAILABLE_MODELS.copy()

def get_model_id():
    """Return the current model ID."""
    global CURRENT_MODEL_INDEX
    
    # Ensure models are loaded if empty
    if not AVAILABLE_MODELS:
        ensure_models_loaded()
    
    if CURRENT_MODEL_INDEX < len(AVAILABLE_MODELS):
        return AVAILABLE_MODELS[CURRENT_MODEL_INDEX].get("id", "fallback-model")
    return "fallback-model"

def get_model_max_tokens():
    """Return max_tokens from current model configuration, capped at reasonable limit."""
    global CURRENT_MODEL_INDEX
    if CURRENT_MODEL_INDEX < len(AVAILABLE_MODELS):
        model_max = AVAILABLE_MODELS[CURRENT_MODEL_INDEX].get("parameters", {}).get("max_tokens", 20000)
        context_size = AVAILABLE_MODELS[CURRENT_MODEL_INDEX].get("context_size", 131072)
        
        # For COMPACT_AND_REFINE, cap at 20% of context window (increased from 10% for longer responses)
        # COMPACT_AND_REFINE requires extra context for chunk repacking, but we can use more for longer reports
        safe_max = min(model_max, int(context_size * 0.2))
        
        if safe_max < model_max:
            logger.info(f"Capped max_tokens from {model_max} to {safe_max} (20% of context window {context_size} for COMPACT_AND_REFINE)")
        
        return safe_max
    return 16384  # Increased fallback default

def get_model_temperature():
    """Return temperature from current model configuration."""
    global CURRENT_MODEL_INDEX
    if CURRENT_MODEL_INDEX < len(AVAILABLE_MODELS):
        return AVAILABLE_MODELS[CURRENT_MODEL_INDEX].get("parameters", {}).get("temperature", 0.7)
    return 0.7  # Fallback default

def get_current_model():
    """Return the current model dictionary."""
    global CURRENT_MODEL_INDEX
    if CURRENT_MODEL_INDEX < len(AVAILABLE_MODELS):
        return AVAILABLE_MODELS[CURRENT_MODEL_INDEX]
    return None

def set_model_by_id(model_id: str, force: bool = False) -> bool:
    """
    Set the current model by model ID.
    
    Args:
        model_id: The model ID to set (e.g., "qwen/qwen3-max")
        force: If True, force this model and disable failover (default: False)
    
    Returns:
        True if model was found and set, False otherwise
    """
    global CURRENT_MODEL_INDEX, MODEL_FAILURE_COUNT, FORCE_MODEL_ID
    
    # Ensure models are loaded
    if not AVAILABLE_MODELS:
        ensure_models_loaded()
    
    # Find the model by ID
    for idx, model in enumerate(AVAILABLE_MODELS):
        if model.get("id") == model_id:
            CURRENT_MODEL_INDEX = idx
            MODEL_FAILURE_COUNT = 0  # Reset failure count
            
            if force:
                FORCE_MODEL_ID = model_id
                logger.info(f"Set and FORCED model to: {model.get('id')} ({model.get('name')}) - failover disabled")
            else:
                FORCE_MODEL_ID = None
                logger.info(f"Set model to: {model.get('id')} ({model.get('name')})")
            return True
    
    logger.error(f"Model '{model_id}' not found in available models")
    return False

def record_model_failure(is_network_error: bool = False) -> bool:
    """
    Record a failure for the current model and switch if threshold reached.
    For network errors, switch faster (after 1 failure instead of 3) since they're often persistent.
    If FORCE_MODEL_ID is set, never switch models (always retry the forced model).
    Returns True if switched to next model, False if no more models available or model is forced.
    """
    global MODEL_FAILURE_COUNT, CURRENT_MODEL_INDEX, FORCE_MODEL_ID
    
    # If a model is forced, never switch - always retry
    if FORCE_MODEL_ID:
        current_model = get_current_model()
        if current_model and current_model.get("id") == FORCE_MODEL_ID:
            logger.warning(f"Model {FORCE_MODEL_ID} is forced - not switching, will retry")
            return False
    
    MODEL_FAILURE_COUNT += 1
    
    # Network errors should trigger faster failover (after 1 failure instead of 3)
    threshold = 1 if is_network_error else MAX_CONSECUTIVE_FAILURES
    logger.warning(f"Model failure count: {MODEL_FAILURE_COUNT}/{threshold} (network_error={is_network_error})")
    
    if MODEL_FAILURE_COUNT >= threshold:
        CURRENT_MODEL_INDEX += 1
        MODEL_FAILURE_COUNT = 0  # Reset count for new model
        
        if CURRENT_MODEL_INDEX < len(AVAILABLE_MODELS):
            new_model = AVAILABLE_MODELS[CURRENT_MODEL_INDEX]
            logger.info(f"Switched to model: {new_model.get('id')} ({new_model.get('name')})")
            return True  # Switched to next model
        else:
            logger.error("No more models available for failover")
            return False  # No more models available
    
    return False  # Not yet at threshold

def record_model_success():
    """Reset failure count on successful request."""
    global MODEL_FAILURE_COUNT
    if MODEL_FAILURE_COUNT > 0:
        logger.info(f"Resetting failure count (was {MODEL_FAILURE_COUNT})")
        MODEL_FAILURE_COUNT = 0

def fetch_models_from_api() -> list:
    """
    Fetch models from Models API.
    
    Returns:
        List of transformed model dictionaries, or empty list on error
    """
    try:
        logger.info(f"Fetching models from Models API: {MODELS_API_URL}")
        response = requests.get(
            MODELS_API_URL,
            headers={
                "Authorization": f"Bearer {MODELS_API_KEY}",
                "accept": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            models = transform_models(response_data)
            logger.info(f"✅ Successfully fetched {len(models)} models from Models API")
            return models
        else:
            logger.error(f"Failed to fetch models: Status {response.status_code}, Response: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching models from API: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching models: {e}")
        return []

def ensure_models_loaded():
    """
    Ensure models are loaded. If AVAILABLE_MODELS is empty, try to fetch from API.
    This is useful when the agent is initialized outside of FastAPI lifespan.
    """
    global AVAILABLE_MODELS
    
    if not AVAILABLE_MODELS:
        logger.warning("AVAILABLE_MODELS is empty. Attempting to fetch from API...")
        models = fetch_models_from_api()
        if models:
            set_available_models(models)
            logger.info(f"✅ Loaded {len(models)} models")
        else:
            logger.error("⚠️  Failed to fetch models. Using fallback model configuration.")
            # Set fallback models as last resort
            fallback_models = [
                {
                    "id": "qwen/qwen3-next-80b-a3b-thinking",
                    "name": "Qwen3 Next 80B",
                    "description": "Fallback model",
                    "context_size": 131072,
                    "parameters": {
                        "max_tokens": 16384,
                        "temperature": 0.7
                    },
                    "type": "chat"
                }
            ]
            set_available_models(fallback_models)
            logger.warning("Using fallback model configuration")
