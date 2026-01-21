#!/usr/bin/env python3
"""
Test script for CompletionsRouter.

This script tests the CompletionsRouter LLM wrapper to verify:
- Model loading and initialization
- Basic completion requests
- Error handling
- Streaming completion (if supported)
- API connectivity
- Timeout handling
"""

import sys
import os
import logging
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure root directory is in python path to import 'agent'
# Script is in scripts/, agent module is in parent directory (hybrid_search/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from scripts/ to hybrid_search/
sys.path.insert(0, project_root)  # Insert at beginning to prioritize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_api_connectivity():
    """Test basic API connectivity before running full tests."""
    logger.info("\n" + "="*80)
    logger.info("PRE-TEST: API Connectivity Check")
    logger.info("="*80)
    
    try:
        from agent.config import COMPLETIONS_ROUTER_URL, COMPLETIONS_ROUTER_API_KEY
        
        logger.info(f"API URL: {COMPLETIONS_ROUTER_URL}")
        logger.info(f"API Key present: {bool(COMPLETIONS_ROUTER_API_KEY)}")
        
        # Try a simple HEAD request or GET to check connectivity
        # Extract base URL (remove /chat/completions)
        base_url = COMPLETIONS_ROUTER_URL.rsplit('/', 1)[0] if '/' in COMPLETIONS_ROUTER_URL else COMPLETIONS_ROUTER_URL
        
        logger.info(f"Testing connectivity to: {base_url}")
        
        try:
            # Try a simple request with short timeout
            response = requests.get(
                base_url.replace('/chat/completions', '/health') if '/chat/completions' in base_url else base_url,
                timeout=5,
                headers={"Authorization": f"Bearer {COMPLETIONS_ROUTER_API_KEY}"} if COMPLETIONS_ROUTER_API_KEY else {}
            )
            logger.info(f"✅ Connectivity check: Status {response.status_code}")
            return True
        except requests.exceptions.Timeout:
            logger.warning("⚠️  Connectivity check timed out (API may be slow)")
            return True  # Continue anyway
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Connection error: {e}")
            logger.error("Cannot reach API endpoint. Check network connectivity.")
            return False
        except Exception as e:
            logger.warning(f"⚠️  Connectivity check failed: {e}")
            return True  # Continue anyway - might be expected
        
    except Exception as e:
        logger.error(f"❌ Error checking API connectivity: {e}")
        return False


def test_configuration():
    """Test and display configuration."""
    logger.info("\n" + "="*80)
    logger.info("PRE-TEST: Configuration Check")
    logger.info("="*80)
    
    try:
        from agent.config import (
            COMPLETIONS_ROUTER_URL,
            COMPLETIONS_ROUTER_API_KEY,
            DEFAULT_TIMEOUT,
            MAX_RETRIES,
            ensure_models_loaded,
            get_current_model
        )
        
        logger.info(f"API URL: {COMPLETIONS_ROUTER_URL}")
        logger.info(f"Default Timeout: {DEFAULT_TIMEOUT} seconds")
        logger.info(f"Max Retries: {MAX_RETRIES}")
        logger.info(f"API Key length: {len(COMPLETIONS_ROUTER_API_KEY) if COMPLETIONS_ROUTER_API_KEY else 0} characters")
        
        ensure_models_loaded()
        current_model = get_current_model()
        if current_model:
            logger.info(f"Current model: {current_model.get('id')} ({current_model.get('name')})")
            logger.info(f"Model max tokens: {current_model.get('parameters', {}).get('max_tokens', 'N/A')}")
            logger.info(f"Model context size: {current_model.get('context_size', 'N/A')}")
        else:
            logger.warning("⚠️  No model available")
        
        return True
    except Exception as e:
        logger.error(f"❌ Configuration check failed: {e}", exc_info=True)
        return False


def test_completions_router():
    """Test the CompletionsRouter with various prompts."""
    
    try:
        # Ensure models are loaded before initializing
        from agent.config import ensure_models_loaded, get_current_model, set_model_by_id
        ensure_models_loaded()
        
        # Get current model info
        current_model = get_current_model()
        if current_model:
            logger.info(f"Current model: {current_model.get('id')} ({current_model.get('name')})")
        else:
            logger.warning("No model available, using default")
        
        # Initialize CompletionsRouter
        logger.info("Initializing CompletionsRouter...")
        from agent.completions_router import CompletionsRouter
        llm = CompletionsRouter()
        
        logger.info(f"CompletionsRouter initialized with model: {llm.model_name}")
        logger.info(f"Max tokens: {llm.num_output}, Temperature: {llm.temperature}")
        logger.info(f"Timeout: {llm.timeout} seconds")
        
        # Test 1: Simple completion
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Simple completion")
        logger.info("="*80)
        
        test_prompt = "What is 2+2? Answer in one sentence."
        logger.info(f"Prompt: {test_prompt}")
        
        start_time = time.time()
        try:
            response = llm.complete(test_prompt)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Success! Response received in {elapsed:.2f} seconds")
            logger.info(f"Response: {response.text}")
            logger.info(f"Response length: {len(response.text)} characters")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return False
        
        # Test 2: Longer completion
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Longer completion")
        logger.info("="*80)
        
        test_prompt2 = "Write a brief 3-sentence explanation of what machine learning is."
        logger.info(f"Prompt: {test_prompt2}")
        
        start_time = time.time()
        try:
            response2 = llm.complete(test_prompt2)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Success! Response received in {elapsed:.2f} seconds")
            logger.info(f"Response: {response2.text}")
            logger.info(f"Response length: {len(response2.text)} characters")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return False
        
        # Test 3: Streaming completion (wrapped)
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Streaming completion")
        logger.info("="*80)
        
        test_prompt3 = "List 3 benefits of renewable energy."
        logger.info(f"Prompt: {test_prompt3}")
        
        start_time = time.time()
        try:
            stream_responses = list(llm.stream_complete(test_prompt3))
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Success! Stream completed in {elapsed:.2f} seconds")
            logger.info(f"Number of chunks: {len(stream_responses)}")
            full_response = "".join([r.text for r in stream_responses])
            logger.info(f"Full response: {full_response}")
            logger.info(f"Response length: {len(full_response)} characters")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return False
        
        # Test 4: Test with custom parameters
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Custom parameters")
        logger.info("="*80)
        
        test_prompt4 = "What is Python?"
        logger.info(f"Prompt: {test_prompt4}")
        
        start_time = time.time()
        try:
            # Note: kwargs might be passed but may not override model defaults
            response4 = llm.complete(test_prompt4, formatted=True)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Success! Response received in {elapsed:.2f} seconds")
            logger.info(f"Response: {response4.text}")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return False
        
        # Test 5: Error handling - empty prompt (should still work)
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Empty prompt handling")
        logger.info("="*80)
        
        test_prompt5 = ""
        logger.info(f"Prompt: (empty)")
        
        start_time = time.time()
        try:
            response5 = llm.complete(test_prompt5)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Success! Response received in {elapsed:.2f} seconds")
            logger.info(f"Response: {response5.text}")
        except Exception as e:
            logger.warning(f"⚠️  Expected behavior: {e}")
        
        # Test 6: Timeout test with shorter timeout
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Timeout handling (short timeout)")
        logger.info("="*80)
        
        test_prompt6 = "Write a detailed explanation of quantum computing."
        logger.info(f"Prompt: {test_prompt6}")
        logger.info("Using shorter timeout (10 seconds) to test timeout handling...")
        
        # Create a new instance with shorter timeout
        llm_short_timeout = CompletionsRouter()
        llm_short_timeout.timeout = 10  # 10 second timeout
        
        start_time = time.time()
        try:
            response6 = llm_short_timeout.complete(test_prompt6)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Success! Response received in {elapsed:.2f} seconds")
            logger.info(f"Response length: {len(response6.text)} characters")
        except requests.exceptions.Timeout:
            logger.warning("⚠️  Request timed out (expected with short timeout)")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            # Don't fail the test suite for timeout test
        
        logger.info("\n" + "="*80)
        logger.info("✅ All tests completed!")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return False


def test_model_switching():
    """Test model switching functionality."""
    
    logger.info("\n" + "="*80)
    logger.info("TEST: Model switching")
    logger.info("="*80)
    
    try:
        from agent.config import (
            ensure_models_loaded,
            get_available_models,
            get_current_model,
            set_model_by_id,
            FORCE_MODEL_ID
        )
        
        ensure_models_loaded()
        models = get_available_models()
        
        logger.info(f"Available models: {len(models)}")
        logger.info(f"Current forced model: {FORCE_MODEL_ID or 'None'}")
        
        if len(models) < 2:
            logger.warning("Need at least 2 models to test switching")
            return
        
        # Show first 5 models
        logger.info("\nFirst 5 available models:")
        for i, model in enumerate(models[:5]):
            logger.info(f"  {i+1}. {model.get('id')} - {model.get('name')}")
        
        # Test switching to first model
        first_model = models[0]
        logger.info(f"\nSwitching to model: {first_model.get('id')}")
        if set_model_by_id(first_model.get('id'), force=False):
            current = get_current_model()
            logger.info(f"✅ Current model: {current.get('id')}")
        else:
            logger.error("❌ Failed to switch model")
            return
        
        # Test with forced model
        logger.info(f"\nForcing model: {first_model.get('id')}")
        if set_model_by_id(first_model.get('id'), force=True):
            current = get_current_model()
            logger.info(f"✅ Current model (forced): {current.get('id')}")
            logger.info(f"⚠️  Note: Forced models will not failover on errors")
        else:
            logger.error("❌ Failed to force model")
            return
        
    except Exception as e:
        logger.error(f"❌ Error testing model switching: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Starting CompletionsRouter test script...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check for required environment variables
    api_key = os.getenv("COMPLETIONS_ROUTER_API_KEY")
    if not api_key:
        logger.error("❌ COMPLETIONS_ROUTER_API_KEY environment variable is not set!")
        logger.error("Please set it in your .env file or environment.")
        sys.exit(1)
    else:
        logger.info(f"✅ API key found (length: {len(api_key)} characters)")
    
    # Run pre-tests
    if not test_configuration():
        logger.error("❌ Configuration check failed!")
        sys.exit(1)
    
    # Test API connectivity (non-blocking)
    connectivity_ok = test_api_connectivity()
    if not connectivity_ok:
        logger.warning("⚠️  API connectivity check failed, but continuing with tests...")
    
    # Run tests
    success = test_completions_router()
    
    # Test model switching
    test_model_switching()
    
    if success:
        logger.info("\n✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed!")
        sys.exit(1)