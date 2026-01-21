
import time
import logging
from typing import Any
import requests
from pydantic import Field
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata
)

from agent.config import (
    LOGGING_LEVEL,
    COMPLETIONS_ROUTER_API_KEY,
    COMPLETIONS_ROUTER_URL,
    MAX_RETRIES,
    get_model_id,
    get_model_max_tokens,
    get_model_temperature,
    DEFAULT_TIMEOUT,
    record_model_failure,
    record_model_success,
    get_current_model
)

# Use root logger configuration from main.py (no need to reconfigure)
logger = logging.getLogger(__name__)


class CompletionsRouter(CustomLLM):
    """LLM (uses Blacksmith completion router)"""
    model_name: str = Field(default="fallback-model", description="The model name to use")
    num_output: int = Field(default=20000, description="Maximum number of output tokens")
    temperature: float = Field(default=0.7, description="Temperature for text generation")
    timeout: int = Field(default=DEFAULT_TIMEOUT, description="Request timeout in seconds")
    
    def __init__(self, **kwargs):
        # Get dynamic model configuration
        model_id = get_model_id()
        max_tokens = get_model_max_tokens()
        temperature = get_model_temperature()
        
        # Set dynamic values in kwargs
        kwargs.setdefault('model_name', model_id if model_id else "fallback-model")
        kwargs.setdefault('num_output', max_tokens)
        kwargs.setdefault('temperature', temperature)
        
        logger.info(f"CompletionsRouter initialized with model: {kwargs.get('model_name')}, max_tokens: {kwargs.get('num_output')}, temperature: {kwargs.get('temperature')}")
        
        super().__init__(**kwargs)
    
    def _update_model_config(self):
        """Update model configuration from config.py"""
        model_id = get_model_id()
        max_tokens = get_model_max_tokens()
        temperature = get_model_temperature()
        
        if model_id:
            # Update the model attributes using object.__setattr__ to bypass Pydantic validation
            object.__setattr__(self, 'model_name', model_id)
            object.__setattr__(self, 'num_output', max_tokens)
            object.__setattr__(self, 'temperature', temperature)
            logger.info(f"Updated model configuration: {self.model_name}, max_tokens: {max_tokens}, temperature: {temperature}")
        else:
            logger.warning("No model selected from models API, using fallback configuration")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.num_output,
            model_name=self.model_name,
            context_window=32768,  # Set context window for qwen models
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt using the Blacksmith completion router with failover support."""
        begin_time = time.time()
        retry_delay = 1.5
        
        # Log any additional kwargs passed to complete()
        if kwargs:
            logger.info(f"Additional kwargs passed to complete(): {kwargs}")

        while True:  # Loop for failover attempts
            # Update model configuration before each request
            self._update_model_config()
            
            # Check if we have a valid model
            if not self.model_name or self.model_name == "fallback-model":
                logger.error("No valid model available for completion")
                raise Exception("No valid model available for completion")
            
            # Check if API key is set
            if not COMPLETIONS_ROUTER_API_KEY:
                logger.error("COMPLETIONS_ROUTER_API_KEY is not set!")
                raise Exception("COMPLETIONS_ROUTER_API_KEY environment variable is not set")
            
            logger.info(f"Calling the Completions Router API with model: {self.model_name}")
            logger.info(f"API URL: {COMPLETIONS_ROUTER_URL}")
            
            # Prepare request payload
            request_payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.num_output,
                "temperature": self.temperature
            }
            
            request_headers = {
                "Authorization": f"Bearer {COMPLETIONS_ROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Log full request details
            logger.info("=== COMPLETIONS ROUTER REQUEST ===")
            logger.info(f"Request URL: {COMPLETIONS_ROUTER_URL}")
            logger.info(f"Request Method: POST")
            logger.info(f"Request Headers: Authorization: Bearer [REDACTED], Content-Type: application/json")
            logger.info(f"Request Payload:")
            logger.info(f"  - model: {request_payload['model']}")
            logger.info(f"  - max_tokens: {request_payload['max_tokens']}")
            logger.info(f"  - temperature: {request_payload['temperature']}")
            logger.info(f"  - messages: {len(request_payload['messages'])} message(s)")
            for idx, msg in enumerate(request_payload['messages']):
                logger.info(f"    Message {idx + 1}:")
                logger.info(f"      - role: {msg['role']}")
                logger.info(f"      - content length: {len(msg['content'])} characters")
                logger.info(f"      - content preview (first 500 chars): {msg['content'][:500]}")
                if len(msg['content']) > 500:
                    logger.info(f"      - content preview (last 500 chars): ...{msg['content'][-500:]}")
                logger.info(f"      - full content:")
                logger.info(f"{msg['content']}")
            logger.info("=== END COMPLETIONS ROUTER REQUEST ===")

            # Track if the last error was a network error
            last_error_was_network = False
            
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Making request attempt {attempt + 1}/{MAX_RETRIES}")
                    response = requests.post(
                        url=COMPLETIONS_ROUTER_URL,
                        headers=request_headers,
                        json=request_payload,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        response_json = response.json()
                        text = response_json["choices"][0]["message"]["content"]
                        
                        # Check for empty response
                        if not text or not text.strip():
                            logger.warning(f"Received empty response from model {self.model_name}")
                            if attempt < MAX_RETRIES - 1:
                                logger.info(f"Retrying due to empty response in {retry_delay} seconds")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            else:
                                logger.error(f"All retry attempts returned empty responses for model {self.model_name}")
                                break
                        
                        logger.debug(f"Completions Router API response: {text}")
                        logger.debug(f"Full API response JSON: {response_json}")
                        # Log first 1000 chars of response for debugging ReAct format parsing
                        logger.info(f"API response preview (first 1000 chars): {text[:1000]}")
                        # Log format check for ReAct parsing
                        has_thought = "Thought:" in text or "thought:" in text.lower()
                        has_action = "Action:" in text or "action:" in text.lower()
                        logger.info(f"Format check - Has 'Thought:': {has_thought}, Has 'Action:': {has_action}")
                        elapsed_time = time.time() - begin_time
                        logger.info(f"Completions router took {elapsed_time:.2f} seconds to respond with {attempt + 1} attempt(s)")
                        
                        # Record successful completion
                        record_model_success()
                        return CompletionResponse(text=text)
                    else:
                        logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: Failed to call the Completions Router API")
                        logger.error(f"Status code: {response.status_code}")
                        logger.error(f"Response text: {response.text}")
                        logger.error(f"Request URL: {COMPLETIONS_ROUTER_URL}")
                        logger.error(f"API Key present: {bool(COMPLETIONS_ROUTER_API_KEY)}")
                        logger.error(f"Model: {self.model_name}")
                        
                        last_error_was_network = False  # HTTP error, not network error
                        
                        if attempt < MAX_RETRIES - 1:
                            logger.info(f"Retrying in {retry_delay} seconds")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            # All retries failed for this model
                            logger.error(f"All retry attempts failed for model {self.model_name}")
                            break
                            
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: Network error")
                    logger.debug(f"Exception: {e}")
                    last_error_was_network = True  # Mark as network error
                    
                    # Check if model is forced - if so, retry instead of breaking
                    from agent.config import FORCE_MODEL_ID
                    if FORCE_MODEL_ID and self.model_name == FORCE_MODEL_ID:
                        # Model is forced - retry instead of switching
                        if attempt < MAX_RETRIES - 1:
                            logger.info(f"Model {self.model_name} is forced - retrying in {retry_delay} seconds")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue  # Continue retrying
                        else:
                            logger.error(f"All retry attempts failed for forced model {self.model_name}")
                            break
                    else:
                        # Model not forced - switch immediately
                        logger.error(f"Network error detected for model {self.model_name} after {attempt + 1} attempt(s). Switching to next model immediately.")
                        break  # Break immediately to switch models
                except Exception as e:
                    logger.error(f"Unexpected error calling the Completions Router API with model {self.model_name}")
                    logger.debug(f"Exception: {e}")
                    last_error_was_network = False  # Other error, not network error
                    break
            
            # Record failure and check if we should try the next model
            # Pass is_network_error flag to trigger faster failover for network errors
            if record_model_failure(is_network_error=last_error_was_network):
                error_type = "network errors" if last_error_was_network else "consecutive failures"
                logger.info(f"Switching to next model due to {error_type}")
                continue  # Try with the next model
            else:
                # Check if model is forced - if so, retry the same model instead of failing
                from agent.config import FORCE_MODEL_ID
                if FORCE_MODEL_ID and self.model_name == FORCE_MODEL_ID:
                    logger.info(f"Model {self.model_name} is forced - retrying same model")
                    # Reset retry delay for new attempt
                    retry_delay = 1.5
                    continue  # Retry the forced model
                else:
                    # No more models available
                    logger.error("No more models available for failover")
                    raise Exception("All available models have failed. Please try again later.")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream a completion using the Blacksmith completion router."""
        # Since the API doesn't support streaming, we wrap the complete() call
        # and yield it as a single chunk
        try:
            response = self.complete(prompt, **kwargs)
            yield response
        except Exception as e:
            logger.error(f"Error in stream_complete: {e}", exc_info=True)
            # Yield an error response
            yield CompletionResponse(text=f"Error: {str(e)}")
