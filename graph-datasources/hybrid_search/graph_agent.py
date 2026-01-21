
import sys
import os
import logging
import re
import asyncio
import time

# Ensure root directory is in python path to import 'agent'
# This path setup mirrors setup_router_rag.py to find the agent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
from tools import (
    graph_rag_tool,
    milvus_search_tool,
    tavily_search_tool,
    fmp_company_profile_tool,
    fmp_financial_statements_tool,
    fmp_financial_ratios_tool,
    fmp_stock_news_tool,
    fmp_analyst_estimates_tool,
    fmp_market_data_tool
)
import config
import os

# Ensure API key is set from environment variables
if not os.getenv("COMPLETIONS_ROUTER_API_KEY"):
    raise ValueError("COMPLETIONS_ROUTER_API_KEY environment variable is required. Please set it in .env.local or environment.")

try:
    from agent.completions_router import CompletionsRouter
except ImportError:
    print("\n❌ Could not import 'agent.completions_router'. Please check your project structure.")
    sys.exit(1)

# Set up logging to see the agent's thought process
# Use root logger configuration from main.py (no need to reconfigure)
logger = logging.getLogger(__name__)

# ReAct Agent Configuration (matching research-agent pattern)
MAX_ITERATIONS = 30  # Maximum number of reasoning steps for the ReAct agent
REACT_WORKFLOW_TIMEOUT = 2000  # Timeout for agent processing (seconds)

# Global agent instance for reuse
_agent_instance = None

def _create_agent_with_prompt(custom_system_prompt: str):
    """
    Create a new agent instance with a custom system prompt.
    Uses the same tools and configuration as the default agent.
    
    Args:
        custom_system_prompt: Custom system prompt string
    
    Returns:
        ReActAgent instance with custom prompt
    """
    try:
        # Ensure models are loaded before initializing
        from agent.config import ensure_models_loaded
        ensure_models_loaded()
        
        llm = CompletionsRouter()
        
        # Same tools as default agent
        tools = [
            graph_rag_tool, 
            milvus_search_tool, 
            tavily_search_tool,
            # FMP Proxy tools for financial data
            fmp_company_profile_tool,
            fmp_financial_statements_tool,
            fmp_financial_ratios_tool,
            fmp_stock_news_tool,
            fmp_analyst_estimates_tool,
            fmp_market_data_tool
        ]
        
        # Create agent with custom prompt
        agent = ReActAgent(
            tools=tools,
            llm=llm,
            system_prompt=custom_system_prompt,
            verbose=True,
            max_iterations=MAX_ITERATIONS
        )
        
        # Override the system prompt using update_prompts method
        try:
            prompts = agent.get_prompts()
            
            if "react_header" in prompts:
                agent.update_prompts({
                    "react_header": PromptTemplate(custom_system_prompt)
                })
            elif "agent_worker:system_prompt" in prompts:
                agent.update_prompts({
                    "agent_worker:system_prompt": PromptTemplate(custom_system_prompt)
                })
            else:
                # Try to override all possible keys
                override_dict = {}
                for key in prompts.keys():
                    if "system" in key.lower() or "header" in key.lower():
                        override_dict[key] = PromptTemplate(custom_system_prompt)
                if override_dict:
                    agent.update_prompts(override_dict)
        except Exception as e:
            logger.warning(f"Error overriding system prompt: {str(e)}")
            # Fallback: try to set directly
            try:
                prompts = agent.get_prompts()
                if "react_header" in prompts:
                    prompts["react_header"].template = custom_system_prompt
                elif "agent_worker:system_prompt" in prompts:
                    prompts["agent_worker:system_prompt"].template = custom_system_prompt
            except Exception as fallback_e:
                logger.warning(f"Fallback system prompt setting failed: {str(fallback_e)}")
        
        return agent
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to create agent with custom prompt: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise e

def get_agent():
    """Lazy initialize the agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        try:
            # Ensure models are loaded before initializing
            from agent.config import ensure_models_loaded
            ensure_models_loaded()
            
            print("Initializing Agent with CompletionsRouter...")
            llm = CompletionsRouter()
            
            # Create agent with tools using from_tools() method (matching research-agent pattern)
            tools = [
                    graph_rag_tool, 
                    milvus_search_tool, 
                    tavily_search_tool,
                # FMP Proxy tools for financial data
                fmp_company_profile_tool,      # Company profile and basic info
                fmp_financial_statements_tool, # Financial statements (income, balance sheet, cash flow)
                fmp_financial_ratios_tool,     # Financial ratios and key metrics
                fmp_stock_news_tool,           # Stock news and articles
                fmp_analyst_estimates_tool,     # Analyst estimates and price targets
                fmp_market_data_tool           # Historical market data (price, volume)
            ]
            
            # Build system prompt from CoT instructions
            cot_react_instructions = _get_cot_instructions()
            
            # Use ReActAgent constructor directly (from_tools() not available in this LlamaIndex version)
            _agent_instance = ReActAgent(
                tools=tools,
                llm=llm,
                system_prompt=cot_react_instructions,
                verbose=True,
                max_iterations=MAX_ITERATIONS
            )
            
            # Override the system prompt using update_prompts method (matching research-agent pattern)
            try:
                prompts = _agent_instance.get_prompts()
                
                # Check which system prompt key is available and override the correct one
                if "react_header" in prompts:
                    _agent_instance.update_prompts({
                        "react_header": PromptTemplate(cot_react_instructions)
                    })
                    logger.info("Successfully overrode system prompt using react_header key")
                elif "agent_worker:system_prompt" in prompts:
                    _agent_instance.update_prompts({
                        "agent_worker:system_prompt": PromptTemplate(cot_react_instructions)
                    })
                    logger.info("Successfully overrode system prompt using agent_worker:system_prompt key")
                else:
                    logger.warning(f"Could not find a known system prompt key. Available keys: {list(prompts.keys())}")
                    # Try to override all possible keys
                    override_dict = {}
                    for key in prompts.keys():
                        if "system" in key.lower() or "header" in key.lower():
                            override_dict[key] = PromptTemplate(cot_react_instructions)
                    if override_dict:
                        _agent_instance.update_prompts(override_dict)
                        logger.info(f"Attempted to override system prompts using keys: {list(override_dict.keys())}")
            except Exception as e:
                logger.error(f"Error overriding system prompt: {str(e)}")
                # Fallback: try to set directly
                try:
                    prompts = _agent_instance.get_prompts()
                    if "react_header" in prompts:
                        prompts["react_header"].template = cot_react_instructions
                        logger.info("Successfully set custom system prompt via direct template assignment (react_header)")
                    elif "agent_worker:system_prompt" in prompts:
                        prompts["agent_worker:system_prompt"].template = cot_react_instructions
                        logger.info("Successfully set custom system prompt via direct template assignment (agent_worker:system_prompt)")
                    else:
                        logger.warning(f"Could not find system prompt template. Available keys: {list(prompts.keys())}")
                except Exception as fallback_e:
                    logger.error(f"Fallback system prompt setting also failed: {str(fallback_e)}")
            
            # Log the agent's system prompt for verification
            try:
                prompts = _agent_instance.get_prompts()
                if "react_header" in prompts:
                    logger.info("=== AGENT SYSTEM PROMPT VERIFICATION ===")
                    logger.info(f"System prompt template: {prompts['react_header'].template[:500]}...")
                    logger.info("=== END SYSTEM PROMPT ===")
                else:
                    logger.warning("Could not retrieve react_header prompt template")
            except Exception as e:
                logger.warning(f"Could not retrieve agent prompts: {str(e)}")
            
        except Exception as e:
            import traceback
            print(f"Failed to initialize CompletionsRouter: {e}")
            print(f"Full traceback:")
            traceback.print_exc()
            raise e
    return _agent_instance

def _get_cot_instructions() -> str:
    """Get Chain of Thought optimized ReAct instructions."""
    return """\
You are a Hybrid Search agent with access to multiple data sources:

DATA SOURCES:
- graph_search: Companies, earnings, relationships, financial data
- milvus_search: Documents, semantic search
- tavily_search: Real-time news, market updates
- FMP TOOLS: Structured financial data (profile, statements, ratios, news, estimates, market data)

DATE FILTERING:
- Milvus: metadata["date"] >= "YYYY-MM-DD" and metadata["date"] <= "YYYY-MM-DD"
- Graph: date range "YYYY-MM-DD to YYYY-MM-DD"
- Historical: metadata["ticker"] == "SYMBOL"

AVAILABLE TOOLS:
{tool_desc}

TOOL NAMES: {tool_names}

REACT FORMAT:
Thought: [Reasoning - understand, plan, execute]
Action: <tool_name>
Action Input: <JSON>
Observation: <result>
[Repeat until complete]
Final Answer: [Synthesized answer]

KEY PRINCIPLES:
- Think step-by-step, plan before executing
- Decompose complex queries
- Analyze results before proceeding
- Synthesize findings with clear reasoning
"""

def clean_llm_response(response: str) -> str:
    """Remove content between <think> and </think> tags from the response."""
    try:
        import re
        # Remove all content between <think> and </think> tags
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Clean up any extra whitespace
        cleaned_response = cleaned_response.strip()
        return cleaned_response
    except Exception as e:
        logger.error(f"Error cleaning LLM response: {e}")
        return response

async def run_agent_query_async(query: str, return_tool_usage: bool = False, custom_system_prompt: str = None):
    """
    Run a single query against the agent and return the response (async version).
    Uses the correct ReActAgent async API pattern with direct await.
    
    Args:
        query: The user query string
        return_tool_usage: If True, returns tuple (response, tool_usage_dict)
        custom_system_prompt: Optional custom system prompt. If provided, creates a new agent instance
                             with this prompt. If None, uses the default singleton agent.
    
    Returns:
        Response string, or tuple (response, tool_usage) if return_tool_usage=True
    """
    try:
        # Use custom prompt agent if provided, otherwise use default singleton
        if custom_system_prompt is not None:
            agent = _create_agent_with_prompt(custom_system_prompt)
            logger.info("Using agent with custom system prompt")
        else:
            agent = get_agent()
            logger.info("Using default agent instance")
        
        # Track tool usage by intercepting tool calls
        tool_usage = {
            'graph_search': 0, 
            'milvus_search': 0, 
            'tavily_search': 0, 
            'get_company_profile': 0,
            'get_financial_statements': 0,
            'get_financial_ratios': 0,
            'get_stock_news': 0,
            'get_analyst_estimates': 0,
            'get_market_data': 0
        }
        
        # Start execution time tracking
        begin_time = time.time()
        
        # Use ReActAgent's run method (async API - returns a handler)
        try:
            logger.info(f"Calling agent.run() with query: {query[:100]}...")
            
            # Call agent.run() which returns a handler
            handler = agent.run(user_msg=query)
            
            # Execute the workflow by consuming all events
            # This actually runs the workflow and tracks tool calls
            final_event = None
            event_count = 0
            tool_call_count = 0
            async for event in handler.stream_events():
                final_event = event
                event_count += 1
                event_type = type(event).__name__
                logger.debug(f"Received event {event_count}: {event_type}")
                
                # Track tool calls from events
                # Check for tool-related events
                if hasattr(event, 'tool_name') or (hasattr(event, 'event') and hasattr(event.event, 'tool_name')):
                    tool_name = getattr(event, 'tool_name', None) or getattr(getattr(event, 'event', None), 'tool_name', None)
                    if tool_name:
                        tool_call_count += 1
                        logger.info(f"Tool call detected: {tool_name}")
                        # Map tool names to our tracking dict
                        tool_name_mapped = tool_name
                        if tool_name in tool_usage:
                            tool_usage[tool_name] += 1
                        elif 'graph' in tool_name.lower() or 'rag' in tool_name.lower():
                            tool_usage['graph_search'] += 1
                        elif 'milvus' in tool_name.lower():
                            tool_usage['milvus_search'] += 1
                        elif 'tavily' in tool_name.lower():
                            tool_usage['tavily_search'] += 1
                        elif 'company_profile' in tool_name.lower():
                            tool_usage['get_company_profile'] += 1
                        elif 'financial_statements' in tool_name.lower():
                            tool_usage['get_financial_statements'] += 1
                        elif 'financial_ratios' in tool_name.lower():
                            tool_usage['get_financial_ratios'] += 1
                        elif 'stock_news' in tool_name.lower():
                            tool_usage['get_stock_news'] += 1
                        elif 'analyst_estimates' in tool_name.lower():
                            tool_usage['get_analyst_estimates'] += 1
                        elif 'market_data' in tool_name.lower():
                            tool_usage['get_market_data'] += 1
                
                # Continue until workflow is done
                if handler.is_done():
                    logger.debug("Handler is done, breaking event loop")
                    break
            
            logger.info(f"Finished streaming events. Total events: {event_count}, Tool calls detected: {tool_call_count}, Handler done: {handler.is_done()}")
            
            # Wait for handler to be done (with timeout)
            max_wait = REACT_WORKFLOW_TIMEOUT  # seconds
            wait_count = 0
            while not handler.is_done() and wait_count < max_wait * 10:
                await asyncio.sleep(0.1)
                wait_count += 1
            
            if not handler.is_done():
                logger.warning(f"Handler not done after {max_wait} seconds")
            
            # Direct await pattern - await the handler directly (matches successful test pattern)
            logger.info("Awaiting handler result...")
            response = await handler
            
            logger.info(f"Got response type: {type(response)}")
            
            # Extract answer using str(response) - this is the pattern that works for AgentOutput
            # Based on test results: Response Type is AgentOutput, extraction method is str(response)
            response_str = str(response)
            logger.info(f"Extracted response using str(response): {len(response_str)} chars")
            
            # Clean the response
            response_str = clean_llm_response(response_str)
            logger.info(f"Final response length: {len(response_str)} characters")
            
            # Track tool usage from response if available (supplement event tracking)
            if hasattr(response, 'sources'):
                logger.info(f"Agent response has {len(response.sources)} sources")
                logger.info(f"Source tool names: {[getattr(source, 'tool_name', 'unknown') for source in response.sources]}")
                for source in response.sources:
                    tool_name = getattr(source, 'tool_name', None)
                    if tool_name and tool_name in tool_usage:
                        tool_usage[tool_name] += 1
            elif hasattr(response, 'tool_calls'):
                logger.info(f"Agent response has {len(response.tool_calls)} tool_calls")
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('tool_name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'tool_name', '')
                    if tool_name and tool_name in tool_usage:
                        tool_usage[tool_name] += 1
            
            # Log final tool usage summary
            total_tool_calls = sum(tool_usage.values())
            if total_tool_calls > 0:
                logger.info(f"Total tool calls: {total_tool_calls}")
                used_tools = {k: v for k, v in tool_usage.items() if v > 0}
                logger.info(f"Tools used: {used_tools}")
            else:
                logger.warning("⚠️  NO TOOLS WERE CALLED! The agent may have hallucinated data.")
            
        except asyncio.TimeoutError:
            logger.error(f"Agent processing timed out after {REACT_WORKFLOW_TIMEOUT} seconds")
            response_str = "I apologize, but the query is taking longer than expected to process. Please try simplifying your request or breaking it into smaller parts."
        except Exception as agent_error:
            logger.error(f"Error during agent processing: {str(agent_error)}", exc_info=True)
            # Check if it's a max iterations error specifically
            if "max iterations" in str(agent_error).lower():
                logger.warning(f"Agent reached max iterations ({MAX_ITERATIONS}). Consider increasing MAX_ITERATIONS in config.")
                response_str = f"I apologize, but your query is quite complex and requires more processing time than currently allowed. The system reached its maximum of {MAX_ITERATIONS} reasoning steps. Please try breaking your question into smaller, more specific parts."
            else:
                # Re-raise other errors
                raise agent_error
        
        execution_time = time.time() - begin_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        if return_tool_usage:
            return response_str, tool_usage
        return response_str
        
    except Exception as e:
        logger.error(f"Error querying agent: {e}", exc_info=True)
        error_msg = f"Error: {str(e)}"
        if return_tool_usage:
            return error_msg, {
                'graph_search': 0, 
                'milvus_search': 0, 
                'tavily_search': 0,
                'get_company_profile': 0,
                'get_financial_statements': 0,
                'get_financial_ratios': 0,
                'get_stock_news': 0,
                'get_analyst_estimates': 0,
                'get_market_data': 0
            }
        return error_msg

def run_agent_query(query: str, return_tool_usage: bool = False, custom_system_prompt: str = None) -> str:
    """
    Run a single query against the agent and return the response (synchronous wrapper).
    Useful for benchmarking and programmatic access.
    
    Args:
        query: The user query string
        return_tool_usage: If True, returns tuple (response, tool_usage_dict)
        custom_system_prompt: Optional custom system prompt. If provided, creates a new agent instance
                             with this prompt. If None, uses the default singleton agent.
    
    Returns:
        Response string, or tuple (response, tool_usage) if return_tool_usage=True
    """
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(run_agent_query_async(query, return_tool_usage, custom_system_prompt))
            except ImportError:
                logger.error("nest_asyncio not available. Install it with: pip install nest-asyncio")
                return "Error: Cannot run async agent in sync context. Please install nest-asyncio or run in async context."
        except RuntimeError:
            # No running loop, we can use asyncio.run()
            return asyncio.run(run_agent_query_async(query, return_tool_usage, custom_system_prompt))
    except Exception as e:
        logger.error(f"Error in run_agent_query wrapper: {e}", exc_info=True)
        error_msg = f"Error: {str(e)}"
        if return_tool_usage:
            return error_msg, {
                'graph_search': 0, 
                'milvus_search': 0, 
                'tavily_search': 0, 
                'get_company_profile': 0,
                'get_financial_statements': 0,
                'get_financial_ratios': 0,
                'get_stock_news': 0,
                'get_analyst_estimates': 0,
                'get_market_data': 0
            }
        return error_msg

def main():
    # Initialize implementation
    try:
        get_agent()
    except:
        return

    print("\n=== Hybrid Search Agent ===")
    print("The agent has access to:")
    print("  - 'graph_search' tool: Search Memgraph graph database")
    print("  - 'milvus_search' tool: Hybrid search in Milvus vector database")
    print("  - 'tavily_search' tool: Real-time web search via Tavily")
    print("\nTry asking:")
    print("  - 'Who is the CEO of Apple based on the database?'")
    print("  - 'Search for documents about Tesla revenue'")
    print("  - 'What are the latest news about Tesla stock?'")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("User: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue

            # The agent will decide whether to use the tool or answer directly
            response = run_agent_query(user_input)
            print(f"Agent: {response}\n")
            
        except Exception as e:
            logger.error(f"Error during chat: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
