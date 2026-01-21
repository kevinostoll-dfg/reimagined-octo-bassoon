#!/usr/bin/env python3
"""
Test script to determine which ReActAgent pattern we're using:
- Pattern 1: Simple ReActAgent - str(response) gives the answer
- Pattern 2: Workflow-style - response["response"] gives the answer
"""

import sys
import os
import asyncio
import logging

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure API key is set
if not os.getenv("COMPLETIONS_ROUTER_API_KEY"):
    logger.warning("COMPLETIONS_ROUTER_API_KEY not set. The test may fail.")
    logger.warning("Please set it in your environment or .env file.")

from graph_agent import get_agent

async def test_pattern_1_simple():
    """Test Pattern 1: Simple ReActAgent - str(response) gives the answer"""
    logger.info("=" * 60)
    logger.info("Testing Pattern 1: Simple ReActAgent (str(response))")
    logger.info("=" * 60)
    
    try:
        agent = get_agent()
        query = "Who is the CEO of Apple?"
        
        logger.info(f"Query: {query}")
        logger.info("Calling agent.run()...")
        
        # Pattern 1: Simple ReActAgent
        handler = agent.run(user_msg=query)
        
        logger.info("Streaming events...")
        event_count = 0
        async for event in handler.stream_events():
            event_count += 1
            logger.debug(f"Event {event_count}: {type(event).__name__}")
            if handler.is_done():
                break
        
        logger.info(f"Total events: {event_count}")
        logger.info("Waiting for handler result...")
        
        response = await handler.result()
        
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response repr: {repr(response)[:200]}")
        
        # Try Pattern 1: str(response)
        answer_text = str(response)
        logger.info(f"Pattern 1 result (str(response)): {answer_text[:200]}...")
        
        return {
            'pattern': 'Pattern 1 (Simple)',
            'success': True,
            'response_type': str(type(response)),
            'answer': answer_text[:500],
            'full_response': str(response),
            'extraction_method': 'str(response)'
        }
        
    except Exception as e:
        logger.error(f"Pattern 1 failed: {e}", exc_info=True)
        return {
            'pattern': 'Pattern 1 (Simple)',
            'success': False,
            'error': str(e)
        }

async def test_pattern_2_workflow():
    """Test Pattern 2: Workflow-style - response["response"] gives the answer"""
    logger.info("=" * 60)
    logger.info("Testing Pattern 2: Workflow-style (response['response'])")
    logger.info("=" * 60)
    
    try:
        agent = get_agent()
        query = "Who is the CEO of Apple?"
        
        logger.info(f"Query: {query}")
        logger.info("Calling agent.run()...")
        
        # Pattern 2: Workflow-style
        handler = agent.run(user_msg=query)
        
        logger.info("Streaming events...")
        event_count = 0
        async for event in handler.stream_events():
            event_count += 1
            logger.debug(f"Event {event_count}: {type(event).__name__}")
            if handler.is_done():
                break
        
        logger.info(f"Total events: {event_count}")
        logger.info("Waiting for handler result...")
        
        response = await handler.result()
        
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response repr: {repr(response)[:200]}")
        
        # Try Pattern 2: response["response"]
        if isinstance(response, dict) or hasattr(response, '__getitem__'):
            try:
                answer_text = response["response"]
                logger.info(f"Pattern 2 result (response['response']): {answer_text[:200]}...")
                return {
                    'pattern': 'Pattern 2 (Workflow)',
                    'success': True,
                    'response_type': str(type(response)),
                    'answer': str(answer_text)[:500],
                    'full_response': str(response)
                }
            except (KeyError, TypeError) as e:
                logger.warning(f"Pattern 2 failed: {e}")
                # Try to see what keys/attributes are available
                if isinstance(response, dict):
                    logger.info(f"Available keys: {list(response.keys())}")
                else:
                    logger.info(f"Response attributes: {dir(response)}")
                return {
                    'pattern': 'Pattern 2 (Workflow)',
                    'success': False,
                    'error': f"Cannot access response['response']: {e}",
                    'response_type': str(type(response))
                }
        else:
            logger.warning("Response is not dict-like, cannot use Pattern 2")
            return {
                'pattern': 'Pattern 2 (Workflow)',
                'success': False,
                'error': 'Response is not dict-like',
                'response_type': str(type(response))
            }
            
    except Exception as e:
        logger.error(f"Pattern 2 failed: {e}", exc_info=True)
        return {
            'pattern': 'Pattern 2 (Workflow)',
            'success': False,
            'error': str(e)
        }

async def test_direct_await():
    """Test direct await handler (simplest pattern from knowledge base)"""
    logger.info("=" * 60)
    logger.info("Testing Direct Await Pattern: response = await handler")
    logger.info("=" * 60)
    
    try:
        agent = get_agent()
        query = "Who is the CEO of Apple?"
        
        logger.info(f"Query: {query}")
        logger.info("Calling agent.run()...")
        
        handler = agent.run(user_msg=query)
        
        logger.info("Streaming events (optional)...")
        event_count = 0
        async for event in handler.stream_events():
            event_count += 1
            logger.debug(f"Event {event_count}: {type(event).__name__}")
            if handler.is_done():
                break
        
        logger.info(f"Total events: {event_count}")
        logger.info("Directly awaiting handler (simplest pattern)...")
        
        # Direct await pattern - this is the simplest according to knowledge base
        response = await handler
        
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response repr: {repr(response)[:200]}")
        
        # Try str(response) first (Pattern 1)
        answer_text = str(response)
        logger.info(f"Direct await result (str(response)): {answer_text[:200]}...")
        
        extraction_method = 'str(response)'
        
        # Try response["response"] if dict-like (Pattern 2)
        if isinstance(response, dict) or hasattr(response, '__getitem__'):
            try:
                dict_answer = response["response"]
                logger.info(f"Direct await result (response['response']): {str(dict_answer)[:200]}...")
                extraction_method = 'response["response"]'
                answer_text = str(dict_answer)
            except (KeyError, TypeError):
                logger.info("Response is dict-like but doesn't have 'response' key")
        
        return {
            'pattern': 'Direct Await',
            'success': True,
            'response_type': str(type(response)),
            'answer': answer_text[:500],
            'full_response': str(response),
            'extraction_method': extraction_method
        }
        
    except Exception as e:
        logger.error(f"Direct await failed: {e}", exc_info=True)
        return {
            'pattern': 'Direct Await',
            'success': False,
            'error': str(e)
        }

async def main():
    """Run all tests and report results"""
    logger.info("\n" + "=" * 60)
    logger.info("ReActAgent Pattern Detection Test")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # Test Pattern 1: Simple ReActAgent
    result1 = await test_pattern_1_simple()
    results.append(result1)
    
    logger.info("\n")
    
    # Test Pattern 2: Workflow-style
    result2 = await test_pattern_2_workflow()
    results.append(result2)
    
    logger.info("\n")
    
    # Test Direct Await
    result3 = await test_direct_await()
    results.append(result3)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(f"\n{result['pattern']}:")
        logger.info(f"  Success: {result['success']}")
        if result['success']:
            logger.info(f"  Response Type: {result['response_type']}")
            logger.info(f"  Answer Preview: {result['answer'][:200]}...")
        else:
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Determine which pattern works
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED PATTERN")
    logger.info("=" * 60)
    
    successful_patterns = [r for r in results if r['success']]
    if successful_patterns:
        # Prefer Direct Await if it worked, otherwise use the first successful one
        best_pattern = next((r for r in successful_patterns if r['pattern'] == 'Direct Await'), successful_patterns[0])
        logger.info(f"✅ Use: {best_pattern['pattern']}")
        logger.info(f"   Response Type: {best_pattern['response_type']}")
        logger.info(f"   Answer Extraction: {best_pattern.get('extraction_method', 'str(response)')}")
        logger.info(f"\n   Code pattern:")
        logger.info(f"   handler = agent.run(user_msg=query)")
        logger.info(f"   async for event in handler.stream_events():")
        logger.info(f"       if handler.is_done(): break")
        logger.info(f"   response = await handler")
        logger.info(f"   answer = {best_pattern.get('extraction_method', 'str(response)')}")
    else:
        logger.error("❌ No pattern succeeded. Check errors above.")
    
    logger.info("\n")

if __name__ == "__main__":
    asyncio.run(main())

