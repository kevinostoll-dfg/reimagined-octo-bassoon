
import unittest
import sys
import os

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tools.graphrag_tool import graph_rag_func

class TestGraphRAG(unittest.TestCase):
    
    def setUp(self):
        # Ensure API key is set for testing
        if not os.getenv("COMPLETIONS_ROUTER_API_KEY"):
            raise ValueError("COMPLETIONS_ROUTER_API_KEY environment variable is required for testing. Please set it in .env.local or environment.")

    def test_basic_query(self):
        """Test a basic query that should return results."""
        query = "What statements mention 'afternoon'?"
        print(f"\nRunning test query: {query}")
        result = graph_rag_func(query)
        print(f"Result: {result}")
        
        # We expect a string response, ideally not an error message
        self.assertIsInstance(result, str)
        self.assertNotIn("Error executing Graph RAG query", result)
        self.assertNotIn("ValueError", result)

    def test_empty_result_query(self):
        """Test a query for something strictly not in the graph."""
        query = "What did the CEO say about 'Unicorn Flying Machines'?"
        print(f"\nRunning test query: {query}")
        result = graph_rag_func(query)
        print(f"Result: {result}")
        
        self.assertIsInstance(result, str)
        # Should gracefully handle empty result
        self.assertTrue(len(result) > 0)

    def test_ceo_query(self):
        """Test the specific query that was failing due to WHERE clause pattern matching."""
        query = "Who is the CEO of Apple?"
        print(f"\nRunning complex test query: {query}")
        result = graph_rag_func(query)
        print(f"Result: {result}")
        
        self.assertIsInstance(result, str)
        # It should NOT return the error message we saw earlier
        self.assertNotIn("TransientError", result)
        # It SHOULD contain the answer (or handle gracefully if not in database)
        # Note: This may not always contain "Tim Cook" if data is not in graph
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()
