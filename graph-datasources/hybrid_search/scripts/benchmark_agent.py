"""
Enhanced Benchmark Agent for GraphRAG Hybrid Search System.

This module provides comprehensive benchmarking capabilities including:
- Multi-metric evaluation (semantic similarity, ROUGE, keyword coverage)
- Tool usage tracking and analysis
- Historical performance tracking
- Configuration file support
- Detailed reporting and export capabilities
"""

import sys
import os
import json
import time
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from tabulate import tabulate

# Ensure we can import the agent (now in scripts/ subdirectory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import Agent and Router
from graph_agent import run_agent_query
from agent.completions_router import CompletionsRouter

# Ensure API key is set from environment variables
if not os.getenv("COMPLETIONS_ROUTER_API_KEY"):
    raise ValueError("COMPLETIONS_ROUTER_API_KEY environment variable is required. Please set it in .env.local or environment.")

# Try to import optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed. Some metrics will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Semantic similarity will use fallback.")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("Warning: rouge-score not installed. ROUGE metrics will be unavailable.")


class EnhancedEvaluator:
    """Comprehensive answer evaluation with multiple metrics."""
    
    def __init__(self, router: CompletionsRouter):
        self.router = router
        self.semantic_model = None
        self.rouge_scorer = None
        
        # Initialize semantic model if available
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load semantic model: {e}")
        
        # Initialize ROUGE scorer if available
        if HAS_ROUGE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            except Exception as e:
                print(f"Warning: Could not initialize ROUGE scorer: {e}")
    
    def evaluate_answer(
        self, 
        question: str, 
        actual_answer: str, 
        expected_phrases: List[str],
        expected_answer: Optional[str] = None
    ) -> Dict[str, float]:
        """Comprehensive evaluation with multiple metrics."""
        
        metrics = {}
        
        # 1. Semantic Similarity Score
        if expected_answer and self.semantic_model:
            try:
                expected_embedding = self.semantic_model.encode([expected_answer])
                actual_embedding = self.semantic_model.encode([actual_answer])
                if HAS_NUMPY:
                    similarity = float(np.dot(expected_embedding[0], actual_embedding[0]))
                    metrics['semantic_similarity'] = similarity
                else:
                    # Fallback without numpy
                    metrics['semantic_similarity'] = 0.0
            except Exception as e:
                print(f"Warning: Semantic similarity calculation failed: {e}")
                metrics['semantic_similarity'] = 0.0
        else:
            metrics['semantic_similarity'] = 0.0
        
        # 2. ROUGE Scores (if expected answer provided)
        if expected_answer and self.rouge_scorer:
            try:
                rouge_scores = self.rouge_scorer.score(expected_answer, actual_answer)
                metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
                metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
            except Exception as e:
                print(f"Warning: ROUGE calculation failed: {e}")
                metrics['rouge1'] = 0.0
                metrics['rouge2'] = 0.0
                metrics['rougeL'] = 0.0
        else:
            metrics['rouge1'] = 0.0
            metrics['rouge2'] = 0.0
            metrics['rougeL'] = 0.0
        
        # 3. Keyword Coverage
        if expected_phrases:
            found_phrases = sum(1 for phrase in expected_phrases if phrase.lower() in actual_answer.lower())
            metrics['keyword_coverage'] = found_phrases / len(expected_phrases)
        else:
            metrics['keyword_coverage'] = 0.0
        
        # 4. LLM-based Evaluation
        metrics['llm_grade'] = self._llm_evaluate(question, actual_answer, expected_phrases)
        
        # 5. Answer Length Score
        answer_length = len(actual_answer)
        metrics['answer_length'] = answer_length
        # Score: optimal length between 50-500 characters
        if 50 <= answer_length <= 500:
            metrics['answer_length_score'] = 1.0
        elif answer_length < 50:
            metrics['answer_length_score'] = answer_length / 50.0  # Partial score
        else:
            metrics['answer_length_score'] = max(0.5, 1.0 - (answer_length - 500) / 1000.0)
        
        # 6. Hallucination Detection Score
        metrics['hallucination_score'] = self._detect_hallucinations(question, actual_answer)
        
        # Calculate overall score with adaptive weights
        # If expected_answer is not provided, reduce weights for semantic/rouge metrics
        has_expected_answer = expected_answer is not None and expected_answer.strip()
        
        if has_expected_answer:
            # Full weights when expected answer is available
            weights = {
                'semantic': 0.3,
                'rouge': 0.2,
                'keyword': 0.2,
                'llm': 0.2,
                'length': 0.05,
                'hallucination': 0.05
            }
        else:
            # Reduced weights when expected answer is not available
            weights = {
                'semantic': 0.0,  # Skip semantic similarity
                'rouge': 0.0,     # Skip ROUGE
                'keyword': 0.3,   # Increase keyword weight
                'llm': 0.4,       # Increase LLM grade weight
                'length': 0.15,   # Increase length weight
                'hallucination': 0.15  # Increase hallucination weight
            }
        
        metrics['overall_score'] = (
            metrics.get('semantic_similarity', 0) * weights['semantic'] +
            metrics.get('rouge1', 0) * weights['rouge'] +
            metrics['keyword_coverage'] * weights['keyword'] +
            metrics['llm_grade'] * weights['llm'] +
            metrics['answer_length_score'] * weights['length'] +
            metrics['hallucination_score'] * weights['hallucination']
        )
        
        return metrics
    
    def _llm_evaluate(self, question: str, answer: str, expected_phrases: List[str]) -> float:
        """LLM-based evaluation returning score 0-1."""
        prompt = (
            f"You are a strict judge evaluating an AI agent's answer.\n"
            f"Question: {question}\n"
            f"Agent Answer: {answer}\n"
            f"Expected Concepts/Key Phrases: {', '.join(expected_phrases)}\n\n"
            "Rate the answer on a scale of 0.0 to 1.0 where:\n"
            "- 1.0 = Perfect answer with all expected concepts\n"
            "- 0.7-0.9 = Good answer with most concepts or clear explanation of missing data\n"
            "- 0.4-0.6 = Partial answer with some concepts\n"
            "- 0.0-0.3 = Poor answer or completely off-topic\n\n"
            "Return ONLY a number between 0.0 and 1.0. Do not explain."
        )
        
        try:
            response = self.router.complete(prompt).text.strip()
            # Extract number from response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1 range
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else 1.0
                return max(0.0, min(1.0, score))
            return 0.5  # Default if parsing fails
        except Exception as e:
            print(f"Warning: LLM evaluation failed: {e}")
            # Fallback to keyword matching
            found_phrases = sum(1 for phrase in expected_phrases if phrase.lower() in answer.lower())
            return found_phrases / len(expected_phrases) if expected_phrases else 0.5
    
    def _detect_hallucinations(self, question: str, answer: str) -> float:
        """Detect if answer contains unsupported claims."""
        # Check for phrases indicating uncertainty or lack of data
        uncertainty_phrases = [
            "i don't know", "no information", "not available", "not found",
            "unable to find", "no data", "cannot find", "don't have"
        ]
        
        # If answer contains uncertainty phrases, it's less likely to be hallucinating
        answer_lower = answer.lower()
        has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
        
        # Check for overly confident but potentially wrong statements
        # This is a simple heuristic - could be enhanced with more sophisticated checks
        if has_uncertainty:
            return 0.9  # High score - acknowledging uncertainty is good
        else:
            # If answer is very short or very long without uncertainty, might be hallucinating
            if len(answer) < 20:
                return 0.6  # Short answers might be incomplete
            elif len(answer) > 1000:
                return 0.7  # Very long answers might contain irrelevant info
            else:
                return 0.8  # Reasonable length, assume good


class ToolUsageTracker:
    """Track tool usage patterns and statistics."""
    
    def __init__(self):
        self.tool_calls = defaultdict(int)
        self.tool_sequences = []
        self.tool_timings = defaultdict(list)
    
    def parse_agent_response(self, response: str, execution_time: float, tool_usage: Optional[Dict] = None):
        """Parse agent response to extract tool usage."""
        # Use provided tool_usage if available, otherwise detect from response
        if tool_usage:
            graph_calls = tool_usage.get('graph_search', 0)
            milvus_calls = tool_usage.get('milvus_search', 0)
        else:
            # Enhanced detection using multiple patterns
            # Direct mentions
            graph_calls = len(re.findall(r'graph_search|graph_rag', response, re.IGNORECASE))
            milvus_calls = len(re.findall(r'milvus_search', response, re.IGNORECASE))
            
            # Indirect indicators for graph search
            graph_indicators = [
                r'statements.*mention|mention.*statements',
                r'connected to|associated with',
                r'relationships.*between|entities.*connected',
                r'earnings.*transcript|transcript.*earnings',
                r'financial.*data.*database|database.*financial',
                r'graph.*database|memgraph'
            ]
            
            # Indirect indicators for milvus search
            milvus_indicators = [
                r'vector database|documents.*found|found \d+ documents',
                r'semantic.*search|search results.*documents',
                r'documents include|documents.*contain',
                r'I found \d+ documents|found.*documents.*vector'
            ]
            
            # Check for graph search patterns
            for pattern in graph_indicators:
                if re.search(pattern, response, re.IGNORECASE):
                    graph_calls = max(graph_calls, 1)
                    break
            
            # Check for milvus search patterns
            for pattern in milvus_indicators:
                if re.search(pattern, response, re.IGNORECASE):
                    milvus_calls = max(milvus_calls, 1)
                    break
        
        self.tool_calls['graph_search'] += graph_calls
        self.tool_calls['milvus_search'] += milvus_calls
        
        # Track sequence
        sequence = []
        if graph_calls > 0:
            sequence.append('graph_search')
        if milvus_calls > 0:
            sequence.append('milvus_search')
        self.tool_sequences.append(sequence)
        
        # Track timing per tool (rough estimate)
        total_calls = graph_calls + milvus_calls
        if total_calls > 0:
            avg_time_per_call = execution_time / total_calls
            if graph_calls > 0:
                self.tool_timings['graph_search'].append(avg_time_per_call)
            if milvus_calls > 0:
                self.tool_timings['milvus_search'].append(avg_time_per_call)
    
    def get_statistics(self) -> Dict:
        """Get tool usage statistics."""
        graph_only = sum(1 for seq in self.tool_sequences if seq == ['graph_search'])
        milvus_only = sum(1 for seq in self.tool_sequences if seq == ['milvus_search'])
        both = sum(1 for seq in self.tool_sequences if len(seq) == 2)
        
        avg_timings = {}
        if HAS_NUMPY:
            for tool, timings in self.tool_timings.items():
                avg_timings[tool] = float(np.mean(timings)) if timings else 0.0
        else:
            for tool, timings in self.tool_timings.items():
                avg_timings[tool] = sum(timings) / len(timings) if timings else 0.0
        
        return {
            'total_calls': dict(self.tool_calls),
            'avg_timings': avg_timings,
            'usage_patterns': {
                'graph_only': graph_only,
                'milvus_only': milvus_only,
                'both_tools': both,
                'total_queries': len(self.tool_sequences)
            }
        }


class PerformanceTracker:
    """Track historical performance across benchmark runs."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def save_run(self, results: List[Dict], metadata: Dict):
        """Save benchmark run to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"benchmark_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'metadata': metadata,
            'results': results,
            'summary': self._calculate_summary(results)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename
    
    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics for a run."""
        passed = sum(1 for r in results if r.get('status') == 'PASS')
        failed = sum(1 for r in results if r.get('status') == 'FAIL')
        errors = sum(1 for r in results if r.get('status') == 'ERROR')
        
        latencies = [r['latency'] for r in results if 'latency' in r]
        scores = [r['evaluation']['overall_score'] for r in results if 'evaluation' in r]
        
        summary = {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'accuracy': passed / len(results) if results else 0
        }
        
        if latencies:
            if HAS_NUMPY:
                summary['avg_latency'] = float(np.mean(latencies))
                summary['p95_latency'] = float(np.percentile(latencies, 95))
                summary['p99_latency'] = float(np.percentile(latencies, 99))
            else:
                latencies_sorted = sorted(latencies)
                summary['avg_latency'] = sum(latencies) / len(latencies)
                summary['p95_latency'] = latencies_sorted[int(len(latencies) * 0.95)] if latencies_sorted else 0
                summary['p99_latency'] = latencies_sorted[int(len(latencies) * 0.99)] if latencies_sorted else 0
        
        if scores:
            if HAS_NUMPY:
                summary['avg_score'] = float(np.mean(scores))
                summary['min_score'] = float(np.min(scores))
                summary['max_score'] = float(np.max(scores))
            else:
                summary['avg_score'] = sum(scores) / len(scores)
                summary['min_score'] = min(scores)
                summary['max_score'] = max(scores)
        
        return summary
    
    def load_history(self, days: int = 30) -> List[Dict]:
        """Load historical results from the last N days."""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        history = []
        
        for file in self.results_dir.glob("benchmark_*.json"):
            try:
                file_time = file.stat().st_mtime
                if file_time >= cutoff_date:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        history.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        
        # Sort by timestamp
        history.sort(key=lambda x: x.get('timestamp', ''))
        return history


class EnhancedAgentBenchmarker:
    """Enhanced benchmark agent with comprehensive evaluation and tracking."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.router = CompletionsRouter()
        self.evaluator = EnhancedEvaluator(self.router)
        self.tool_tracker = ToolUsageTracker()
        self.performance_tracker = PerformanceTracker()
        
        # Load dataset from config file or use default
        self.dataset = self._load_dataset(config_path)
        self.results = []
        self.metrics_history = []
    
    def _load_dataset(self, config_path: Optional[str]) -> List[Dict]:
        """Load dataset from config file or use default."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('test_cases', self._get_default_dataset())
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
                print("Using default dataset instead.")
        
        return self._get_default_dataset()
    
    def _get_default_dataset(self) -> List[Dict]:
        """Get default test dataset."""
        return [
            {
                "question": "What is AL GORE connected to?",
                "expected_phrases": ["James A. Bell", "Timothy D. Cook", "Securities Exchange Act"],
                "type": "Relation",
                "expected_tool": "graph_search",
                "complexity": "medium"
            },
            {
                "question": "Who is the CEO of Apple?",
                "expected_phrases": ["No information available", "empty result", "Tim Cook"],
                "type": "Missing Data Handling",
                "complexity": "low",
                "pass_threshold": 0.5
            },
            {
                "question": "What statements mention 'revenue'?",
                "expected_phrases": ["revenue", "statement"],
                "expected_answer": "The database contains statements mentioning revenue from various companies, primarily Tesla and NVIDIA. These statements cover revenue-related topics including automotive revenue growth, FSD subscription trends, data center revenue, and quarterly revenue guidance discussions.",
                "type": "Content Search",
                "expected_tool": "graph_search",
                "complexity": "medium",
                "pass_threshold": 0.6
            },
            {
                "question": "What risks are associated with TSLA?",
                "expected_phrases": ["risk", "TSLA", "Tesla"],
                "expected_answer": "Tesla (TSLA) faces several key risks including financial and operational risks such as earnings misses, revenue decline, balance sheet pressure, and high valuation metrics. Market risks include European sales decline and competitive pressures. Regulatory risks involve legal challenges and regulatory hurdles for robotaxi services.",
                "type": "Graph Relations",
                "expected_tool": "graph_search",
                "complexity": "medium",
                "pass_threshold": 0.6
            },
            {
                "question": "Find documents about Tesla revenue discussions",
                "expected_phrases": ["Tesla", "revenue"],
                "expected_answer": "Found documents related to Tesla revenue discussions in the vector database. The documents include Tesla Financial Statistics with revenue metrics, quarterly revenue growth data, financial analysis documents, and discussions about Tesla's revenue streams and financial performance.",
                "type": "Semantic Search",
                "expected_tool": "milvus_search",
                "complexity": "medium",
                "pass_threshold": 0.6
            },
            {
                "question": "What did executives say about Tesla's financial performance?",
                "expected_phrases": ["Tesla", "financial", "executive"],
                "expected_answer": "Tesla executives commented on the company's financial performance, focusing on profitability, cash flow, and strategic decisions. Elon Musk highlighted that Tesla remains profitable despite industry challenges. CFO Vaibhav Taneja mentioned record operating cash flow and discussed revenue growth, financing incentives, and strategic accessibility programs.",
                "type": "Hybrid Query",
                "expected_tools": ["graph_search", "milvus_search"],
                "complexity": "high",
                "pass_threshold": 0.6
            }
        ]
    
    def run_benchmark(
        self, 
        iterations: int = 1,
        export_format: str = "json"
    ):
        """Run comprehensive benchmark."""
        print("\n" + "="*80)
        print("=== Enhanced GraphRAG Agent Benchmark ===")
        print("="*80 + "\n")
        
        all_results = []
        
        for iteration in range(iterations):
            if iterations > 1:
                print(f"\n--- Iteration {iteration + 1}/{iterations} ---\n")
            
            for case in self.dataset:
                result = self._run_test_case(case)
                all_results.append(result)
                self.results.append(result)
            
            # Calculate iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(all_results)
            self.metrics_history.append(iteration_metrics)
        
        # Generate comprehensive report
        self._generate_report(export_format)
        
        # Save results
        metadata = {
            'iterations': iterations,
            'total_tests': len(self.dataset) * iterations,
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(self.dataset)
        }
        self.performance_tracker.save_run(all_results, metadata)
    
    def _run_test_case(self, case: Dict) -> Dict:
        """Run a single test case with comprehensive tracking."""
        q = case["question"]
        print(f">> Running: '{q}' [{case.get('type', 'Unknown')}]")
        
        start_time = time.time()
        
        try:
            # Run agent with detailed tracking and tool usage capture
            response, tool_usage = run_agent_query(q, return_tool_usage=True)
            total_latency = time.time() - start_time
            
            # Track tool usage with captured data
            self.tool_tracker.parse_agent_response(response, total_latency, tool_usage=tool_usage)
            
            # Comprehensive evaluation
            evaluation = self.evaluator.evaluate_answer(
                question=q,
                actual_answer=response,
                expected_phrases=case.get("expected_phrases", []),
                expected_answer=case.get("expected_answer")
            )
            
            # Check tool usage expectations
            tool_analysis = self._analyze_tool_usage(
                response, 
                case.get("expected_tool"),
                case.get("expected_tools"),
                tool_usage=tool_usage
            )
            
            # Determine status based on overall score
            pass_threshold = case.get('pass_threshold', 0.7)
            status = 'PASS' if evaluation['overall_score'] >= pass_threshold else 'FAIL'
            
            result = {
                'question': q,
                'type': case.get('type', 'Unknown'),
                'complexity': case.get('complexity', 'medium'),
                'response': response,
                'latency': total_latency,
                'evaluation': evaluation,
                'tool_analysis': tool_analysis,
                'status': status,
                'timestamp': datetime.now().isoformat()
            }
            
            status_icon = "✅" if status == 'PASS' else "❌"
            print(f"   {status_icon} Score: {evaluation['overall_score']:.2f} | Latency: {total_latency:.2f}s")
            
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"   ❌ ERROR: {str(e)}")
            
            return {
                'question': q,
                'type': case.get('type', 'Unknown'),
                'status': 'ERROR',
                'error': str(e),
                'error_trace': error_trace,
                'latency': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_tool_usage(self, response: str, expected_tool: Optional[str], expected_tools: Optional[List[str]], tool_usage: Optional[Dict] = None) -> Dict:
        """Analyze if correct tools were used."""
        # Use provided tool_usage if available
        if tool_usage:
            used_graph = tool_usage.get('graph_search', 0) > 0
            used_milvus = tool_usage.get('milvus_search', 0) > 0
        else:
            # Fallback to pattern matching
            used_graph = bool(re.search(r'graph_search|graph_rag', response, re.IGNORECASE))
            used_milvus = bool(re.search(r'milvus_search', response, re.IGNORECASE))
            
            # Also check indirect indicators
            if not used_graph:
                graph_patterns = [
                    r'statements.*mention|database.*contains.*statements',
                    r'connected to|associated with',
                    r'relationships|entities',
                    r'earnings.*transcript'
                ]
                used_graph = any(re.search(p, response, re.IGNORECASE) for p in graph_patterns)
            
            if not used_milvus:
                milvus_patterns = [
                    r'vector database|found \d+ documents',
                    r'documents.*include|documents.*found',
                    r'semantic.*search.*results'
                ]
                used_milvus = any(re.search(p, response, re.IGNORECASE) for p in milvus_patterns)
        
        analysis = {
            'graph_used': used_graph,
            'milvus_used': used_milvus,
            'expected_tool_match': True
        }
        
        if expected_tool:
            if expected_tool == 'graph_search':
                analysis['expected_tool_match'] = used_graph
            elif expected_tool == 'milvus_search':
                analysis['expected_tool_match'] = used_milvus
        
        if expected_tools:
            graph_expected = 'graph_search' in expected_tools
            milvus_expected = 'milvus_search' in expected_tools
            analysis['expected_tools_match'] = (
                (graph_expected == used_graph) and (milvus_expected == used_milvus)
            )
        
        return analysis
    
    def _calculate_iteration_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for an iteration."""
        passed = sum(1 for r in results if r.get('status') == 'PASS')
        failed = sum(1 for r in results if r.get('status') == 'FAIL')
        errors = sum(1 for r in results if r.get('status') == 'ERROR')
        
        latencies = [r['latency'] for r in results if 'latency' in r]
        scores = [r['evaluation']['overall_score'] for r in results if 'evaluation' in r]
        
        metrics = {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'accuracy': passed / len(results) if results else 0
        }
        
        if latencies:
            if HAS_NUMPY:
                metrics['avg_latency'] = float(np.mean(latencies))
                metrics['p95_latency'] = float(np.percentile(latencies, 95))
            else:
                latencies_sorted = sorted(latencies)
                metrics['avg_latency'] = sum(latencies) / len(latencies)
                metrics['p95_latency'] = latencies_sorted[int(len(latencies) * 0.95)] if latencies_sorted else 0
        
        if scores:
            if HAS_NUMPY:
                metrics['avg_score'] = float(np.mean(scores))
            else:
                metrics['avg_score'] = sum(scores) / len(scores)
        
        metrics['tool_stats'] = self.tool_tracker.get_statistics()
        
        return metrics
    
    def _generate_report(self, export_format: str = "json"):
        """Generate comprehensive benchmark report."""
        print("\n" + "="*80)
        print("=== COMPREHENSIVE BENCHMARK REPORT ===")
        print("="*80)
        
        # Summary statistics
        metrics = self._calculate_iteration_metrics(self.results)
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Tests: {metrics['total']}")
        print(f"   ✅ Passed: {metrics['passed']} ({metrics['accuracy']*100:.1f}%)")
        print(f"   ❌ Failed: {metrics['failed']}")
        print(f"   ⚠️  Errors: {metrics['errors']}")
        if 'avg_latency' in metrics:
            print(f"   ⏱️  Avg Latency: {metrics['avg_latency']:.2f}s")
        if 'p95_latency' in metrics:
            print(f"   ⏱️  P95 Latency: {metrics['p95_latency']:.2f}s")
        if 'avg_score' in metrics:
            print(f"   📈 Avg Score: {metrics['avg_score']:.2f}")
        
        # Tool usage statistics
        tool_stats = metrics.get('tool_stats', {})
        if tool_stats:
            print(f"\n🔧 Tool Usage:")
            total_calls = tool_stats.get('total_calls', {})
            print(f"   Graph Search: {total_calls.get('graph_search', 0)} calls")
            print(f"   Milvus Search: {total_calls.get('milvus_search', 0)} calls")
            usage_patterns = tool_stats.get('usage_patterns', {})
            if usage_patterns:
                print(f"   Graph Only: {usage_patterns.get('graph_only', 0)}")
                print(f"   Milvus Only: {usage_patterns.get('milvus_only', 0)}")
                print(f"   Both Tools: {usage_patterns.get('both_tools', 0)}")
        
        # Detailed results table
        print(f"\n📋 Detailed Results:")
        table_data = []
        for r in self.results:
            score = r.get('evaluation', {}).get('overall_score', 0) if 'evaluation' in r else 0
            table_data.append([
                r['question'][:40] + "..." if len(r['question']) > 40 else r['question'],
                r['type'],
                r['status'],
                f"{score:.2f}" if score > 0 else "N/A",
                f"{r.get('latency', 0):.2f}s"
            ])
        print(tabulate(table_data, headers=["Question", "Type", "Status", "Score", "Latency"], tablefmt="grid"))
        
        # Performance by type
        print(f"\n📊 Performance by Query Type:")
        type_stats = self._calculate_type_statistics()
        for query_type, stats in type_stats.items():
            print(f"   {query_type}:")
            print(f"      Accuracy: {stats['accuracy']*100:.1f}%")
            print(f"      Avg Latency: {stats['avg_latency']:.2f}s")
            print(f"      Avg Score: {stats['avg_score']:.2f}")
            print(f"      Count: {stats['count']}")
        
        # Export results
        if export_format == "json":
            self._export_json()
        elif export_format == "csv":
            self._export_csv()
    
    def _calculate_type_statistics(self) -> Dict:
        """Calculate statistics grouped by query type."""
        type_results = defaultdict(list)
        for r in self.results:
            type_results[r['type']].append(r)
        
        stats = {}
        for query_type, results in type_results.items():
            passed = sum(1 for r in results if r.get('status') == 'PASS')
            latencies = [r['latency'] for r in results if 'latency' in r]
            scores = [r['evaluation']['overall_score'] for r in results if 'evaluation' in r]
            
            stats[query_type] = {
                'accuracy': passed / len(results) if results else 0,
                'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'count': len(results)
            }
        
        return stats
    
    def _export_json(self):
        """Export results to JSON."""
        filename = self.performance_tracker.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_data = {
            'summary': self._calculate_iteration_metrics(self.results),
            'results': self.results,
            'tool_stats': self.tool_tracker.get_statistics(),
            'type_statistics': self._calculate_type_statistics()
        }
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\n💾 Results exported to: {filename}")
    
    def _export_csv(self):
        """Export results to CSV."""
        import csv
        filename = self.performance_tracker.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Question', 'Type', 'Status', 'Score', 'Latency', 'Graph Used', 'Milvus Used', 'Keyword Coverage', 'LLM Grade'])
            for r in self.results:
                eval_data = r.get('evaluation', {})
                tool_analysis = r.get('tool_analysis', {})
                writer.writerow([
                    r['question'],
                    r['type'],
                    r['status'],
                    eval_data.get('overall_score', 0) if eval_data else 0,
                    r.get('latency', 0),
                    tool_analysis.get('graph_used', False),
                    tool_analysis.get('milvus_used', False),
                    eval_data.get('keyword_coverage', 0) if eval_data else 0,
                    eval_data.get('llm_grade', 0) if eval_data else 0
                ])
        print(f"\n💾 Results exported to: {filename}")


# Backward compatibility - keep original class name
class AgentBenchmarker(EnhancedAgentBenchmarker):
    """Backward compatibility alias."""
    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced benchmark for GraphRAG agent')
    parser.add_argument('--config', type=str, help='Path to benchmark configuration JSON file')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations to run')
    parser.add_argument('--export', type=str, choices=['json', 'csv', 'both'], default='json', help='Export format')
    
    args = parser.parse_args()
    
    benchmarker = EnhancedAgentBenchmarker(config_path=args.config)
    
    export_format = args.export if args.export != 'both' else 'json'
    benchmarker.run_benchmark(iterations=args.iterations, export_format=export_format)
    
    if args.export == 'both':
        benchmarker._export_csv()
