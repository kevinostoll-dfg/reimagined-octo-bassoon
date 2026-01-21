# Enhanced Benchmark Agent

## Overview

The Enhanced Benchmark Agent provides comprehensive testing and evaluation capabilities for the GraphRAG Hybrid Search system. It includes multi-metric evaluation, tool usage tracking, historical performance tracking, and detailed reporting.

## Features

### 1. **Multi-Metric Evaluation**
- **Semantic Similarity**: Uses sentence transformers to measure semantic similarity between expected and actual answers
- **ROUGE Scores**: Measures overlap of n-grams (ROUGE-1, ROUGE-2, ROUGE-L)
- **Keyword Coverage**: Percentage of expected keywords found in the answer
- **LLM-Based Grading**: Uses the LLM to evaluate answer quality on a 0-1 scale
- **Answer Length Score**: Evaluates if answer length is appropriate (50-500 chars optimal)
- **Hallucination Detection**: Detects potentially unsupported claims

### 2. **Tool Usage Tracking**
- Tracks which tools are used (graph_search, milvus_search)
- Analyzes tool usage patterns (graph only, milvus only, both)
- Measures average execution time per tool
- Validates if correct tools were used for each query type

### 3. **Historical Performance Tracking**
- Saves benchmark results to JSON files with timestamps
- Loads historical results for comparison
- Calculates trends and improvements over time
- Stores comprehensive metadata with each run

### 4. **Comprehensive Reporting**
- Overall statistics (accuracy, latency, scores)
- Performance breakdown by query type
- Tool usage statistics
- Detailed results table
- Export to JSON or CSV formats

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies (for enhanced metrics):
- `numpy`: For statistical calculations
- `sentence-transformers`: For semantic similarity
- `rouge-score`: For ROUGE metrics

The benchmark will work without these but with reduced functionality.

## Usage

### Basic Usage

```bash
# Run with default dataset
python scripts/benchmark_agent.py

# Run with custom config file
python scripts/benchmark_agent.py --config scripts/benchmark_config.json

# Run multiple iterations
python scripts/benchmark_agent.py --iterations 3

# Export to CSV
python scripts/benchmark_agent.py --export csv

# Export to both formats
python scripts/benchmark_agent.py --export both
```

### Command Line Arguments

- `--config`: Path to benchmark configuration JSON file (optional)
- `--iterations`: Number of iterations to run (default: 1)
- `--export`: Export format - 'json', 'csv', or 'both' (default: 'json')

### Programmatic Usage

```python
from scripts.benchmark_agent import EnhancedAgentBenchmarker

# Initialize benchmarker
benchmarker = EnhancedAgentBenchmarker(config_path="scripts/benchmark_config.json")

# Run benchmark
benchmarker.run_benchmark(iterations=3, export_format="json")

# Access results
results = benchmarker.results
metrics = benchmarker.metrics_history

# Load historical results
history = benchmarker.performance_tracker.load_history(days=7)
```

## Configuration File Format

Create a JSON configuration file to customize test cases:

```json
{
  "test_cases": [
    {
      "question": "Your test question",
      "expected_phrases": ["keyword1", "keyword2"],
      "expected_answer": "Expected answer text (optional)",
      "type": "Query Type",
      "expected_tool": "graph_search",
      "complexity": "low|medium|high",
      "pass_threshold": 0.7,
      "weight": 1.0
    }
  ],
  "evaluation": {
    "semantic_similarity_weight": 0.3,
    "rouge_weight": 0.2,
    "keyword_coverage_weight": 0.2,
    "llm_grade_weight": 0.2,
    "answer_length_weight": 0.05,
    "hallucination_weight": 0.05,
    "default_pass_threshold": 0.7
  },
  "performance": {
    "max_latency_seconds": 30.0,
    "target_accuracy": 0.8,
    "warn_on_slow_queries": true
  }
}
```

### Test Case Fields

- `question`: The query to test
- `expected_phrases`: List of keywords/phrases that should appear in the answer
- `expected_answer`: (Optional) Full expected answer for semantic similarity
- `type`: Category of the query (e.g., "Relation", "Content Search")
- `expected_tool`: Which tool should be used ("graph_search" or "milvus_search")
- `expected_tools`: (Optional) List of tools that should be used
- `complexity`: Query complexity level ("low", "medium", "high")
- `pass_threshold`: Minimum score to pass (0.0-1.0)
- `weight`: Weight for this test case in overall scoring

## Output

### Console Output

The benchmark prints:
- Overall statistics (accuracy, latency, scores)
- Tool usage statistics
- Detailed results table
- Performance breakdown by query type
- Export file locations

### Exported Files

Results are saved to `benchmark_results/` directory:

- `benchmark_YYYYMMDD_HHMMSS.json`: Individual run results
- `benchmark_report_YYYYMMDD_HHMMSS.json`: Comprehensive report (JSON)
- `benchmark_report_YYYYMMDD_HHMMSS.csv`: Comprehensive report (CSV)

### JSON Export Format

```json
{
  "summary": {
    "total": 6,
    "passed": 5,
    "failed": 1,
    "errors": 0,
    "accuracy": 0.833,
    "avg_latency": 12.5,
    "p95_latency": 18.2,
    "avg_score": 0.78
  },
  "results": [
    {
      "question": "...",
      "type": "...",
      "status": "PASS",
      "latency": 10.5,
      "evaluation": {
        "overall_score": 0.85,
        "semantic_similarity": 0.9,
        "rouge1": 0.8,
        "keyword_coverage": 1.0,
        "llm_grade": 0.9
      },
      "tool_analysis": {
        "graph_used": true,
        "milvus_used": false
      }
    }
  ],
  "tool_stats": {
    "total_calls": {
      "graph_search": 5,
      "milvus_search": 1
    },
    "usage_patterns": {
      "graph_only": 4,
      "milvus_only": 1,
      "both_tools": 1
    }
  }
}
```

## Metrics Explained

### Overall Score

Weighted combination of:
- Semantic Similarity (30%): How semantically similar the answer is to expected
- ROUGE-1 (20%): Unigram overlap
- Keyword Coverage (20%): Percentage of expected keywords found
- LLM Grade (20%): LLM-based quality assessment
- Answer Length (5%): Appropriate answer length
- Hallucination Score (5%): Detection of unsupported claims

### Status Determination

- **PASS**: Overall score >= pass_threshold (default 0.7)
- **FAIL**: Overall score < pass_threshold
- **ERROR**: Exception occurred during execution

## Best Practices

1. **Start with Default Dataset**: Run with default dataset first to establish baseline
2. **Add Domain-Specific Tests**: Add test cases specific to your use case
3. **Run Multiple Iterations**: Run 3-5 iterations to account for variability
4. **Track Over Time**: Compare results across runs to identify regressions
5. **Set Appropriate Thresholds**: Adjust pass_threshold based on your quality requirements
6. **Monitor Tool Usage**: Ensure correct tools are being used for each query type

## Troubleshooting

### Missing Dependencies

If you see warnings about missing dependencies:
```bash
pip install numpy sentence-transformers rouge-score
```

### Low Scores

- Check if expected_phrases match actual data
- Verify expected_tool matches query type
- Review actual responses to understand failures
- Adjust pass_threshold if needed

### Slow Execution

- Reduce number of iterations
- Use smaller dataset for quick tests
- Check network latency to APIs
- Review tool execution times in statistics

## Examples

### Example 1: Quick Test

```bash
python scripts/benchmark_agent.py
```

### Example 2: Comprehensive Test Suite

```bash
python scripts/benchmark_agent.py \
  --config scripts/benchmark_config.json \
  --iterations 5 \
  --export both
```

### Example 3: Custom Test Case

Add to `benchmark_config.json`:
```json
{
  "question": "What are the main risks for technology companies?",
  "expected_phrases": ["risk", "technology", "company"],
  "type": "Domain-Specific Query",
  "expected_tool": "graph_search",
  "complexity": "high"
}
```

## Integration with CI/CD

Add to your CI pipeline:

```yaml
- name: Run Benchmark
  run: |
    python scripts/benchmark_agent.py --iterations 1 --export json
    # Check if accuracy meets threshold
    python -c "
    import json
    with open('benchmark_results/benchmark_report_*.json') as f:
        data = json.load(f)
        assert data['summary']['accuracy'] >= 0.8, 'Accuracy below threshold'
    "
```

## Future Enhancements

Potential improvements:
- A/B testing between different agent configurations
- Automatic test case generation
- Integration with observability tools (Arize Phoenix, etc.)
- Performance regression detection
- Automated alerting on failures

