
from llama_index.core.tools import FunctionTool
from llama_index.core import PromptTemplate

# Import local modules
import os
import sys
import logging
import re

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config

# Add scripts directory to path for fetch_schema import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from fetch_schema import get_schema_from_dragonfly, format_schema_for_prompt

logger = logging.getLogger(__name__)

def graph_rag_func(query: str) -> str:
    """
    Queries the Memgraph graph database to answer questions about relationships, 
    entities, and structured data (e.g., earnings transcripts, people, organizations).
    """
    try:
        # Import inside function to avoid early config loading issues and ensure env vars are set
        from agent.completions_router import CompletionsRouter
        # Import new PropertyGraph components
        from llama_index.core import PropertyGraphIndex, Settings
        from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
        from llama_index.core.indices.property_graph import TextToCypherRetriever
        from llama_index.core import PromptTemplate

        # 0. Explicitly disable embeddings to prevent OpenAI fallback
        Settings.embed_model = None
        
        # 0.5. Ensure models are loaded before initializing LLM
        from agent.config import ensure_models_loaded
        ensure_models_loaded()
        
        # 1. Initialize LLM (Router)
        llm = CompletionsRouter()
        
        # 2. Connect to Memgraph using the PropertyGraphStore interface
        graph_store = MemgraphPropertyGraphStore(
            username=config.MEMGRAPH_USER,
            password=config.MEMGRAPH_PASSWORD,
            url=config.MEMGRAPH_URI
        )
        
        # 3. Create Index from Existing Store
        # Using text-to-cypher only (no embeddings)
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=llm,
            embed_model=None  # Explicitly disable embeddings
        )

        # 3. Fetch Schema from Dragonfly for the Prompt
        raw_schema = get_schema_from_dragonfly(key="memgraph:cipher:full")
        if raw_schema:
            schema_info = format_schema_for_prompt(raw_schema)
        else:
            schema_info = """
            Nodes: CONCEPT, PERSON, STATEMENT, ORG, TRANSCRIPT
            Relationships: SVO_TRIPLE, SAID, CO_MENTIONED, HAS_SPEAKER
            """

        # 4. Create TextToCypherRetriever with Custom Prompt
        # Enhanced prompt template with examples for complex relationships
        
        cypher_generation_template_str = (
            "You are an expert Cypher query generator for Memgraph graph database.\n"
            f"{schema_info}\n\n"
            
            "CRITICAL RULES:\n"
            "1. Use explicit MATCH clauses for all relationships - DO NOT use pattern matching in WHERE clauses.\n"
            "2. Use 'canonical_name' property for entity names (not 'name').\n"
            "3. For text search, use 'CONTAINS' operator: WHERE s.text CONTAINS 'Tesla'\n"
            "4. Use OPTIONAL MATCH when relationships might not exist.\n"
            "5. Use DISTINCT to avoid duplicate results.\n"
            "6. Use COLLECT() to aggregate related entities (e.g., multiple speakers per statement).\n"
            "7. ALWAYS include LIMIT clause when returning potentially large result sets:\n"
            "   - For statements, people lists, entities: Use LIMIT 50 (or LIMIT 10 for top N queries)\n"
            "   - For counts/aggregations: LIMIT is optional but recommended\n"
            "   - NEVER omit LIMIT when returning STATEMENT, PERSON, ORG, or other entity lists\n"
            "   - Example: RETURN ... LIMIT 50\n\n"
            
            "COMPLEX RELATIONSHIP PATTERNS:\n\n"
            
            "Pattern 1: Multi-hop relationships (2+ hops)\n"
            "Example: Find statements about a company through shared entities\n"
            "MATCH (org:ORG {canonical_name: 'Tesla'})-[r1:CO_MENTIONED]-(shared)\n"
            "MATCH (s:STATEMENT)-[r2:CO_MENTIONED]-(shared)\n"
            "WHERE shared <> org\n"
            "RETURN DISTINCT s.statement_id, s.text\n"
            "LIMIT 50\n\n"
            
            "Pattern 2: Combining text search with graph traversal\n"
            "Example: Find people who made statements mentioning a company\n"
            "MATCH (s:STATEMENT)\n"
            "WHERE s.text CONTAINS 'Tesla' OR s.text CONTAINS 'TSLA'\n"
            "MATCH (p:PERSON)-[:SAID]->(s)\n"
            "RETURN p.canonical_name, COUNT(DISTINCT s.statement_id) as count\n\n"
            
            "Pattern 3: Indirect relationships (entities don't directly connect)\n"
            "Example: Find statements related to Tesla via shared concepts/dates\n"
            "MATCH (tesla:ORG {canonical_name: 'Tesla'})-[r1:CO_MENTIONED|TEMPORAL_CONTEXT]-(shared)\n"
            "MATCH (s:STATEMENT)-[r2:CO_MENTIONED|TEMPORAL_CONTEXT|SAID]-(shared)\n"
            "WHERE shared <> tesla AND (s.text CONTAINS 'Tesla' OR shared.canonical_name CONTAINS 'Tesla')\n"
            "OPTIONAL MATCH (person:PERSON)-[:SAID]->(s)\n"
            "WITH DISTINCT s, COLLECT(DISTINCT person.canonical_name) as speakers\n"
            "RETURN s.statement_id, s.text, speakers[0] as speaker\n"
            "LIMIT 50\n\n"
            
            "Pattern 4: Aggregating multiple relationships\n"
            "Example: Get all speakers for statements with their roles\n"
            "MATCH (s:STATEMENT)\n"
            "WHERE s.text CONTAINS 'Tesla'\n"
            "OPTIONAL MATCH (p:PERSON)-[:SAID]->(s)\n"
            "OPTIONAL MATCH (p)-[:HAS_ROLE]->(r:ROLE)\n"
            "WITH s, COLLECT(DISTINCT p.canonical_name) as speakers, COLLECT(DISTINCT r.title) as roles\n"
            "RETURN s.statement_id, speakers, roles\n"
            "LIMIT 50\n\n"
            
            "Pattern 5: Relationship direction matters\n"
            "SAID goes FROM PERSON TO STATEMENT: (p:PERSON)-[:SAID]->(s:STATEMENT)\n"
            "CO_MENTIONED is bidirectional: (entity1)-[:CO_MENTIONED]-(entity2)\n"
            "SVO_TRIPLE can go both ways: (subject)-[:SVO_TRIPLE]->(object) or <-[:SVO_TRIPLE]-\n\n"
            
            "COMMON QUERY TYPES:\n\n"
            
            "1. Finding statements about a company:\n"
            "   - First try: Text search in STATEMENT.text (ALWAYS add LIMIT 50)\n"
            "   - Second try: Multi-hop via CO_MENTIONED relationships (ALWAYS add LIMIT 50)\n"
            "   - Third try: Via SVO_TRIPLE relationships (ALWAYS add LIMIT 50)\n\n"
            
            "2. Finding people associated with a company:\n"
            "   - Find statements mentioning company (text search)\n"
            "   - Then find people who SAID those statements\n"
            "   - Or find people connected via shared entities\n\n"
            
            "3. Finding metrics/quantities:\n"
            "   - Use QUANTITY_OF or SVO_TRIPLE relationships\n"
            "   - Filter by METRIC, MONEY, PERCENT, CARDINAL node types\n\n"
            
            "4. Finding risks:\n"
            "   - Use EXPOSES_TO_RISK or MITIGATES_RISK relationships\n"
            "   - Filter by RISK node type\n\n"
            
            "MEMGRAPH-SPECIFIC NOTES:\n"
            "- Memgraph does NOT support GROUP BY - use WITH clauses for aggregation\n"
            "- Use COUNT(DISTINCT node) instead of COUNT(*)\n"
            "- Relationship types can be chained: -[:CO_MENTIONED|SAID]-\n"
            "- Use labels(node)[0] to get the first label of a node\n"
            "- ALWAYS include LIMIT 50 (or appropriate number) at the end of RETURN clauses for entity lists\n"
            "- Missing LIMIT can cause performance issues and return too much data\n\n"
            
            "CRITICAL SYNTAX RULES:\n"
            "- NEVER use MATCH after OPTIONAL MATCH - if you need optional matching, use OPTIONAL MATCH for both\n"
            "- Always define variables before using them in WHERE/RETURN clauses\n"
            "- Use WITH clauses to pass variables between query parts\n"
            "- Ensure all variables referenced in RETURN exist in previous MATCH clauses\n"
            "- Example of CORRECT pattern: OPTIONAL MATCH (a)-[:REL]->(b) OPTIONAL MATCH (b)-[:REL2]->(c)\n"
            "- Example of WRONG pattern: OPTIONAL MATCH (a)-[:REL]->(b) MATCH (b)-[:REL2]->(c) <- This will fail!\n\n"
            
            "Question: {question}\n"
            "CRITICAL: Return ONLY the Cypher query. Do NOT include any explanations, comments, or text before or after the query.\n"
            "Return ONLY valid Cypher starting with MATCH, CREATE, RETURN, or other Cypher keywords.\n"
            "Generate a Cypher query that handles complex relationships appropriately:"
        )
        
        # Enhanced validator with automatic fixes
        def clean_cypher(cypher_query: str) -> str:
            """Remove markdown code fences and extract Cypher query from mixed text."""
            # Strip markdown code blocks (```cypher ... ``` or ``` ... ```)
            cypher_query = cypher_query.strip()
            if cypher_query.startswith("```"):
                # Remove opening fence
                lines = cypher_query.split("\n")
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]  # Remove first line (```cypher or ```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove last line (```)
                cypher_query = "\n".join(lines)
            
            # Extract Cypher query if there's explanatory text
            # Look for Cypher keywords to find where the actual query starts
            cypher_keywords = [
                r'^\s*MATCH\b',
                r'^\s*CREATE\b',
                r'^\s*RETURN\b',
                r'^\s*MERGE\b',
                r'^\s*WITH\b',
                r'^\s*UNWIND\b',
                r'^\s*CALL\b',
                r'^\s*OPTIONAL\s+MATCH\b'
            ]
            
            lines = cypher_query.split('\n')
            cypher_start_idx = -1
            
            # Find the first line that starts with a Cypher keyword
            for i, line in enumerate(lines):
                for pattern in cypher_keywords:
                    if re.match(pattern, line, re.IGNORECASE):
                        cypher_start_idx = i
                        break
                if cypher_start_idx >= 0:
                    break
            
            # If we found where Cypher starts, extract from there
            if cypher_start_idx > 0:
                logger.info(f"Extracting Cypher query from line {cypher_start_idx + 1} (removed {cypher_start_idx} lines of explanatory text)")
                cypher_query = '\n'.join(lines[cypher_start_idx:])
            
            # Also remove any trailing explanatory text
            # Look for lines that don't look like Cypher (contain common English words)
            cypher_lines = []
            for line in cypher_query.split('\n'):
                line_stripped = line.strip()
                # Skip empty lines
                if not line_stripped:
                    continue
                # Skip lines that look like explanations (contain common English words but no Cypher keywords)
                has_cypher_keyword = any(re.search(keyword.replace('^\\s*', '').replace('\\b', ''), line, re.IGNORECASE) for keyword in cypher_keywords)
                # Check if it's a comment (starts with // or --)
                if line_stripped.startswith('//') or line_stripped.startswith('--'):
                    cypher_lines.append(line)
                    continue
                # Check if it contains Cypher-like patterns (MATCH, RETURN, etc.)
                if re.search(r'\b(MATCH|RETURN|WITH|CREATE|MERGE|WHERE|LIMIT|ORDER|BY)\b', line, re.IGNORECASE):
                    cypher_lines.append(line)
                # If line contains Cypher syntax characters, keep it
                elif re.search(r'[:\(\)\[\]\{\}\-\>]', line):  # Contains Cypher syntax characters
                    cypher_lines.append(line)
                elif has_cypher_keyword:
                    cypher_lines.append(line)
                else:
                    # Might be explanatory text, skip it
                    logger.debug(f"Skipping line that looks like explanation: {line[:50]}...")
            
            return '\n'.join(cypher_lines).strip()
        
        def validate_and_fix_cypher(cypher_query: str) -> str:
            """Validate and attempt to fix common Cypher errors."""
            # First clean markdown
            cypher_query = clean_cypher(cypher_query)
            
            # Common fixes for Memgraph compatibility
            fixes = [
                # Fix GROUP BY -> WITH (Memgraph doesn't support GROUP BY)
                (r'\bGROUP BY\b', 'WITH'),
                # Ensure canonical_name is used instead of name (when not part of another word)
                (r'\.name\b(?!\w)', '.canonical_name'),
            ]
            
            for pattern, replacement in fixes:
                cypher_query = re.sub(pattern, replacement, cypher_query, flags=re.IGNORECASE)
            
            # Fix: MATCH can't be put after OPTIONAL MATCH
            # Convert "OPTIONAL MATCH ... MATCH" to "OPTIONAL MATCH ... OPTIONAL MATCH"
            lines = cypher_query.split('\n')
            fixed_lines = []
            prev_was_optional_match = False
            
            for line in lines:
                line_stripped = line.strip()
                # Check if this line starts with MATCH (not OPTIONAL MATCH)
                if re.match(r'^\s*MATCH\b', line_stripped, re.IGNORECASE) and prev_was_optional_match:
                    # Convert MATCH to OPTIONAL MATCH if previous line was OPTIONAL MATCH
                    line = re.sub(r'^\s*MATCH\b', 'OPTIONAL MATCH', line, flags=re.IGNORECASE)
                    logger.warning("Fixed: Converted MATCH to OPTIONAL MATCH after OPTIONAL MATCH clause")
                
                # Track if this line is OPTIONAL MATCH
                prev_was_optional_match = bool(re.match(r'^\s*OPTIONAL\s+MATCH\b', line_stripped, re.IGNORECASE))
                
                # Reset if we hit a WITH clause (new context)
                if re.match(r'^\s*WITH\b', line_stripped, re.IGNORECASE):
                    prev_was_optional_match = False
                
                fixed_lines.append(line)
            
            cypher_query = '\n'.join(fixed_lines)
            
            # Check for unbound variables (simple check - variables used but not defined)
            # Extract all variable names from MATCH/OPTIONAL MATCH clauses
            defined_vars = set()
            for line in cypher_query.split('\n'):
                # Match patterns like (var:LABEL) or (var)-[:REL]->(var2)
                matches = re.findall(r'\((\w+):', line)
                defined_vars.update(matches)
                # Also match standalone variables in patterns
                matches = re.findall(r'\((\w+)\)', line)
                defined_vars.update(matches)
                # Extract variables from WITH clauses (they're defined there)
                if re.search(r'\bWITH\b', line, re.IGNORECASE):
                    # Extract variable names after WITH and before AS (aliases)
                    with_expr = re.search(r'\bWITH\s+(.+?)(?:\s+RETURN|\s+MATCH|$)', line, re.IGNORECASE)
                    if with_expr:
                        expr = with_expr.group(1)
                        # Extract variable names (before AS or comma)
                        var_names = re.findall(r'\b([a-z][a-z0-9_]*)\s*(?:AS|,|$)', expr, re.IGNORECASE)
                        defined_vars.update(var_names)
                        # Also extract aliases after AS
                        aliases = re.findall(r'\s+AS\s+([a-z][a-z0-9_]*)', expr, re.IGNORECASE)
                        defined_vars.update(aliases)
            
            # Check for undefined variables in WHERE/RETURN clauses
            used_vars = set()
            cypher_keywords = {
                'where', 'return', 'with', 'match', 'optional', 'and', 'or', 'not', 'is', 'null', 
                'distinct', 'limit', 'order', 'by', 'as', 'count', 'collect', 'sum', 'avg', 'max', 
                'min', 'contains', 'starts', 'ends', 'in', 'exists', 'true', 'false'
            }
            property_names = {'text', 'canonical_name', 'statement_id', 'title', 'mention_count', 'entity_id', 'speakers', 'roles'}
            
            for line in cypher_query.split('\n'):
                # Find variable references (word followed by dot)
                matches = re.findall(r'\b(\w+)\.', line)
                used_vars.update(matches)
                # Also check standalone variable references in WHERE/RETURN/WITH
                if re.search(r'\b(WHERE|RETURN|WITH)\b', line, re.IGNORECASE):
                    matches = re.findall(r'\b([a-z][a-z0-9_]*)\b', line, re.IGNORECASE)
                    # Filter out keywords, property names, and string literals
                    used_vars.update([
                        m for m in matches 
                        if m.lower() not in cypher_keywords 
                        and m.lower() not in property_names
                        and not (m.startswith("'") or m.startswith('"'))  # Not a string literal
                    ])
            
            undefined_vars = used_vars - defined_vars
            # Filter out common false positives (property names, string literals, uppercase constants)
            undefined_vars = {
                v for v in undefined_vars 
                if v.lower() not in property_names 
                and not v.isupper()  # Filter out uppercase constants like 'TSLA', 'CONTAINS'
                and len(v) > 1  # Filter out single character false positives
            }
            
            if undefined_vars:
                logger.warning(f"Potential unbound variables detected: {undefined_vars}. Query may fail.")
            
            # Log if COUNT(*) is found (but don't auto-fix as it requires context)
            if re.search(r'COUNT\s*\(\s*\*\s*\)', cypher_query, re.IGNORECASE):
                logger.warning("Query contains COUNT(*) - consider using COUNT(DISTINCT node) for Memgraph compatibility")
            
            # Check if LIMIT is missing for queries returning entity lists
            # Look for RETURN clauses that return entity properties (statement_id, canonical_name, etc.)
            # but don't have LIMIT
            has_limit = re.search(r'\bLIMIT\s+\d+', cypher_query, re.IGNORECASE)
            has_return = re.search(r'\bRETURN\b', cypher_query, re.IGNORECASE)
            
            # Entity indicators that suggest we're returning lists (not just counts)
            entity_indicators = [
                r'statement_id', r'\.text\b', r'canonical_name', r'\.speaker\b',
                r's\.statement_id', r's\.text', r'p\.canonical_name', r'org\.canonical_name',
                r'person\b', r'statement\b', r'entity\b'
            ]
            
            # Check if query returns entity data (not just aggregated counts)
            returns_entities = any(re.search(indicator, cypher_query, re.IGNORECASE) for indicator in entity_indicators)
            # Don't add LIMIT if query only returns counts/aggregations
            is_aggregation_only = re.search(r'RETURN\s+.*COUNT\s*\(', cypher_query, re.IGNORECASE) and not returns_entities
            
            # If query returns entities but has no LIMIT, add LIMIT 50
            if has_return and returns_entities and not has_limit and not is_aggregation_only:
                # Find the end of the query (last non-empty line)
                lines = cypher_query.split('\n')
                last_non_empty_idx = -1
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip():
                        last_non_empty_idx = i
                        break
                
                if last_non_empty_idx >= 0:
                    # Check if last line already has LIMIT
                    if not re.search(r'\bLIMIT\b', lines[last_non_empty_idx], re.IGNORECASE):
                        # Add LIMIT 50 as a new line
                        lines.append('LIMIT 50')
                        cypher_query = '\n'.join(lines)
                        logger.info("Automatically added LIMIT 50 to query returning entity lists")
            
            return cypher_query.strip()
        
        text_to_cypher_retriever = TextToCypherRetriever(
            index.property_graph_store,
            llm=llm,
            text_to_cypher_template=PromptTemplate(cypher_generation_template_str),
            cypher_validator=validate_and_fix_cypher,  # Validate and fix common errors
        )

        # 5. Create Query Engine - Text-to-Cypher only (no embeddings)
        # Use COMPACT mode (default) with node limits to avoid context window errors
        # COMPACT mode compacts chunks first, then refines - similar to COMPACT_AND_REFINE behavior
        from llama_index.core.response_synthesizers import ResponseMode
        
        query_engine = index.as_query_engine(
            sub_retrievers=[text_to_cypher_retriever],
            llm=llm,
            embed_model=None,  # Explicitly disable embeddings
            response_mode=ResponseMode.COMPACT,  # COMPACT is the default and provides compact-and-refine behavior
            similarity_top_k=10,  # Limit retrieved nodes to reduce context pressure
            verbose=True
        )
        
        # 6. Execute Query with Self-Healing (LlamaIndex self-healing pattern)
        # This implements execution-time error handling with LLM-based query fixes
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # First attempt: execute query normally
                    response = query_engine.query(query)
                    return str(response)
                else:
                    # Retry: enhance the prompt with error context and regenerate Cypher
                    logger.info(f"Self-healing: Attempting to fix Cypher query (attempt {attempt}/{max_retries})")
                    
                    # Create enhanced prompt template with error context
                    enhanced_template_str = (
                        cypher_generation_template_str + 
                        f"\n\nPREVIOUS ATTEMPT ERROR:\n"
                        f"The following error occurred when executing a previous version of this query:\n"
                        f"Error: {last_error}\n"
                        f"Please ensure your Cypher query avoids this error.\n"
                        f"Review the MEMGRAPH-SPECIFIC NOTES and CRITICAL SYNTAX RULES above carefully.\n"
                    )
                    
                    # Create new retriever with enhanced prompt
                    enhanced_retriever = TextToCypherRetriever(
                        index.property_graph_store,
                        llm=llm,
                        text_to_cypher_template=PromptTemplate(enhanced_template_str),
                        cypher_validator=validate_and_fix_cypher,
                    )
                    
                    # Create new query engine with enhanced retriever
                    enhanced_query_engine = index.as_query_engine(
                        sub_retrievers=[enhanced_retriever],
                        llm=llm,
                        embed_model=None,
                        response_mode=ResponseMode.COMPACT,
                        similarity_top_k=10,
                        verbose=True
                    )
                    
                    # Retry with enhanced query engine
                    response = enhanced_query_engine.query(query)
                    logger.info(f"Self-healing succeeded on attempt {attempt + 1}")
                    return str(response)
                    
            except Exception as e:
                last_error = str(e)
                error_msg = str(e)
                logger.warning(f"Query execution failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                # Check if error is Cypher-related (Memgraph errors typically contain "cypher" or "syntax")
                is_cypher_error = any(keyword in error_msg.lower() for keyword in [
                    'cypher', 'syntax', 'unbound', 'variable', 'match', 'optional', 
                    'group by', 'memgraph', 'query', 'parse', 'clienterror'
                ])
                
                if attempt < max_retries and is_cypher_error:
                    # Continue to retry with enhanced prompt
                    continue
                else:
                    # Max retries reached or non-Cypher error
                    if attempt >= max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Last error: {error_msg}")
                    raise

    except ImportError as e:
        return f"Dependency Error: {str(e)}. Please ensure llama-index-graph-stores-memgraph is updated."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error executing Graph RAG query: {repr(e)}"

# Create the Tool object
graph_rag_tool = FunctionTool.from_defaults(
    fn=graph_rag_func,
    name="graph_search",
    description="""ESSENTIAL: This tool searches the Memgraph graph database containing:
    - Companies/organizations (e.g., TSLA, Apple, etc.) and their information
    - Earnings transcripts and financial statements
    - What people said about specific topics, companies, or concepts
    - Relationships between people, organizations, and concepts
    - Statements, quotes, and mentions from transcripts
    - Financial data, earnings information, and business updates
    
    Use graph_search for:
    - Company queries (e.g., "What's going on with TSLA?", "Tell me about Apple")
    - Earnings or financial information
    - What executives or people said
    - Company relationships or connections
    - Any question that might be answered by structured data
    
    This tool searches structured data stored in the Memgraph database. Use it along with other tools for comprehensive analysis."""
)

if __name__ == "__main__":
    # Ensure API key is set from environment variables
    if not os.getenv("COMPLETIONS_ROUTER_API_KEY"):
        print("ERROR: COMPLETIONS_ROUTER_API_KEY environment variable is not set.")
        print("Please set it in .env.local or environment variables.")
        sys.exit(1)
    
    print("\n=== Testing Graph RAG Tool (PropertyGraphIndex) ===\n")
    
    test_query = "What statements mention 'afternoon'?"
    print(f"Executing Query: '{test_query}'...")
    
    result = graph_rag_func(test_query)
    
    print("\n--- Result ---")
    print(result)
    print("\n=== Test Complete ===")
