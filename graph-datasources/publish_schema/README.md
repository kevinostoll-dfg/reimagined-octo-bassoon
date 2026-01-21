# 📊 Memgraph Schema Publisher

> **Automated Memgraph CIPHER schema introspection and storage service**

This service queries Memgraph databases to introspect graph schemas, tags schema elements by service, and stores both complete and service-specific schemas in DragonFly for easy retrieval and reporting.

## 🎯 Overview

The Memgraph Schema Publisher is a Python service that:

1. **Queries Memgraph** to introspect the actual graph schema (node labels, relationship types, properties, indexes)
2. **Tags schema elements** with which service(s) created them (earnings announcements, FOMC transcripts, SEC filings, etc.)
3. **Stores schemas** in DragonFly (Redis-compatible) for:
   - Complete cross-service schema view
   - Service-specific filtered schemas for focused reporting
   - Easy programmatic access via Redis keys

## 🚀 Features

### ✨ Core Capabilities

- **🔍 Full Schema Introspection** - Automatically discovers all node labels, relationship types, properties, and indexes from Memgraph
- **🏷️ Service Tagging** - Each schema element is tagged with which service(s) it belongs to
- **📦 Multi-Schema Storage** - Stores both full and filtered schemas in DragonFly
- **🔄 Automatic Fallbacks** - Uses fallback queries when Memgraph procedures aren't available
- **🔐 Authentication Support** - Supports Memgraph username/password authentication
- **📊 Property Type Inference** - Automatically infers property types from sample data

### 📋 Stored Schemas

The service stores the following schemas in DragonFly:

| Key | Description | Use Case |
|-----|-------------|----------|
| `memgraph:cipher:full` | Complete schema from all services | Cross-service analysis, complete overview |
| `memgraph:cipher:ea` | Earnings Announcement Transcripts only | Focused reporting on earnings calls |
| `memgraph:cipher:fomc` | FOMC Transcripts only | Federal Reserve policy analysis |
| `memgraph:cipher:sec-10k` | SEC 10-K Filings only | Annual report analysis |
| `memgraph:cipher:sec-f4` | SEC Form 4 Filings only | Insider trading analysis |

## 🏗️ Architecture

### Service Definitions

The service recognizes the following graph datasource services:

#### 📈 Earnings Announcement Transcripts (`ea`)
- **Node Labels**: PERSON, ORG, PRODUCT, CONCEPT, METRIC, METRIC_DEFINITION, STATEMENT, ROLE, TECHNOLOGY, DATE, TIME, MONEY, PERCENT, CARDINAL, QUANTITY
- **Relationships**: SAID, HAS_ROLE, SVO_TRIPLE, TEMPORAL_CONTEXT, QUANTITY_OF, QUANTITY_IN_ACTION, CO_MENTIONED, SAME_AS, CAUSES, DRIVES, BOOSTS, HURTS, MITIGATES, RESULTS_IN, LEADS_TO, TARGETS, OUTPERFORMED, POSITIVE_ABOUT, NEGATIVE_ABOUT, EVENT_INVOLVES

#### 🏛️ FOMC Transcripts (`fomc`)
- **Node Labels**: PERSON, ORG, PRODUCT, CONCEPT, METRIC, METRIC_DEFINITION, SECTION, TECHNOLOGY, DATE, TIME, MONEY, PERCENT, CARDINAL, QUANTITY
- **Relationships**: SVO_TRIPLE, TEMPORAL_CONTEXT, QUANTITY_OF, QUANTITY_IN_ACTION, CO_MENTIONED, SAME_AS, CAUSES, DRIVES, BOOSTS, HURTS, MITIGATES, RESULTS_IN, LEADS_TO, TARGETS, OUTPERFORMED, POSITIVE_ABOUT, NEGATIVE_ABOUT, EVENT_INVOLVES

#### 📄 SEC 10-K Filings (`sec-10k`)
- **Node Labels**: PERSON, ORG, PRODUCT, CONCEPT, METRIC, METRIC_DEFINITION, SECTION, TECHNOLOGY, DATE, TIME, MONEY, PERCENT, CARDINAL, QUANTITY
- **Relationships**: SVO_TRIPLE, TEMPORAL_CONTEXT, QUANTITY_OF, QUANTITY_IN_ACTION, CO_MENTIONED, SAME_AS, CAUSES, DRIVES, BOOSTS, HURTS, MITIGATES, RESULTS_IN, LEADS_TO, TARGETS, OUTPERFORMED, POSITIVE_ABOUT, NEGATIVE_ABOUT, EVENT_INVOLVES

#### 💼 SEC Form 4 Filings (`sec-f4`)
- **Node Labels**: Insider, Company, Transaction, Security
- **Relationships**: FILED, INVOLVES, TRADES, HOLDS_POSITION

## 📦 Installation

### Prerequisites

- Python 3.8+
- Access to Memgraph database
- Access to DragonFly (Redis-compatible) database

### Setup

```bash
# Clone or navigate to the service directory
cd graph-datasources/publish_schema

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `GQLAlchemy>=1.8.0` - Memgraph database client
- `redis>=5.0.0` - DragonFly/Redis client
- `python-dotenv>=1.0.0` - Environment variable management

## ⚙️ Configuration

Create a `.env` file in the service directory:

```bash
# Memgraph Configuration
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687
MEMGRAPH_USER=memgraphdb          # Optional
MEMGRAPH_PASSWORD=memgraphdb      # Optional

# DragonFly Configuration
DRAGONFLY_HOST=dragonfly.blacksmith.deerfieldgreen.com
DRAGONFLY_PORT=6379
DRAGONFLY_PASSWORD=your_password  # Optional
DRAGONFLY_DB=0                    # Optional, default 0
DRAGONFLY_SSL=false               # Optional, default false
```

## 🚀 Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the schema publisher
python3 store_schema_dragonfly.py
```

### Output

The script will:

1. ✅ Connect to Memgraph and authenticate
2. 🔍 Query the database for schema information
3. 🏷️ Tag each element with service information
4. 💾 Store schemas in DragonFly
5. ✅ Verify storage was successful

### Example Output

```
================================================================================
MEMGRAPH CIPHER SCHEMA STORAGE (FROM MEMGRAPH INTROSPECTION)
================================================================================

2025-12-19 15:28:15,452 - INFO - Connecting to Memgraph at localhost:7687...
2025-12-19 15:28:15,452 - INFO -    Using authentication (user: memgraphdb)
2025-12-19 15:28:15,487 - INFO - ✅ Successfully connected to Memgraph
2025-12-19 15:28:15,487 - INFO - Building full schema from Memgraph...
2025-12-19 15:28:15,487 - INFO - Querying Memgraph for schema information...
2025-12-19 15:28:15,538 - INFO -    Found 34 node labels
2025-12-19 15:28:15,538 - INFO -    Found 13 relationship types
2025-12-19 15:28:15,538 - INFO -    Found 7 property keys
2025-12-19 15:28:15,538 - INFO -    Found 52 indexes
2025-12-19 15:28:15,717 - INFO - ✅ Built full schema
2025-12-19 15:28:17,520 - INFO - ✅ Stored schema in DragonFly at key: memgraph:cipher:full
2025-12-19 15:28:18,577 - INFO - ✅ Stored schema in DragonFly at key: memgraph:cipher:ea
...

================================================================================
✅ SUCCESS
================================================================================
Stored 5 schema(s) in DragonFly:
  - memgraph:cipher:full
  - memgraph:cipher:ea
  - memgraph:cipher:fomc
  - memgraph:cipher:sec-10k
  - memgraph:cipher:sec-f4
```

## 📖 Schema Structure

### Full Schema Format

```json
{
  "version": "1.0",
  "generated_from": "memgraph_introspection",
  "description": "Complete Memgraph CIPHER schema from all services",
  "generated_at": "2025-12-19T15:28:15.717000+00:00",
  "node_types": [
    {
      "label": "PERSON",
      "services": ["ea", "fomc", "sec-10k"],
      "properties": {
        "canonical_name": "string (inferred)",
        "entity_id": "string (inferred)",
        "mention_count": "integer (inferred)"
      },
      "description": "Node label: PERSON"
    }
  ],
  "relationship_types": [
    {
      "type": "SAID",
      "services": ["ea"],
      "source_labels": ["PERSON"],
      "target_labels": ["STATEMENT"],
      "properties": {
        "sentence_ids": "list (inferred)",
        "method": "string (inferred)"
      },
      "description": "Relationship type: SAID"
    }
  ],
  "indexes": [
    {
      "label": "PERSON",
      "property": "canonical_name",
      "type": "unknown"
    }
  ],
  "property_keys": ["canonical_name", "entity_id", "mention_count", ...],
  "services": {
    "ea": "Earnings Announcement Transcripts",
    "fomc": "FOMC Transcripts",
    "sec-10k": "SEC 10-K Filings",
    "sec-f4": "SEC Form 4 Filings"
  }
}
```

### Service-Specific Schema Format

```json
{
  "version": "1.0",
  "service": "ea",
  "service_name": "Earnings Announcement Transcripts",
  "description": "Memgraph CIPHER schema filtered for Earnings Announcement Transcripts",
  "generated_at": "2025-12-19T15:28:15.717000+00:00",
  "node_types": [...],
  "relationship_types": [...],
  "indexes": [...],
  "total_node_types": 14,
  "total_relationship_types": 7,
  "total_indexes": 4
}
```

## 🔍 Retrieving Schemas

### From Redis CLI

```bash
# Get full schema
redis-cli GET memgraph:cipher:full

# Get service-specific schema
redis-cli GET memgraph:cipher:ea
redis-cli GET memgraph:cipher:fomc
redis-cli GET memgraph:cipher:sec-10k
redis-cli GET memgraph:cipher:sec-f4
```

### From Python

```python
import redis
import json

# Connect to DragonFly
client = redis.Redis(
    host='dragonfly.blacksmith.deerfieldgreen.com',
    port=6379,
    password='your_password',
    decode_responses=True
)

# Get full schema
full_schema = json.loads(client.get('memgraph:cipher:full'))

# Get service-specific schema
ea_schema = json.loads(client.get('memgraph:cipher:ea'))

# Access schema elements
print(f"Node types: {len(full_schema['node_types'])}")
print(f"Relationship types: {len(full_schema['relationship_types'])}")

# Filter by service
ea_nodes = [n for n in full_schema['node_types'] if 'ea' in n.get('services', [])]
```

## 🔧 How It Works

### 1. Schema Introspection

The service queries Memgraph using multiple strategies:

**Primary Queries** (if supported):
- `CALL db.labels()` - Get all node labels
- `CALL db.relationshipTypes()` - Get all relationship types
- `CALL db.propertyKeys()` - Get all property keys
- `SHOW INDEX INFO` - Get all indexes

**Fallback Queries** (when procedures aren't available):
- `MATCH (n) UNWIND labels(n) as label RETURN DISTINCT label` - Get node labels
- `MATCH ()-[r]->() RETURN DISTINCT type(r)` - Get relationship types
- Sample nodes and relationships to discover properties

### 2. Property Type Inference

For each node label and relationship type, the service:
1. Samples up to 10 instances
2. Examines property values
3. Infers types (boolean, integer, float, list, string)

### 3. Service Tagging

Each schema element is tagged based on predefined service definitions:
- Checks if node label exists in service's `node_labels` set
- Checks if relationship type exists in service's `relationship_types` set
- Tags with all matching services (elements can belong to multiple services)

### 4. Schema Filtering

Service-specific schemas are created by:
- Filtering node types where service ID is in `services` array
- Filtering relationship types where service ID is in `services` array
- Filtering indexes for node labels in the filtered node types

### 5. Storage

Schemas are stored in DragonFly as:
- **Key**: `memgraph:cipher:{schema_name}`
- **Value**: JSON string (pretty-printed, 2-space indent)
- **TTL**: None (persistent storage)

## 🎯 Use Cases

### 1. **Complete Schema Overview**
Query `memgraph:cipher:full` to see all node types, relationships, and indexes across all services.

### 2. **Service-Specific Reporting**
Query service-specific schemas (e.g., `memgraph:cipher:ea`) to focus on a single document type.

### 3. **Cross-Service Analysis**
Use the full schema to identify shared node types and relationships between services.

### 4. **Schema Documentation**
Generate documentation from stored schemas for API consumers and developers.

### 5. **Schema Validation**
Compare expected schemas against actual database schemas to detect drift.

## 🛠️ Troubleshooting

### Connection Issues

**Memgraph Connection Failed**
```
❌ Failed to connect to Memgraph: Connection refused
```
- Check that Memgraph is running
- Verify `MEMGRAPH_HOST` and `MEMGRAPH_PORT` in `.env`
- Check firewall/network settings

**DragonFly Connection Failed**
```
❌ Failed to connect to DragonFly: Connection refused
```
- Check that DragonFly is running
- Verify `DRAGONFLY_HOST` and `DRAGONFLY_PORT` in `.env`
- Check authentication credentials

### Schema Query Issues

**Procedures Not Available**
```
⚠️  Could not query node labels (may not be supported): There is no procedure named 'db.labels'
```
- This is normal - the service automatically uses fallback queries
- No action needed

**GQLAlchemy Warnings**
```
GQLAlchemySubclassNotFoundWarning: ({'PERSON'}, <class 'gqlalchemy.models.Node'>)
```
- These are informational warnings from GQLAlchemy
- They don't affect functionality
- Can be safely ignored

## 📝 Maintenance

### Updating Service Definitions

To add a new service or update existing service definitions, edit the `SERVICE_DEFINITIONS` dictionary in `store_schema_dragonfly.py`:

```python
SERVICE_DEFINITIONS = {
    "new-service": {
        "name": "New Service Name",
        "node_labels": {"LABEL1", "LABEL2", ...},
        "relationship_types": {"REL_TYPE1", "REL_TYPE2", ...}
    },
    ...
}
```

### Running Periodically

Consider setting up a cron job or scheduled task to run this service periodically:

```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/publish_schema && /path/to/venv/bin/python store_schema_dragonfly.py
```

## 📊 Performance

- **Schema Introspection**: ~200-500ms (depends on graph size)
- **Schema Storage**: ~1-5 seconds (depends on schema size and network)
- **Total Runtime**: Typically 5-10 seconds for complete execution

## 🔒 Security

- **Credentials**: Store in `.env` file (not committed to version control)
- **Network**: Use SSL/TLS for DragonFly connections in production (`DRAGONFLY_SSL=true`)
- **Authentication**: Supports Memgraph username/password authentication

## 📚 Related Services

- **graph-earnings-announcement-transcripts** - Earnings call transcript processing
- **graph-fomc-transcripts** - Federal Reserve transcript processing
- **graph-sec-10k-filings** - SEC 10-K annual report processing
- **graph-sec-F4-filings** - SEC Form 4 insider trading processing

## 🤝 Contributing

When adding new services or updating schema definitions:

1. Update `SERVICE_DEFINITIONS` in `store_schema_dragonfly.py`
2. Test with actual Memgraph database
3. Verify schemas are stored correctly in DragonFly
4. Update this README if needed

## 📄 License

Part of the Blacksmith graph datasources project.

---

**Last Updated**: December 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready

