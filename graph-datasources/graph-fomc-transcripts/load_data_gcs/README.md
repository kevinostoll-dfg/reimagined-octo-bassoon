# 📊 FOMC Document Ingestion & NLP Processing Pipeline

> **Automated pipeline for ingesting Federal Reserve FOMC documents and extracting structured insights using spaCy Transformer models**

---

## 🎯 Overview

This pipeline automates the collection, processing, and NLP analysis of Federal Open Market Committee (FOMC) documents from the Federal Reserve's RSS feed. It downloads press releases, statements, and minutes, extracts clean text, and prepares documents for advanced NLP processing using spaCy's transformer-based models.

---

## 🚀 Features

### 📥 **Document Ingestion** (`consume_rss_improved.py`)

- ✅ **Parallel Processing**: Downloads and processes documents concurrently (20 workers)
- ✅ **RSS Feed Integration**: Automatically fetches all FOMC documents from `press_all.xml`
- ✅ **Multi-Format Support**: Handles HTML, PDF, and extracted text files
- ✅ **GCS Integration**: Streams directly to Google Cloud Storage (no local disk)
- ✅ **Metadata Extraction**: Automatically extracts titles, dates, meeting numbers, and more
- ✅ **Smart Deduplication**: Skips already-processed documents
- ✅ **Connection Pooling**: Optimized HTTP connections for high throughput
- ✅ **Retry Logic**: Exponential backoff for resilient downloads
- ✅ **Progress Tracking**: Real-time statistics and progress reporting

### 📋 **Document Types Processed**

- 📄 **Press Releases**: All Federal Reserve press releases
- 📝 **FOMC Statements**: Post-meeting policy statements
- 📑 **FOMC Minutes**: Detailed meeting minutes
- 📊 **Economic Projections**: FOMC economic outlook documents
- 📈 **Discount Rate Minutes**: Board discount rate meeting minutes

### 🗂️ **GCS Structure**

```
gs://blacksmith-sec-filings/fomc/
├── press_releases/          # Press release HTML pages
├── minutes_html/            # FOMC minutes HTML pages
├── minutes_pdf/             # FOMC minutes PDF files
├── minutes_text/            # Extracted plain text from minutes
├── statements_html/         # FOMC statement HTML pages
├── statements_pdf/          # FOMC statement PDF files
├── statements_text/         # Extracted plain text from statements
├── metadata_index.json      # Complete metadata index
└── metadata_index_by_type.json  # Metadata grouped by document type
```

---

## 🔧 Installation & Setup

### Prerequisites

```bash
python >= 3.8
Google Cloud SDK configured with appropriate credentials
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `requests>=2.31.0` - HTTP client with connection pooling
- `feedparser` - RSS feed parsing
- `beautifulsoup4>=4.12.0` - HTML parsing and text extraction
- `google-cloud-storage>=2.10.0` - GCS integration

### Configuration

The script uses the following GCS configuration:

- **Bucket**: `blacksmith-sec-filings`
- **Prefix**: `fomc`
- **Max Workers**: 20 (configurable)
- **Connection Pool Size**: 50

---

## 🏃 Usage

### Run the Ingestion Script

```bash
python3 consume_rss_improved.py
```

### Expected Output

```
================================================================================
FOMC RSS Consumer - Improved Version
================================================================================
Feed URL: https://www.federalreserve.gov/feeds/press_all.xml
GCS Bucket: blacksmith-sec-filings
GCS Prefix: fomc
Max Workers: 20
================================================================================
Parsing RSS feed...
Found 20 items in RSS feed
Processing press releases...
Collected 32 document tasks
Processing documents in parallel...
Progress: 10/32 documents processed
Progress: 20/32 documents processed
Progress: 30/32 documents processed
Creating metadata index...
Uploaded metadata index: gs://blacksmith-sec-filings/fomc/metadata_index.json
Uploaded metadata index by type: gs://blacksmith-sec-filings/fomc/metadata_index_by_type.json
================================================================================
PROCESSING COMPLETE
================================================================================
press_releases: 20 uploaded, 0 skipped, 0 failed (total: 20)
minutes_html: 1 uploaded, 0 skipped, 0 failed (total: 1)
minutes_pdf: 1 uploaded, 0 skipped, 0 failed (total: 1)
minutes_text: 1 uploaded, 0 skipped, 0 failed (total: 1)
statements_html: 19 uploaded, 9 skipped, 0 failed (total: 28)
statements_pdf: 1 uploaded, 0 skipped, 0 failed (total: 1)
statements_text: 1 uploaded, 0 skipped, 0 failed (total: 1)
Metadata index: 42 documents indexed
Time elapsed: 9.80 seconds (0.2 minutes)
================================================================================
```

---

## 📊 Metadata Structure

Each document includes comprehensive metadata stored as JSON files:

### Metadata Fields

```json
{
  "source_url": "https://www.federalreserve.gov/...",
  "doc_type": "minutes_text",
  "extracted_at": "2025-12-11T12:28:25Z",
  "title": "Minutes of the Federal Open Market Committee, October 28-29, 2025",
  "published_date": "2025-11-19",
  "meeting_date": "October 28-29, 2025",
  "meeting_number": "2025-10",
  "category": "Monetary Policy"
}
```

### Index Files

- **`metadata_index.json`**: Complete index with full metadata for all documents
- **`metadata_index_by_type.json`**: Simplified index grouped by document type for easy filtering

---

## 🤖 spaCy TRF Entity & Relationship Discovery

### 🎯 Overview

The next phase of this pipeline will use **spaCy Transformer (TRF) models** to extract entities, relationships, and structured insights from FOMC documents. This enables advanced analysis of monetary policy decisions, economic indicators, and policy maker statements.

### 🔍 Planned NLP Capabilities

#### 1. **Named Entity Recognition (NER)** 🏷️

Extract structured entities from FOMC documents:

- **👥 People**: Federal Reserve officials, economists, policymakers
  - *Example*: "Jerome H. Powell", "John C. Williams", "Michelle W. Bowman"
  
- **📊 Economic Indicators**: Key metrics and statistics
  - *Example*: "unemployment rate", "inflation", "GDP growth", "interest rates"
  
- **💰 Financial Terms**: Monetary policy instruments
  - *Example*: "federal funds rate", "reserve balances", "quantitative easing"
  
- **📅 Dates & Timeframes**: Meeting dates, policy periods
  - *Example*: "October 28-29, 2025", "three weeks", "next meeting"
  
- **🏛️ Organizations**: Federal Reserve entities, economic institutions
  - *Example*: "Federal Open Market Committee", "Board of Governors"

#### 2. **Relationship Extraction** 🔗

Identify relationships between entities:

- **Policy Decisions** → **Economic Indicators**
  - *Example*: "The Committee decided to lower the target range for the federal funds rate by 1/4 percentage point due to rising unemployment"
  
- **Officials** → **Statements/Positions**
  - *Example*: "Jerome H. Powell stated that inflation remains elevated"
  
- **Meetings** → **Outcomes**
  - *Example*: "The October 28-29 meeting resulted in a rate cut decision"
  
- **Economic Conditions** → **Policy Responses**
  - *Example*: "In response to slowing job gains, the Committee adjusted monetary policy"

#### 3. **Sentiment & Tone Analysis** 😊😐😟

Analyze the sentiment and tone of policy statements:

- **Hawkish vs. Dovish**: Policy stance indicators
- **Uncertainty Levels**: Language indicating economic uncertainty
- **Confidence Measures**: Strength of policy commitments

#### 4. **Temporal Relationships** ⏰

Track changes over time:

- **Policy Evolution**: How policy positions change across meetings
- **Economic Trend Analysis**: Tracking mentions of economic indicators over time
- **Decision Patterns**: Identifying recurring policy decision patterns

#### 5. **Knowledge Graph Construction** 🕸️

Build a knowledge graph connecting:

```
Meeting → Participants → Statements → Decisions → Economic Indicators → Outcomes
```

### 🛠️ Technical Approach

#### Model Selection

- **Base Model**: `en_core_web_trf` (spaCy Transformer model)
- **Custom Training**: Fine-tuned on financial/economic domain data
- **Entity Linking**: Link entities to external knowledge bases (e.g., FRED, economic databases)

#### Processing Pipeline

```python
# Pseudo-code for spaCy TRF processing
1. Load documents from GCS using metadata index
2. Process with spaCy TRF model:
   - Tokenization
   - Named Entity Recognition
   - Dependency Parsing
   - Relationship Extraction
3. Extract structured data:
   - Entities with confidence scores
   - Relationships with types
   - Temporal information
4. Store results:
   - Entity database
   - Relationship graph
   - Document annotations
```

#### Output Structure

```
fomc/
├── processed/
│   ├── entities/           # Extracted entities per document
│   ├── relationships/      # Extracted relationships
│   ├── annotations/        # Full spaCy Doc objects (serialized)
│   └── knowledge_graph/    # Graph database exports
```

### 📈 Use Cases

1. **Policy Analysis** 📊
   - Track policy decisions and their justifications
   - Analyze voting patterns and dissents
   - Identify policy pivot points

2. **Economic Indicator Tracking** 📉📈
   - Monitor mentions of key economic metrics
   - Track sentiment around economic conditions
   - Identify leading indicators

3. **Official Statement Analysis** 🎤
   - Extract key quotes and positions
   - Track consistency of messaging
   - Identify policy signals

4. **Historical Research** 📚
   - Compare policy responses across economic cycles
   - Analyze language evolution over time
   - Build policy decision timelines

5. **Predictive Analytics** 🔮
   - Identify patterns preceding policy changes
   - Predict policy decisions based on language
   - Forecast economic outlook based on statements

---

## 🔄 Pipeline Workflow

```mermaid
graph LR
    A[RSS Feed] --> B[Parse Feed]
    B --> C[Download Press Releases]
    C --> D[Extract Links]
    D --> E[Download Documents]
    E --> F[Extract Text]
    F --> G[Extract Metadata]
    G --> H[Upload to GCS]
    H --> I[Create Index]
    I --> J[spaCy TRF Processing]
    J --> K[Entity Extraction]
    K --> L[Relationship Discovery]
    L --> M[Knowledge Graph]
```

---

## 📝 Example: Loading Documents for spaCy Processing

```python
from google.cloud import storage
import json
import spacy

# Load spaCy TRF model
nlp = spacy.load("en_core_web_trf")

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket("blacksmith-sec-filings")

# Load metadata index
index_blob = bucket.blob("fomc/metadata_index_by_type.json")
index = json.loads(index_blob.download_as_text())

# Process all minutes text documents
for doc in index["documents_by_type"]["minutes_text"]:
    # Load text
    text_blob = bucket.blob(doc["gcs_path"])
    text = text_blob.download_as_text()
    
    # Load metadata
    metadata_blob = bucket.blob(doc["metadata_path"])
    metadata = json.loads(metadata_blob.download_as_text())
    
    # Process with spaCy TRF
    doc_nlp = nlp(text)
    
    # Extract entities
    entities = [(ent.text, ent.label_, ent.start, ent.end) 
                for ent in doc_nlp.ents]
    
    # Extract relationships (custom logic)
    relationships = extract_relationships(doc_nlp, metadata)
    
    # Store results
    save_entities_and_relationships(doc["gcs_path"], entities, relationships)
```

---

## 🎯 Future Enhancements

- [ ] **Custom spaCy Model Training**: Fine-tune on FOMC-specific terminology
- [ ] **Multi-document Analysis**: Cross-document relationship extraction
- [ ] **Real-time Processing**: Stream processing for new documents
- [ ] **API Endpoints**: REST API for querying extracted entities and relationships
- [ ] **Visualization Dashboard**: Interactive graphs and timelines
- [ ] **Alerting System**: Notifications for policy changes or key events

---

## 📚 Resources

- [Federal Reserve RSS Feed](https://www.federalreserve.gov/feeds/press_all.xml)
- [spaCy Documentation](https://spacy.io/)
- [spaCy Transformer Models](https://spacy.io/models/en#en_core_web_trf)
- [Google Cloud Storage](https://cloud.google.com/storage)

---

## 📄 License

This project is part of the Blacksmith data processing pipeline.

---

## 👥 Contributing

Contributions welcome! Please ensure all tests pass and follow the existing code style.

---

**Built with ❤️ for financial data analysis and NLP research**

