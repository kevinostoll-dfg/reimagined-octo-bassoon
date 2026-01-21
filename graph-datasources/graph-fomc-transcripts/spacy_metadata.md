from google.cloud import storage
import json

storage_client = storage.Client()
bucket = storage_client.bucket("blacksmith-sec-filings")

# Load the index
index_blob = bucket.blob("fomc/metadata_index_by_type.json")
index = json.loads(index_blob.download_as_text())

# Get all minutes text documents
minutes_docs = index["documents_by_type"]["minutes_text"]

# Process each document
for doc in minutes_docs:
    # Load text
    text_blob = bucket.blob(doc["gcs_path"])
    text = text_blob.download_as_text()
    
    # Load metadata
    metadata_blob = bucket.blob(doc["metadata_path"])
    metadata = json.loads(metadata_blob.download_as_text())
    
    # Process with spaCy TRF
    # ...