#!/usr/bin/env python3
"""
Improved FOMC RSS Consumer with Parallel Processing and Pre-processing

Features:
- Parallel downloads/uploads using ThreadPoolExecutor
- Batch GCS existence checks
- Enhanced text extraction with metadata
- Pre-processing pipeline for spaCy TRF
- Retry logic with exponential backoff
- Progress tracking and statistics
"""

import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from google.cloud import storage
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
FOMC_FEED_URL = "https://www.federalreserve.gov/feeds/press_all.xml"
BASE_URL = "https://www.federalreserve.gov"
GCS_BUCKET_NAME = "blacksmith-sec-filings"
GCS_PREFIX = "fomc"

# Parallel processing configuration
MAX_WORKERS = 20  # Concurrent downloads/uploads
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 30
REQUEST_TIMEOUT = 30
CONNECTION_POOL_SIZE = 50  # Increase connection pool size for parallel requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress connection pool warnings from urllib3 (used by google-cloud-storage)
# These are harmless - the library handles connection reuse automatically
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Thread-safe statistics
stats_lock = Lock()
stats = {
    "press_releases": {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0},
    "minutes_html": {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0},
    "minutes_pdf": {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0},
    "minutes_text": {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0},
    "statements_html": {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0},
    "statements_pdf": {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0},
    "statements_text": {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0},
}

# Metadata index for all documents (thread-safe)
metadata_index_lock = Lock()
metadata_index = []


def extract_metadata_from_html(html_content: bytes, doc_type: str, url: str) -> Dict:
    """Extract metadata from HTML for spaCy processing."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        metadata = {
            "source_url": url,
            "doc_type": doc_type,
            "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        
        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)
        
        # Extract publication date from meta tags
        pub_date = (
            soup.find("meta", property="article:published_time") or
            soup.find("meta", attrs={"name": "pubdate"}) or
            soup.find("meta", attrs={"name": "date"})
        )
        if pub_date:
            metadata["published_date"] = pub_date.get("content", "")
        
        # Extract meeting date from title or content (e.g., "October 28-29, 2025")
        date_pattern = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:-(\d{1,2}))?,\s+(\d{4})"
        title = metadata.get("title", "")
        date_match = re.search(date_pattern, title)
        if date_match:
            metadata["meeting_date"] = date_match.group(0)
        
        # Extract FOMC meeting number if available
        meeting_match = re.search(r"meeting\s+(\d+)", title, re.IGNORECASE)
        if meeting_match:
            metadata["meeting_number"] = meeting_match.group(1)
        
        return metadata
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
        return {"source_url": url, "doc_type": doc_type, "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


def extract_clean_text(soup: BeautifulSoup) -> str:
    """Extract clean, structured text from HTML for NLP processing."""
    # Try to find main content area
    article = (
        soup.find("div", id="article") or
        soup.find("div", class_="article") or
        soup.find("article") or
        soup.find("div", class_="content") or
        soup.find("main") or
        soup.find("div", class_="col-xs-12 col-sm-8") or
        soup.select_one("div.col-xs-12.col-sm-8") or
        soup.find("body")
    )
    
    if not article:
        return ""
    
    # Remove unwanted elements
    for element in article(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        element.decompose()
    
    # Extract text with paragraph preservation
    text_parts = []
    for element in article.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        text = element.get_text(separator=" ", strip=True)
        if text:
            text_parts.append(text)
    
    # Fallback to simple text extraction
    if not text_parts:
        text = article.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    return "\n\n".join(text_parts)


def check_gcs_exists(gcs_path: str) -> bool:
    """Check if blob exists in GCS."""
    try:
        blob = bucket.blob(gcs_path)
        return blob.exists()
    except Exception as e:
        logger.warning(f"Error checking GCS existence for {gcs_path}: {e}")
        return False


def upload_to_gcs(data: bytes, gcs_path: str, content_type: str = "text/html", metadata: Optional[Dict] = None) -> bool:
    """Upload data to GCS with retry logic."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            blob = bucket.blob(gcs_path)
            
            # Check if already exists
            if attempt == 0 and blob.exists():
                return False  # Already exists, skip
            
            # Upload with metadata
            blob.upload_from_file(io.BytesIO(data), content_type=content_type)
            
            # Add custom metadata if provided (for GCS blob metadata)
            if metadata:
                blob.metadata = metadata
                blob.patch()
            
            # Also upload metadata as separate JSON file for spaCy TRF processing
            if metadata and content_type in ["text/plain", "text/html"]:
                metadata_gcs_path = gcs_path.rsplit(".", 1)[0] + ".metadata.json"
                metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
                metadata_blob = bucket.blob(metadata_gcs_path)
                if not metadata_blob.exists():  # Only upload if doesn't exist
                    metadata_blob.upload_from_string(
                        metadata_json,
                        content_type="application/json"
                    )
                    logger.debug(f"Uploaded metadata: {metadata_gcs_path}")
                
                # Add to metadata index for easy discovery
                with metadata_index_lock:
                    metadata_index.append({
                        "gcs_path": gcs_path,
                        "metadata_path": metadata_gcs_path,
                        "content_type": content_type,
                        "metadata": metadata
                    })
            
            return True
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                time.sleep(delay)
                continue
            else:
                logger.error(f"Failed to upload {gcs_path} after {MAX_RETRIES} retries: {e}")
                return False
    return False


def create_session() -> requests.Session:
    """Create a requests session with connection pooling."""
    session = requests.Session()
    session.headers.update({"User-Agent": "research-script/2.0"})
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=INITIAL_RETRY_DELAY,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Create adapter with connection pooling
    adapter = HTTPAdapter(
        pool_connections=CONNECTION_POOL_SIZE,
        pool_maxsize=CONNECTION_POOL_SIZE,
        max_retries=retry_strategy
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def download_and_upload(url: str, gcs_path: str, content_type: str, category: str, metadata: Optional[Dict] = None, session: Optional[requests.Session] = None) -> Tuple[bool, str]:
    """Download URL and upload to GCS."""
    try:
        if session is None:
            session = create_session()
        
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        
        success = upload_to_gcs(resp.content, gcs_path, content_type, metadata)
        return success, "uploaded" if success else "skipped"
    except requests.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False, f"download_error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}")
        return False, f"error: {str(e)}"


def process_press_release(item: Dict) -> List[Tuple[str, str, str, Optional[Dict]]]:
    """Process a press release and return list of tasks for related documents."""
    url = item["link"]
    filename = url.rstrip("/").split("/")[-1]
    gcs_path = f"{GCS_PREFIX}/press_releases/{filename}"
    
    tasks = []
    
    # Download press release
    try:
        session = create_session()
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        
        # Extract metadata
        metadata = extract_metadata_from_html(resp.content, "press_release", url)
        metadata.update({
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "published": item.get("published", ""),
        })
        
        # Upload press release
        if upload_to_gcs(resp.content, gcs_path, "text/html", metadata):
            with stats_lock:
                stats["press_releases"]["uploaded"] += 1
            logger.info(f"Uploaded press release: {filename}")
        else:
            with stats_lock:
                stats["press_releases"]["skipped"] += 1
        
        with stats_lock:
            stats["press_releases"]["total"] += 1
        
        # For FOMC statements and minutes, parse for additional documents
        is_statement = item.get("is_statement", False)
        is_minutes = item.get("is_minutes", False)
        
        if not (is_statement or is_minutes):
            return tasks
        
        # Parse HTML to find document links
        html = resp.content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            
            # Minutes HTML
            if "fomcminutes" in href.lower() and href.lower().endswith(".htm"):
                full_url = urljoin(BASE_URL, href)
                html_filename = full_url.rstrip("/").split("/")[-1]
                tasks.append((
                    full_url,
                    f"{GCS_PREFIX}/minutes_html/{html_filename}",
                    "text/html",
                    "minutes_html"
                ))
            
            # Minutes PDF
            elif "fomcminutes" in href.lower() and href.lower().endswith(".pdf"):
                full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                pdf_filename = full_url.rstrip("/").split("/")[-1]
                tasks.append((
                    full_url,
                    f"{GCS_PREFIX}/minutes_pdf/{pdf_filename}",
                    "application/pdf",
                    "minutes_pdf"
                ))
            
            # Statement PDF
            elif "monetary" in href.lower() and href.lower().endswith(".pdf"):
                if "/monetarypolicy/files/" in href.lower():
                    full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    pdf_filename = full_url.rstrip("/").split("/")[-1]
                    tasks.append((
                        full_url,
                        f"{GCS_PREFIX}/statements_pdf/{pdf_filename}",
                        "application/pdf",
                        "statements_pdf"
                    ))
            
            # Statement HTML (separate pages)
            elif "monetary" in href.lower() and href.lower().endswith(".htm"):
                if filename.lower() not in href.lower():
                    full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    html_filename = full_url.rstrip("/").split("/")[-1]
                    tasks.append((
                        full_url,
                        f"{GCS_PREFIX}/statements_html/{html_filename}",
                        "text/html",
                        "statements_html"
                    ))
        
        # Extract text from statement press release if needed
        if is_statement:
            text = extract_clean_text(soup)
            if text:
                text_filename = filename.replace(".htm", ".txt").replace(".html", ".txt")
                text_gcs_path = f"{GCS_PREFIX}/statements_text/{text_filename}"
                text_bytes = text.encode("utf-8")
                text_metadata = metadata.copy()
                text_metadata["doc_type"] = "statement_text"
                tasks.append((
                    None,  # No URL, we have the data
                    text_gcs_path,
                    "text/plain",
                    "statements_text",
                    text_bytes,
                    text_metadata
                ))
    
    except Exception as e:
        logger.error(f"Error processing press release {url}: {e}")
        with stats_lock:
            stats["press_releases"]["failed"] += 1
    
    return tasks


def process_html_minutes(url: str, gcs_path: str) -> List[Tuple]:
    """Download HTML minutes and extract text."""
    tasks = []
    
    try:
        session = create_session()
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        
        # Extract metadata
        metadata = extract_metadata_from_html(resp.content, "minutes_html", url)
        
        # Upload HTML
        if upload_to_gcs(resp.content, gcs_path, "text/html", metadata):
            with stats_lock:
                stats["minutes_html"]["uploaded"] += 1
        else:
            with stats_lock:
                stats["minutes_html"]["skipped"] += 1
        
        with stats_lock:
            stats["minutes_html"]["total"] += 1
        
        # Extract and upload text
        soup = BeautifulSoup(resp.content, "html.parser")
        text = extract_clean_text(soup)
        
        if text:
            text_filename = gcs_path.split("/")[-1].replace(".htm", ".txt").replace(".html", ".txt")
            text_gcs_path = f"{GCS_PREFIX}/minutes_text/{text_filename}"
            text_bytes = text.encode("utf-8")
            text_metadata = metadata.copy()
            text_metadata["doc_type"] = "minutes_text"
            
            if upload_to_gcs(text_bytes, text_gcs_path, "text/plain", text_metadata):
                with stats_lock:
                    stats["minutes_text"]["uploaded"] += 1
            else:
                with stats_lock:
                    stats["minutes_text"]["skipped"] += 1
            
            with stats_lock:
                stats["minutes_text"]["total"] += 1
    
    except Exception as e:
        logger.error(f"Error processing minutes HTML {url}: {e}")
        with stats_lock:
            stats["minutes_html"]["failed"] += 1
    
    return tasks


def process_document_task(task: Tuple) -> Tuple[bool, str]:
    """Process a single document download/upload task."""
    if len(task) == 6:  # Text extraction task (has data)
        _, gcs_path, content_type, category, data, metadata = task
        success = upload_to_gcs(data, gcs_path, content_type, metadata)
        status = "uploaded" if success else "skipped"
    else:
        url, gcs_path, content_type, category = task[:4]
        
        # Special handling for HTML minutes (extract text too)
        if category == "minutes_html":
            process_html_minutes(url, gcs_path)
            return True, "processed"
        
        # Regular download and upload
        success, status = download_and_upload(url, gcs_path, content_type, category)
    
    # Update statistics
    with stats_lock:
        if category in stats:
            stats[category]["total"] += 1
            if success:
                stats[category]["uploaded"] += 1
            elif status == "skipped":
                stats[category]["skipped"] += 1
            else:
                stats[category]["failed"] += 1
    
    return success, status


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("FOMC RSS Consumer - Improved Version")
    logger.info("=" * 80)
    logger.info(f"Feed URL: {FOMC_FEED_URL}")
    logger.info(f"GCS Bucket: {GCS_BUCKET_NAME}")
    logger.info(f"GCS Prefix: {GCS_PREFIX}")
    logger.info(f"Max Workers: {MAX_WORKERS}")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Parse RSS feed
    logger.info("Parsing RSS feed...")
    feed = feedparser.parse(FOMC_FEED_URL)
    
    # Process all items
    all_items = []
    for e in feed.entries:
        category = e.tags[0].term if getattr(e, "tags", None) else ""
        title = e.title
        
        is_monetary = category == "Monetary Policy"
        is_statement = "FOMC statement" in title
        is_minutes = "Minutes of the Federal Open Market Committee" in title
        
        all_items.append({
            "title": title,
            "link": e.link,
            "published": getattr(e, "published", ""),
            "category": category,
            "is_statement": is_statement,
            "is_minutes": is_minutes,
        })
    
    logger.info(f"Found {len(all_items)} items in RSS feed")
    
    # Process press releases and collect document tasks
    logger.info("Processing press releases...")
    all_document_tasks = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit press release processing
        future_to_item = {
            executor.submit(process_press_release, item): item
            for item in all_items
        }
        
        # Collect document tasks from press releases
        for future in as_completed(future_to_item):
            try:
                tasks = future.result()
                all_document_tasks.extend(tasks)
            except Exception as e:
                logger.error(f"Error in press release processing: {e}")
    
    logger.info(f"Collected {len(all_document_tasks)} document tasks")
    
    # Process all document tasks in parallel
    logger.info("Processing documents in parallel...")
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_document_task, task): task
            for task in all_document_tasks
        }
        
        for future in as_completed(future_to_task):
            completed += 1
            try:
                success, status = future.result()
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(all_document_tasks)} documents processed")
            except Exception as e:
                logger.error(f"Error processing document task: {e}")
    
    # Create and upload metadata index
    logger.info("Creating metadata index...")
    with metadata_index_lock:
        index_data = {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_documents": len(metadata_index),
            "documents": metadata_index
        }
    
    index_json = json.dumps(index_data, indent=2, ensure_ascii=False)
    index_gcs_path = f"{GCS_PREFIX}/metadata_index.json"
    index_blob = bucket.blob(index_gcs_path)
    index_blob.upload_from_string(index_json, content_type="application/json")
    logger.info(f"Uploaded metadata index: gs://{GCS_BUCKET_NAME}/{index_gcs_path}")
    
    # Also create a simplified index by document type for easier filtering
    index_by_type = {}
    for doc in metadata_index:
        doc_type = doc["metadata"].get("doc_type", "unknown")
        if doc_type not in index_by_type:
            index_by_type[doc_type] = []
        index_by_type[doc_type].append({
            "gcs_path": doc["gcs_path"],
            "metadata_path": doc["metadata_path"],
            "title": doc["metadata"].get("title", ""),
            "meeting_date": doc["metadata"].get("meeting_date", ""),
            "published_date": doc["metadata"].get("published_date", ""),
        })
    
    index_by_type_data = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_documents": len(metadata_index),
        "documents_by_type": index_by_type
    }
    
    index_by_type_json = json.dumps(index_by_type_data, indent=2, ensure_ascii=False)
    index_by_type_gcs_path = f"{GCS_PREFIX}/metadata_index_by_type.json"
    index_by_type_blob = bucket.blob(index_by_type_gcs_path)
    index_by_type_blob.upload_from_string(index_by_type_json, content_type="application/json")
    logger.info(f"Uploaded metadata index by type: gs://{GCS_BUCKET_NAME}/{index_by_type_gcs_path}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    
    for category, category_stats in stats.items():
        total = category_stats["total"]
        if total > 0:
            logger.info(f"{category}: {category_stats['uploaded']} uploaded, "
                       f"{category_stats['skipped']} skipped, "
                       f"{category_stats['failed']} failed (total: {total})")
    
    logger.info(f"Metadata index: {len(metadata_index)} documents indexed")
    logger.info(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

