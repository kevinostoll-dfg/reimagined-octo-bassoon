import io
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google.cloud import storage

FOMC_FEED_URL = "https://www.federalreserve.gov/feeds/press_all.xml"
BASE_URL = "https://www.federalreserve.gov"
GCS_BUCKET_NAME = "blacksmith-sec-filings"
GCS_PREFIX = "fomc"

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

feed = feedparser.parse(FOMC_FEED_URL)

# Process ALL items from the RSS feed
all_items = []
for e in feed.entries:
    category = e.tags[0].term if getattr(e, "tags", None) else ""
    title = e.title

    is_monetary = category == "Monetary Policy"
    is_statement = "FOMC statement" in title
    is_minutes = "Minutes of the Federal Open Market Committee" in title

    all_items.append(
        {
            "title": title,
            "link": e.link,
            "published": getattr(e, "published", ""),
            "category": category,
            "is_statement": is_statement,
            "is_minutes": is_minutes,
        }
    )

def upload_to_gcs(data: bytes, gcs_path: str, content_type: str = "text/html") -> bool:
    """Upload data directly to GCS bucket."""
    try:
        blob = bucket.blob(gcs_path)
        # Check if blob already exists
        if blob.exists():
            print(f"  Skipping existing: gs://{GCS_BUCKET_NAME}/{gcs_path}")
            return False
        blob.upload_from_file(io.BytesIO(data), content_type=content_type)
        print(f"  Uploaded: gs://{GCS_BUCKET_NAME}/{gcs_path}")
        return True
    except Exception as exc:
        print(f"  ERROR uploading to gs://{GCS_BUCKET_NAME}/{gcs_path}: {exc}")
        return False

# Download each press release HTML page
session = requests.Session()
session.headers.update({"User-Agent": "research-script/1.0"})

for item in all_items:
    url = item["link"]
    filename = url.rstrip("/").split("/")[-1]  # e.g. monetary20251119a.htm
    gcs_path = f"{GCS_PREFIX}/press_releases/{filename}"

    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"ERROR downloading {url}: {exc}")
        continue

    # Upload press release HTML to GCS
    upload_to_gcs(resp.content, gcs_path, content_type="text/html")
    print(f"Saved {url} -> gs://{GCS_BUCKET_NAME}/{gcs_path}")

    # For FOMC statements and minutes, parse the press-release HTML to find links
    # to actual minutes/statement documents (PDFs and separate HTML pages)
    is_statement = item.get("is_statement", False)
    is_minutes = item.get("is_minutes", False)
    
    if not (is_statement or is_minutes):
        # For non-FOMC items, just download the press release and continue
        continue

    # Parse the press-release HTML to find links to actual minutes/statement documents
    try:
        html = resp.content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        minutes_pdf_links = []
        minutes_html_links = []
        statement_pdf_links = []
        statement_html_links = []

        # Find all links in the document
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = (a.get_text() or "").strip().lower()

            # Look for FOMC minutes HTML links (fomcminutesYYYYMMDD.htm)
            if "fomcminutes" in href.lower() and href.lower().endswith(".htm"):
                full_url = urljoin(BASE_URL, href)
                if full_url not in minutes_html_links:
                    minutes_html_links.append(full_url)

            # Look for FOMC minutes PDF links (fomcminutesYYYYMMDD.pdf)
            elif "fomcminutes" in href.lower() and href.lower().endswith(".pdf"):
                full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                if full_url not in minutes_pdf_links:
                    minutes_pdf_links.append(full_url)

            # Look for statement PDF links (monetaryYYYYMMDDa1.pdf pattern)
            # These are typically in /monetarypolicy/files/ directory
            elif "monetary" in href.lower() and href.lower().endswith(".pdf"):
                # Check if it matches the statement PDF pattern (monetaryYYYYMMDDa1.pdf or similar)
                if "/monetarypolicy/files/" in href.lower() or "monetary" in href.lower():
                    full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    if full_url not in statement_pdf_links:
                        statement_pdf_links.append(full_url)

            # Look for statement HTML links (monetaryYYYYMMDDa1.htm or similar)
            # These might be separate HTML pages for statements
            elif "monetary" in href.lower() and href.lower().endswith(".htm"):
                # Exclude the press-release page itself
                if filename.lower() not in href.lower() or href.lower() != f"/newsevents/pressreleases/{filename.lower()}":
                    full_url = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    if full_url not in statement_html_links:
                        statement_html_links.append(full_url)

        # Download HTML minutes pages
        for html_url in minutes_html_links:
            html_filename = html_url.rstrip("/").split("/")[-1]
            gcs_path = f"{GCS_PREFIX}/minutes_html/{html_filename}"

            try:
                html_resp = session.get(html_url, timeout=15)
                html_resp.raise_for_status()
                
                # Upload HTML minutes to GCS
                upload_to_gcs(html_resp.content, gcs_path, content_type="text/html")
                print(f"  Saved HTML minutes: {html_url} -> gs://{GCS_BUCKET_NAME}/{gcs_path}")

                # Extract plain text from HTML minutes
                try:
                    minutes_soup = BeautifulSoup(html_resp.content, "html.parser")
                    article = (
                        minutes_soup.find("div", id="article") or
                        minutes_soup.find("div", class_="article") or
                        minutes_soup.find("article") or
                        minutes_soup.find("div", class_="content") or
                        minutes_soup.find("main") or
                        minutes_soup.find("body")
                    )

                    if article:
                        for script in article(["script", "style", "nav", "header", "footer"]):
                            script.decompose()
                        text = article.get_text(separator="\n", strip=True)
                        text_filename = html_filename.replace(".htm", ".txt").replace(".html", ".txt")
                        text_gcs_path = f"{GCS_PREFIX}/minutes_text/{text_filename}"
                        text_bytes = text.encode("utf-8")
                        upload_to_gcs(text_bytes, text_gcs_path, content_type="text/plain")
                        print(f"    Extracted text: gs://{GCS_BUCKET_NAME}/{text_gcs_path}")
                except Exception as text_exc:
                    print(f"    WARNING: Could not extract text from {html_filename}: {text_exc}")

            except requests.RequestException as exc:
                print(f"  ERROR downloading HTML minutes {html_url}: {exc}")

        # Download PDF minutes files
        for pdf_url in minutes_pdf_links:
            pdf_filename = pdf_url.rstrip("/").split("/")[-1]
            gcs_path = f"{GCS_PREFIX}/minutes_pdf/{pdf_filename}"

            try:
                pdf_resp = session.get(pdf_url, timeout=15)
                pdf_resp.raise_for_status()
                upload_to_gcs(pdf_resp.content, gcs_path, content_type="application/pdf")
                print(f"  Saved PDF minutes: {pdf_url} -> gs://{GCS_BUCKET_NAME}/{gcs_path}")
            except requests.RequestException as exc:
                print(f"  ERROR downloading PDF minutes {pdf_url}: {exc}")

        # Download HTML statement pages (if separate from press-release)
        for html_url in statement_html_links:
            html_filename = html_url.rstrip("/").split("/")[-1]
            gcs_path = f"{GCS_PREFIX}/statements_html/{html_filename}"

            try:
                html_resp = session.get(html_url, timeout=15)
                html_resp.raise_for_status()
                upload_to_gcs(html_resp.content, gcs_path, content_type="text/html")
                print(f"  Saved HTML statement: {html_url} -> gs://{GCS_BUCKET_NAME}/{gcs_path}")

                # Extract plain text from HTML statements
                try:
                    statement_soup = BeautifulSoup(html_resp.content, "html.parser")
                    article = (
                        statement_soup.find("div", id="article") or
                        statement_soup.find("div", class_="article") or
                        statement_soup.find("article") or
                        statement_soup.find("div", class_="content") or
                        statement_soup.find("main") or
                        statement_soup.find("body")
                    )

                    if article:
                        for script in article(["script", "style", "nav", "header", "footer"]):
                            script.decompose()
                        text = article.get_text(separator="\n", strip=True)
                        text_filename = html_filename.replace(".htm", ".txt").replace(".html", ".txt")
                        text_gcs_path = f"{GCS_PREFIX}/statements_text/{text_filename}"
                        text_bytes = text.encode("utf-8")
                        upload_to_gcs(text_bytes, text_gcs_path, content_type="text/plain")
                        print(f"    Extracted text: gs://{GCS_BUCKET_NAME}/{text_gcs_path}")
                except Exception as text_exc:
                    print(f"    WARNING: Could not extract text from {html_filename}: {text_exc}")

            except requests.RequestException as exc:
                print(f"  ERROR downloading HTML statement {html_url}: {exc}")

        # Download PDF statement files
        for pdf_url in statement_pdf_links:
            pdf_filename = pdf_url.rstrip("/").split("/")[-1]
            gcs_path = f"{GCS_PREFIX}/statements_pdf/{pdf_filename}"

            try:
                pdf_resp = session.get(pdf_url, timeout=15)
                pdf_resp.raise_for_status()
                upload_to_gcs(pdf_resp.content, gcs_path, content_type="application/pdf")
                print(f"  Saved PDF statement: {pdf_url} -> gs://{GCS_BUCKET_NAME}/{gcs_path}")
            except requests.RequestException as exc:
                print(f"  ERROR downloading PDF statement {pdf_url}: {exc}")

        # For statements, also extract text from the press-release page itself
        # (since statement text is often embedded in the press-release)
        if is_statement:
            try:
                # Try to find the main content area in the press-release
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

                if article:
                    # Remove script, style, nav, header, footer elements
                    for script in article(["script", "style", "nav", "header", "footer"]):
                        script.decompose()

                    text = article.get_text(separator="\n", strip=True)
                    text_filename = filename.replace(".htm", ".txt").replace(".html", ".txt")
                    text_gcs_path = f"{GCS_PREFIX}/statements_text/{text_filename}"
                    text_bytes = text.encode("utf-8")
                    upload_to_gcs(text_bytes, text_gcs_path, content_type="text/plain")
                    print(f"  Extracted statement text from press-release: gs://{GCS_BUCKET_NAME}/{text_gcs_path}")
            except Exception as text_exc:
                print(f"  WARNING: Could not extract statement text from {filename}: {text_exc}")

    except Exception as parse_exc:
        print(f"  ERROR parsing {filename}: {parse_exc}")
