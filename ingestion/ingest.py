import sys
import os
import requests
import argparse
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8000")

def read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(filepath: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

def read_url(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ")
    except ImportError:
        print("beautifulsoup4 not installed. Run: pip install beautifulsoup4 lxml")
        sys.exit(1)

def ingest(title: str, content: str, source: str = "", mode: str = "personal"):
    response = requests.post(
        f"{API_URL}/ingest",
        json={"title": title, "content": content, "source": source, "mode": mode}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Ingested: {data['message']} (document_id={data['document_id']})")
    else:
        print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a document into DocuQuery")
    parser.add_argument("--file", help="Path to a .txt or .pdf file")
    parser.add_argument("--url", help="URL to scrape and ingest")
    parser.add_argument("--title", required=True, help="Title for this document")
    parser.add_argument("--mode", default="personal", help="Mode: personal or jobs")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if path.suffix == ".pdf":
            content = read_pdf(args.file)
        else:
            content = read_txt(args.file)
        ingest(args.title, content, source=args.file, mode=args.mode)

    elif args.url:
        content = read_url(args.url)
        ingest(args.title, content, source=args.url, mode=args.mode)

    else:
        print("Provide --file or --url")
        sys.exit(1)
