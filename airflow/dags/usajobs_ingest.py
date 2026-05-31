from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import os

API_URL = os.getenv("DOCUQUERY_API_URL", "https://docuquery-production-872a.up.railway.app")

TECH_KEYWORDS = [
    "data engineer", "software engineer", "machine learning",
    "cloud architect", "cybersecurity", "data scientist",
    "devops", "backend engineer", "full stack"
]

default_args = {
    "owner": "michael",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

def fetch_and_ingest_jobs():
    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": "michael.demissie@gwu.edu",
        "Authorization-Key": os.getenv("USAJOBS_API_KEY", "")
    }

    all_jobs = []

    for keyword in TECH_KEYWORDS:
        params = {
            "Keyword": keyword,
            "LocationName": "Washington DC",
            "ResultsPerPage": 50,
            "Fields": "Min"
        }
        try:
            res = requests.get(
                "https://data.usajobs.gov/api/search",
                headers=headers,
                params=params,
                timeout=10
            )
            data = res.json()
            jobs = data.get("SearchResult", {}).get("SearchResultItems", [])
            all_jobs.extend(jobs)
            print(f"Fetched {len(jobs)} jobs for '{keyword}'")
        except Exception as e:
            print(f"Error fetching {keyword}: {e}")

    seen = set()
    unique_jobs = []
    for job in all_jobs:
        jid = job.get("MatchedObjectId")
        if jid and jid not in seen:
            seen.add(jid)
            unique_jobs.append(job)

    print(f"Total unique jobs: {len(unique_jobs)}")

    for job in unique_jobs:
        detail = job.get("MatchedObjectDescriptor", {})
        title = detail.get("PositionTitle", "Unknown")
        agency = detail.get("OrganizationName", "")
        location = detail.get("PositionLocationDisplay", "")
        salary_min = detail.get("PositionRemuneration", [{}])[0].get("MinimumRange", "")
        salary_max = detail.get("PositionRemuneration", [{}])[0].get("MaximumRange", "")
        qualifications = detail.get("QualificationSummary", "")
        duties = detail.get("UserArea", {}).get("Details", {}).get("MajorDuties", "")
        close_date = detail.get("ApplicationCloseDate", "")
        url = detail.get("PositionURI", "")

        content = f"""
Job Title: {title}
Agency: {agency}
Location: {location}
Salary: ${salary_min} - ${salary_max}
Close Date: {close_date}
URL: {url}

Qualifications:
{qualifications}

Major Duties:
{duties}
        """.strip()

        try:
            requests.post(
                f"{API_URL}/ingest",
                json={
                    "title": f"{title} — {agency}",
                    "source": url,
                    "content": content,
                    "mode": "jobs"
                },
                timeout=30
            )
        except Exception as e:
            print(f"Error ingesting {title}: {e}")

    print("Ingestion complete.")

with DAG(
    dag_id="usajobs_dc_tech_ingest",
    default_args=default_args,
    description="Daily ingestion of DC tech jobs from USAJobs API",
    schedule_interval="0 7 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rag", "jobs", "usajobs"]
) as dag:

    ingest_task = PythonOperator(
        task_id="fetch_and_ingest_jobs",
        python_callable=fetch_and_ingest_jobs,
    )
