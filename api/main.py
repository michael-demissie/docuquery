from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import threading
from pydantic import BaseModel
from groq import Groq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import json
import time
from typing import List, Optional
from dotenv import load_dotenv

from database import get_db, execute_query
from embeddings import embed_text
from chunker import chunk_text, clean_text

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

def run_cleanup():
    while True:
        time.sleep(120)
        try:
            conn = next(get_db())
            execute_query(conn, "DELETE FROM documents WHERE session_id != 'default' AND mode = 'personal' AND created_at < NOW() AT TIME ZONE 'UTC' - INTERVAL '2 minutes'")
            print("Auto cleanup ran")
        except Exception as e:
            print(f"Cleanup error: {e}")

cleanup_thread = threading.Thread(target=run_cleanup, daemon=True)
cleanup_thread.start()

app = FastAPI(title="DocuQuery RAG API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class IngestRequest(BaseModel):
    title: str
    source: str = ""
    content: str
    mode: str = "personal"
    session_id: str = "default"

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    history: Optional[List[Message]] = []
    mode: str = "personal"
    session_id: str = "default"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(request: IngestRequest, conn=Depends(get_db)):
    cleaned = clean_text(request.content)
    chunks = chunk_text(cleaned)

    doc = execute_query(
        conn,
        "INSERT INTO documents (title, source, mode, session_id) VALUES (%s, %s, %s, %s) RETURNING id",
        (request.title, request.source, request.mode, request.session_id)
    )
    document_id = doc[0]["id"]

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        execute_query(
            conn,
            "INSERT INTO chunks (document_id, content, embedding, chunk_index) VALUES (%s, %s, %s, %s)",
            (document_id, chunk, embedding, i)
        )

    return {"message": f"Ingested {len(chunks)} chunks", "document_id": document_id}

@app.post("/query")
@limiter.limit("10/minute")
def query(request: Request, body: QueryRequest, conn=Depends(get_db)):
    question_embedding = embed_text(body.question)

    results = execute_query(
        conn,
        """
        SELECT c.content, d.title,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.mode = %s AND (d.session_id = %s OR d.session_id = 'default')
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (question_embedding, body.mode, body.session_id, question_embedding, body.top_k)
    )

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    context = "\n\n".join([r["content"] for r in results])
    sources = [{"title": r["title"], "similarity": round(r["similarity"], 4)} for r in results]

    system_prompt = f"""You are a helpful assistant. Answer questions using only the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in (body.history or []):
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": body.question})

    def stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        stream_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=512,
            stream=True,
        )

        for chunk in stream_response:
            token = chunk.choices[0].delta.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/documents")
def list_documents(mode: str = "personal", session_id: str = "default", conn=Depends(get_db)):
    if mode == "jobs":
        docs = execute_query(conn, "SELECT id, title, source, created_at, mode FROM documents WHERE mode = %s ORDER BY created_at DESC", (mode,))
    else:
        docs = execute_query(conn, "SELECT id, title, source, created_at, mode FROM documents WHERE mode = %s AND (session_id = %s OR session_id = 'default') ORDER BY created_at DESC", (mode, session_id))
    return {"documents": docs}

@app.delete("/documents/{document_id}")
def delete_document(document_id: int, conn=Depends(get_db)):
    execute_query(conn, "DELETE FROM documents WHERE id = %s", (document_id,))
    return {"message": f"Document {document_id} deleted"}

@app.delete("/sessions/{session_id}")
def cleanup_session(session_id: str, conn=Depends(get_db)):
    execute_query(conn, "DELETE FROM documents WHERE session_id = %s AND session_id != 'default'", (session_id,))
    return {"message": f"Session {session_id} cleaned up"}

@app.post("/cleanup-expired")
def cleanup_expired(conn=Depends(get_db)):
    execute_query(conn, "DELETE FROM documents WHERE session_id != 'default' AND mode = 'personal' AND created_at < NOW() AT TIME ZONE 'UTC' - INTERVAL '2 minutes'")
    return {"message": "Expired sessions cleaned up"}

from fastapi import UploadFile, File, Form
import pypdf
import io

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_file(request: Request, file: UploadFile = File(...), title: str = Form(""), session_id: str = Form("default"), conn=Depends(get_db)):
    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    if file.filename.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    doc_title = title or file.filename
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    doc = execute_query(
        conn,
        "INSERT INTO documents (title, source, mode, session_id) VALUES (%s, %s, %s, %s) RETURNING id",
        (doc_title, file.filename, "personal", session_id)
    )
    document_id = doc[0]["id"]

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        execute_query(
            conn,
            "INSERT INTO chunks (document_id, content, embedding, chunk_index) VALUES (%s, %s, %s, %s)",
            (document_id, chunk, embedding, i)
        )

    return {"message": f"Ingested {len(chunks)} chunks", "document_id": document_id}
