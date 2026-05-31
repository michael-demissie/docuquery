from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import json
from typing import List, Optional
from dotenv import load_dotenv

from database import get_db, execute_query
from embeddings import embed_text
from chunker import chunk_text, clean_text

load_dotenv()

limiter = Limiter(key_func=get_remote_address)
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

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    history: Optional[List[Message]] = []
    mode: str = "personal"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(request: IngestRequest, conn=Depends(get_db)):
    cleaned = clean_text(request.content)
    chunks = chunk_text(cleaned)

    doc = execute_query(
        conn,
        "INSERT INTO documents (title, source, mode) VALUES (%s, %s, %s) RETURNING id",
        (request.title, request.source, request.mode)
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
        WHERE d.mode = %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (question_embedding, body.mode, question_embedding, body.top_k)
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
def list_documents(mode: str = "personal", conn=Depends(get_db)):
    docs = execute_query(conn, "SELECT id, title, source, created_at, mode FROM documents WHERE mode = %s ORDER BY created_at DESC", (mode,))
    return {"documents": docs}

@app.delete("/documents/{document_id}")
def delete_document(document_id: int, conn=Depends(get_db)):
    execute_query(conn, "DELETE FROM documents WHERE id = %s", (document_id,))
    return {"message": f"Document {document_id} deleted"}

from fastapi import UploadFile, File
import pypdf
import io

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_file(request: Request, file: UploadFile = File(...), title: str = "", conn=Depends(get_db)):
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
        "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id",
        (doc_title, file.filename)
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
