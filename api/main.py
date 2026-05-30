from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

from database import get_db, execute_query
from embeddings import embed_text
from chunker import chunk_text, clean_text

load_dotenv()

app = FastAPI(title="DocuQuery RAG API", version="1.0.0")

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

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(request: IngestRequest, conn=Depends(get_db)):
    cleaned = clean_text(request.content)
    chunks = chunk_text(cleaned)

    doc = execute_query(
        conn,
        "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id",
        (request.title, request.source)
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
def query(request: QueryRequest, conn=Depends(get_db)):
    question_embedding = embed_text(request.question)

    results = execute_query(
        conn,
        """
        SELECT c.content, d.title, 
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (question_embedding, question_embedding, request.top_k)
    )

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    context = "\n\n".join([r["content"] for r in results])

    prompt = f"""You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {request.question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [{"title": r["title"], "similarity": round(r["similarity"], 4)} for r in results]
    }

@app.get("/documents")
def list_documents(conn=Depends(get_db)):
    docs = execute_query(conn, "SELECT id, title, source, created_at FROM documents ORDER BY created_at DESC")
    return {"documents": docs}
