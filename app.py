from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from embeddings import generate_embedding
from db_utils import connect_db, init_db, insert_embedding, fetch_all_embeddings
from vector_store import VectorStore
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize SQLite connection and create tables
db_conn = connect_db()
init_db(db_conn)

# Initialize Vector Store
dimension = 1536  # Change based on your embedding model
vector_store = VectorStore(dimension)

# Load existing embeddings from DB into the vector store
embeddings_data = fetch_all_embeddings(db_conn)
for document_id, content, embedding_blob in embeddings_data:
    import pickle
    embedding = pickle.loads(embedding_blob)
    vector_store.add(embedding, {"document_id": document_id, "content": content})

# Request models
class IngestRequest(BaseModel):
    document_id: str
    content: str

class QueryRequest(BaseModel):
    query: str

# API to ingest documents
@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    try:
        # Generate embedding for the document
        embedding = generate_embedding(request.content)

        # Store embedding and metadata in DB and vector store
        insert_embedding(db_conn, request.document_id, request.content, embedding)
        vector_store.add(embedding, {"document_id": request.document_id, "content": request.content})

        return {"message": "Document ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

# API to query similar documents
@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        # Generate query embedding
        query_embedding = generate_embedding(request.query)

        # Search for similar embeddings in the vector store
        results = vector_store.search(query_embedding, k=5)

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
