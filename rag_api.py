from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_system import RAGSystem
import uvicorn

app = FastAPI()
rag = RAGSystem()

class Query(BaseModel):
    question: str
    model: str = "llama2"

@app.post("/rag/query")
async def query_rag(query: Query):
    try:
        answer = rag.answer_question(query.question, query.model)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 