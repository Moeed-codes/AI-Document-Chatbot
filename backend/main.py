import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Document Chatbot API")

# Initialize RAG Pipeline
# Ensure GROQ_API_KEY is active in the environment
rag = RAGPipeline()

# Ensure temp data directory exists
os.makedirs("data/temp", exist_ok=True)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Document Chatbot API"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    file_path = f"data/temp/{file.filename}"
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process and index the document
        rag.process_file(file_path)
        
        return {"filename": file.filename, "message": "Document uploaded and indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Optionally cleanup the temp file if we don't need it after indexing
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        answer = rag.answer_question(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
