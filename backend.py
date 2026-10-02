from fastapi import FastAPI, UploadFile, File, HTTPException
import aiofiles
import os
import uuid
from pathlib import Path

from main import uploaded_file, vector_store, RRF, rewrite_query, prompt, llm

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

sessions = {}

MAX_FILE_SIZE = 20 * 1024 * 1024

@app.post("/upload")
async def upload_file(file: UploadFile = File(...),session_id: str = "default_user"):
    file_path = None
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid file name."
            )

        extension = Path(file.filename).suffix.lower()

        if extension not in [".pdf", ".txt"]:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported."
            )

        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 20 MB."
            )

        unique_filename = f"{uuid.uuid4().hex}{extension}"
        file_path = UPLOAD_DIR / unique_filename

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        documents = uploaded_file(str(file_path))

        sim, BM, namespace = vector_store(documents)

        sessions[session_id] = {
            "sim": sim,
            "BM": BM,
            "namespace": namespace,
            "filename": file.filename
        }

        return {
            "filename": file.filename,
            "message": "File processed and indexed successfully."
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/ask_question")
def ask_question(query: str, session_id: str = "default_user"):
    try:
        query = query.strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        session_data = sessions.get(session_id)

        if not session_data:
            raise HTTPException(
                status_code=400,
                detail="Please upload and process a file first."
            )

        rewritten_query = rewrite_query(query)

        reranking = RRF(
            session_data["sim"],
            session_data["BM"],
            rewritten_query
        )

        context = "\n\n".join(doc.page_content for doc in reranking)

        chain = prompt | llm

        result = chain.invoke({
            "context": context,
            "question": query
        })

        return {
            "question": query,
            "rewritten_query": rewritten_query,
            "answer": result.content
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

