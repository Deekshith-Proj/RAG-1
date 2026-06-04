import os
import requests

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

from app.rag import process_pdf
from app.rag import retrieve

app = FastAPI(
    title="Enterprise RAG Assistant"
)


@app.get("/")
def root():

    return {
        "message":
        "Enterprise RAG running"
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    os.makedirs(
        "app/uploads",
        exist_ok=True
    )

    filepath = (
        f"app/uploads/{file.filename}"
    )

    with open(
        filepath,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    process_pdf(filepath)

    return {
        "message":
        f"{file.filename} uploaded and indexed"
    }


@app.post("/ask")
async def ask_question(
    question: str
):

    try:

        chunks = retrieve(
            question
        )

        context = "\n\n".join(
            chunk.page_content
            for chunk in chunks
        )

        prompt = f"""
You are an enterprise AI assistant.

Answer ONLY using the supplied context.

If the answer is not present,
say:

"I could not find that information."

Context:
{context}

Question:
{question}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        answer = response.json()[
            "response"
        ]

        sources = []

        for chunk in chunks:

            sources.append(
                {
                    "page":
                    chunk.metadata.get(
                        "page"
                    )
                }
            )

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        return {
            "error": str(e)
        }