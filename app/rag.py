import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = []
index = None


def process_pdf(file_path: str):

    global documents
    global index

    loader = PyPDFLoader(file_path)

    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    if index is None:

        index = faiss.IndexFlatL2(
            dimension
        )

    index.add(
        embeddings.astype(np.float32)
    )

    documents.extend(chunks)

    print(
        f"Indexed {len(chunks)} chunks"
    )


def retrieve(
    query: str,
    k: int = 3
):

    global index

    if index is None:
        raise Exception(
            "No documents uploaded yet"
        )

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding.astype(np.float32),
        k
    )

    results = []

    for idx in indices[0]:

        results.append(
            documents[idx]
        )

    return results