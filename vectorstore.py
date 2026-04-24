from langchain_community.vectorstores import Chroma
import os

DB_DIR = "./chroma_db"

def store_in_chroma(chunks, embeddings):
    if os.path.exists(DB_DIR):
        print("Loading existing DB...")
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        print("Creating new DB...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )

    db.persist()
    print("DB updated and saved.")

    return db