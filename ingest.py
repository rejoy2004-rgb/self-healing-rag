import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def load_and_split_documents(folder):
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    for file in Path(folder).glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs = loader.load()
        split_docs = splitter.split_documents(docs)
        chunks.extend(split_docs)

    return chunks


def ingest_to_chromadb(chunks):
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="knowledge_base"
    )

    documents = [c.page_content for c in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "local"} for _ in chunks]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Stored {len(documents)} chunks")


if __name__ == "__main__":
    chunks = load_and_split_documents("docs")
    ingest_to_chromadb(chunks)