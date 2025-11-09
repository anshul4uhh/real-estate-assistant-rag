import os
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import streamlit as st

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    load_dotenv()



# constraints
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=500
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    """Scrape URLs → split text → store embeddings in Chroma DB"""

    yield "Initializing components…"
    initialize_components()

    # Reset vector store if collection exists
    yield "Reseting vector store…"
    try:
        vector_store.reset_collection()
        yield "Vector store cleared"
    except Exception as e:
        print("Could not reset collection:", e)

    yield "Loading data…"
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={"User-Agent": "Mozilla/5.0"},
        continue_on_failure=True,
        ssl_verify=False
    )
    data = loader.load()

    yield "Splitting text…"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)

    yield "Adding embeddings to vector DB…"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    yield "Data stored successfully in vector db!"



def answer_query(query, k=3):
    """
    Retrieve top-k relevant documents and generate an answer using LLM.
    This function is fully compatible with your existing code structure.
    """

    initialize_components()   # ensure llm + vector_store are loaded

    # Step 1: Retrieve relevant chunks
    docs = vector_store.similarity_search(query, k=k)

    # Step 2: Build RAG context
    context = "\n\n".join([d.page_content for d in docs])

    # Step 3: Build prompt
    template = """
    You are a knowledgeable assistant. Answer ONLY using the information from the context.
    If the information is not available, reply with "I don't know".

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    final_prompt = prompt.format(context=context, question=query)

    # Step 4: Call LLM
    result = llm.invoke(final_prompt)
    answer = result.content if hasattr(result, "content") else result

    # Step 5: Return results
    return {
        "answer": answer,
        "sources": [d.metadata.get("source") for d in docs]
    }



if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2025/11/07/rightmove-shares-plummet-as-ai-investments-to-hit-2026-profit-.html"
    ]

    process_urls(urls)
    answer = answer_query("The Federal Reserve lowered its interest rate target three times in which year?", k=2)

    print("\nAnswer of query is:",answer["answer"])


