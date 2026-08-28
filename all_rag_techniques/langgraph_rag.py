import os
from typing import TypedDict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Load PDF

loader = PyMuPDFLoader("data/Sandeep.pdf")

documents = loader.load()

print(f"Loaded {len(documents)} pages")

# Split documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Huggingface embeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {
        "device": "cpu"
    },
    encode_kwargs = {
        "normalize_embeddings": True
    }
)

# FAISS VECTOR STORE

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k":4
    }
)

# Groq LLM

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2048,
    api_key=GROQ_API_KEY,
)

# LangGraph State

class RAGState(TypedDict):
    question: str
    documents: list
    answer: str

# Retrieve Node

def retrieve(state: RAGState):
    question = state['question']
    documents = retriever.invoke(question)

    return {
        "documents": documents
    }

# Generate Node

def generate(state: RAGState):

    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful AI study assistant.

Answer the question using only the provided context.

If the answer is not available in the context,
say that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# Build LangGraph

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_app = graph.compile()

# Ask Question

question = input("Ask a question: ")

result = rag_app.invoke({
    "question": question,
    "documents": [],
    "answer": "",
})

# Output
print("\nAnswer: ")
print(result['answer'])