import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, List
from langchain_core.documents import Document

load_dotenv()

# FastAPI - create app first so health check works even if other services fail
app = FastAPI(title="Moxi-RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Lazy initialization
rag_graph = None
init_error = None

def init_rag():
    global rag_graph, init_error
    try:
        from qdrant_client import QdrantClient
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import StateGraph, START, END

        OPENAI_API_BASE = os.getenv("LLM_BASE_URL", "https://litellm.confer.today")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_base=OPENAI_API_BASE)

        vector_store = QdrantVectorStore(
            client=QdrantClient(url=os.getenv("QDRANT_URL"), port=443, api_key=os.getenv("QDRANT_API_KEY")),
            collection_name=os.getenv("QDRANT_COLLECTION", "moxi-website"),
            embedding=embeddings,
            content_payload_key="content",
        )

        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL_NAME", "gpt-4.1-nano"),
            openai_api_base=OPENAI_API_BASE,
            temperature=0,
        )

        SYSTEM_PROMPT = """You are Moxi, a friendly and knowledgeable AI assistant for Moxi Solutions.

Your Personality:
- Warm, professional, and genuinely helpful
- Clear and concise in explanations
- Proactive in offering relevant suggestions

Core Principles:
1. ACCURACY: Answer ONLY using the provided context. Never make up information.
2. CLARITY: Use simple language. Break complex topics into easy-to-understand points.
3. HELPFULNESS: Anticipate follow-up questions and provide actionable guidance.
4. HONESTY: If information isn't in the context, say: "I don't have that specific information available. Would you like me to help you connect with our team?"

Response Style:
- Keep responses focused and relevant
- Use bullet points for lists or multiple items
- Highlight key information naturally
- End with a helpful follow-up when appropriate

Context:
{context}"""

        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        class GraphState(TypedDict):
            query: str
            documents: List[Document]
            response: str

        def retrieve(state: GraphState) -> dict:
            return {"documents": retriever.invoke(state["query"])}

        def generate(state: GraphState) -> dict:
            context = "\n\n".join(doc.page_content for doc in state["documents"])
            result = (prompt | llm).invoke({"context": context, "input": state["query"]})
            return {"response": result.content}

        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        rag_graph = workflow.compile()
        init_error = None
    except Exception as e:
        init_error = str(e)
        print(f"RAG init error: {e}")

@app.on_event("startup")
async def startup():
    init_rag()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    if rag_graph is None:
        raise HTTPException(status_code=503, detail=init_error or "RAG not initialized")
    return {"status": "ready"}

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_graph is None:
        raise HTTPException(status_code=503, detail=init_error or "RAG not initialized")
    result = rag_graph.invoke({"query": request.question, "documents": [], "response": ""})
    return {"answer": result["response"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
