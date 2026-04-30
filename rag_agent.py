import os
from dotenv import load_dotenv
import chromadb

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# ---------------- STATE ---------------- #
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[str]
    answer: str
    grade: str
    retry_count: int


# ---------------- SETUP ---------------- #
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")


# ---------------- RETRIEVE ---------------- #
def retrieve(state: AgentState):
    question = state.get("rewritten_question") or state["question"]

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    docs = results["documents"][0] if results["documents"] else []

    return {**state, "documents": docs}


# ---------------- GENERATE ---------------- #
def generate(state: AgentState):
    docs = state["documents"]
    question = state["question"]

    if not docs:
        return {**state, "answer": "No relevant documents found."}

    context = "\n\n".join(docs)

    prompt = f"""
Answer ONLY from the documents below.

Documents:
{context}

Question:
{question}
"""

    response = llm.invoke([
        SystemMessage(content="Answer strictly from given docs."),
        HumanMessage(content=prompt)
    ])

    return {**state, "answer": response.content}


# ---------------- GRADE ---------------- #
def grade(state: AgentState):
    docs = "\n\n".join(state["documents"])
    answer = state["answer"]

    prompt = f"""
Is this answer supported by the documents?

Documents:
{docs}

Answer:
{answer}

Reply ONLY with: PASS or FAIL
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    grade = response.content.strip().upper()

    return {**state, "grade": grade}


# ---------------- REWRITE ---------------- #
def rewrite(state: AgentState):
    question = state["question"]

    prompt = f"Rewrite this question better: {question}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "rewritten_question": response.content,
        "retry_count": state["retry_count"] + 1
    }


# ---------------- ROUTER ---------------- #
def route(state: AgentState):
    if state["grade"] == "PASS":
        return "end"
    elif state["retry_count"] >= 2:
        return "give_up"
    else:
        return "rewrite"


def give_up(state: AgentState):
    return {**state, "answer": "I don't know."}


# ---------------- GRAPH ---------------- #
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("grade", grade)
graph.add_node("rewrite", rewrite)
graph.add_node("give_up", give_up)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "grade")

graph.add_conditional_edges(
    "grade",
    route,
    {
        "end": END,
        "rewrite": "rewrite",
        "give_up": "give_up"
    }
)

graph.add_edge("rewrite", "retrieve")
graph.add_edge("give_up", END)

app = graph.compile()


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    question = input("Ask: ")

    result = app.invoke({
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "answer": "",
        "grade": "",
        "retry_count": 0
    })

    print("\nFinal Answer:\n", result["answer"])