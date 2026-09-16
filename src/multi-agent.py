import os
import subprocess
import uuid
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from state import State,IntentClassifier

load_dotenv()

llm = init_chat_model(model="google_genai:gemini-3.5-flash-lite")

knowledge = [
    "LangChain is a toolkit for building LLM applications.",
    "LangGraph is meant for agent orchestration and durable processes."
]
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=[Document(page_content=text) for text in knowledge])

def classify_intent(state: State):
    """Classifies the user's intent to route to the correct agent."""
    structured_llm = llm.with_structured_output(IntentClassifier)
    user_content = state["messages"][-1].content
    
    messages = [
        {"role": "system", "content": "Determine if the user wants to chat, retrieve knowledge, or change code."},
        {"role": "user", "content": user_content}
    ]
    result = structured_llm.invoke(messages)
    return {"message_intent": result.message_intent}

def prompt_llm_chat(state: State):
    """Handles general conversational chat."""
    messages = [{"role": "system", "content": "You are a talkative chatbot for fun. Be nice."}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def prompt_llm_rag(state: State):
    """Handles knowledge retrieval using the Vector Store."""
    query = state["messages"][-1].content
    documents = vector_store.similarity_search(query, k=3)
    
    context = "\n".join([f"- {doc.page_content}" for doc in documents])
    system_prompt = f"You are a RAG agent. Answer using only the context below. If not found, say you don't know.\nContext:\n{context}"
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def prepare_coding_request(state: State):
    """Rewrites the user prompt into explicit instructions for Claude Code."""
    messages = [
        {"role": "system", "content": "Rewrite the latest user coding request into a clear instruction for Claude Code. Use the conversation history as context. Only output the instruction, no explanation."}
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def accept_coding(state: State):
    """Human-in-the-loop interruption block for code execution."""
    user_prompt = state["messages"][-1].content
    
    decision = interrupt(f"About to run Claude Code with request: '{user_prompt}'. Approve (yes/no) or type a revised request:")
    text = str(decision).strip().lower()
    
    if text in ["y", "yes", "approve", "ok"]:
        return {"next_node": "coding_agent"}
    elif text in ["n", "no", "deny", "cancel"]:
        return {
            "messages": [{"role": "assistant", "content": "Coding request was denied by the user."}],
            "next_node": "denied"
        }
    else:
        return {
            "messages": [{"role": "user", "content": text}],
            "next_node": "prepare_coding_request"
        }

def prompt_llm_code(state: State):
    """Executes Claude Code in a subprocess within a local workspace."""
    user_prompt = state["messages"][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
    os.makedirs(workspace, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["claude", "-p", user_prompt, "--permission-mode", "accept-edits"],
            cwd=workspace,
            capture_output=True,
            text=True
        )
        output = result.stdout.strip() if result.stdout else result.stderr
    except FileNotFoundError:
        output = "Error: Claude CLI not found on this system. (Make sure Claude Code is installed)"
        
    return {"messages": [{"role": "assistant", "content": output}]}

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("prepare_coding_request", prepare_coding_request)
graph_builder.add_node("accept_coding", accept_coding)
graph_builder.add_node("coding_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")

graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state.get("message_intent"),
    {
        "chat": "chat_agent",
        "knowledge": "rag_agent",
        "code": "prepare_coding_request"
    }
)

graph_builder.add_edge("prepare_coding_request", "accept_coding")

graph_builder.add_conditional_edges(
    "accept_coding",
    lambda state: state.get("next_node"),
    {
        "denied": END,
        "coding_agent": "coding_agent",
        "prepare_coding_request": "prepare_coding_request"
    }
)


graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("coding_agent", END)

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("LangGraph Multi-Agent Workflow Started (type 'quit' to exit)")
    
    while True:
        try:
            user_msg = input("\nEnter message: ")
            if user_msg.lower() in ['quit', 'exit']:
                break
                
            state_input = {"messages": [{"role": "user", "content": user_msg}]}
            result = graph.invoke(state_input, config=config)
            
            while "__interrupt__" in result:
                prompt_text = result["__interrupt__"][0].value
                decision = input(f"\n[INTERRUPT] {prompt_text}\n> ")
                
                result = graph.invoke(Command(resume=decision), config=config)
                
            print("\nAssistant:", result["messages"][-1].content)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
