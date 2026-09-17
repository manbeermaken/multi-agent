import os
import subprocess
from langgraph.types import interrupt
from state import State, IntentClassifier
from config import llm, vector_store


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