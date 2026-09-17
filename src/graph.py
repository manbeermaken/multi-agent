from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Any
from state import State
from nodes import (
    classify_intent, prompt_llm_chat, prompt_llm_rag,
    prepare_coding_request, accept_coding, prompt_llm_code
)

def build_graph() -> CompiledStateGraph[Any, None, Any, Any]:
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
    return graph_builder.compile(checkpointer=checkpointer)