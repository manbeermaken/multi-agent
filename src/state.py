from typing import Annotated, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None
    next_node: str | None

class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ..., 
        description="Classify whether the user wants to just chat, ask for knowledge, or change code in the project."
    )