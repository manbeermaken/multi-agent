import uuid
from langgraph.types import Command
from graph import build_graph


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph = build_graph()    
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
