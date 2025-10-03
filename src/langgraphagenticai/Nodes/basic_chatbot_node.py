from src.langgraphagenticai.State.state import State

class BasicChatbotNode:
    """
    This is basic chatbot logic implementation
    """

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """process the input and generate the response"""
        return {"message": self.llm.invoke(state["message"])}
    
