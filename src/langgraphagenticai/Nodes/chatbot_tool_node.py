from src.langgraphagenticai.State.state import State

class ChatbotWithToolNode:
    """
    Chatbot logic enhanced with tools
    """
    def __init__(self, model):
        self.llm = model
    
    def process(self, state : State)-> dict:
        """
        Processes the input state and generates a response with tool integration
        """

        user_input = state["messages"][-1] if state["messages"] else ""
        llm_response = self.llm.invoke([{"role": "user", "content": user_input}])

        tools_response = f"Tool integration for : '{user_input}'"
        return {"messages": [llm_response, tools_response]}
    
    def create_chatbot(self, tools):
        """
        Returns a chatbot node function integrated with tools
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            """
            Proper chatbot logic for LangGraph with tool support
            """
            
            return {"messages": [llm_with_tools.invoke(state['messages'])]}

        return chatbot_node