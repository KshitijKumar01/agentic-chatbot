import streamlit as st
from src.langgraphagenticai.UI.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.Graph.graph_builder import GraphBuilder
from src.langgraphagenticai.UI.streamlitui.display_result import DisplayResultStreamlit


def load_langgraph_agentic_app():
    """
    Loads and runs the LangGraph Agentinc application with Streamlit UI.
    This function initializes the UI, handels user input, configures the LLM models,
    sets up the graph based on the selected use case, and displays the output while implementing 
    exceeption handling for robustness.
    """

    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI")
        return
    
    user_message = st.chat_input("Enter your message: ")

    if(user_message):
        try:
            ## configure the llms
            obj_llm_config = GroqLLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_model()

            if not model:
                st.error("Error: Model could not be initialized")
                return
            
            # Initialize and set up the graph based on the usecase
            usecase = user_input.get("selected_usecase")
            if not usecase:
                st.error("Error: No usecase seleted")
                return
            
            #Grapb builder
            graph_builer = GraphBuilder(model)
            try:
                graph = graph_builer.setup_graph(usecase)
                print(user_message)
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Graph setup failed {e}")
                return

        except Exception as e:
            st.error(f"Error: Graph setup failed {e}")
            return
    