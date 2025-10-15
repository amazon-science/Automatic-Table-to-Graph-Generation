import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from utils.config import LLM_MODELS

# Load environment variables
load_dotenv()

# Set page title and layout
st.set_page_config(
    page_title="AutoG Web App",
    layout="wide"
)

# Load model configuration
model_config = LLM_MODELS

# Create a sidebar
with st.sidebar:
    # Add a title to the sidebar
    st.title("AutoG Web App")
    
    # Add a selectbox to choose the LLM model
    st.title("LLM Model Selection")
    model_select = st.selectbox(
        "Choose an LLM Model",
        list(model_config.keys())
    )
    # Get the selected model configuration
    selected_model_config = model_config[model_select]

    # Display the model description
    st.write(selected_model_config["description"])

    # Add a file uploader to upload input files
    uploaded_files = st.file_uploader(
        "Upload Input Files",
        type=["csv"],
        accept_multiple_files=True
    )

# Create a main container
main_container = st.container()

# Create a sub-container for the input section
with main_container:
    # Add a title to the input section
    st.title("Input Section")
    
    # Add a text area to input text
    input_text = st.text_area(
        "Input Text",
        height=200
    )

# Create a sub-container for the output section
with main_container:
    # Add a title to the output section
    st.title("Output Section")
    
    # Add a button to run the LLM model
    run_button = st.button(
        "Run AutoG"
    )

    # Run the LLM model when the button is clicked
    if run_button:
    # Get the selected model configuration
        selected_model_config = model_config[model_select]

        # Create an LLM client instance
        # from services.llm_client import LLMClient
        # llm_client = LLMClient(model_select)

        # Invoke the LLM model
        # response = llm_client.invoke_model(input_text)

        # Display the response
        output_contents = f"You select model: {selected_model_config}"
        st.session_state.output_text = output_contents

    # Add a text area to display the output
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""
    output_text = st.text_area(
        "Output Text",
        value=st.session_state.output_text,
        height=200,
        disabled=True)

