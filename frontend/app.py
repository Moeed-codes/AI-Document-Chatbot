import streamlit as st
import requests

# Constants
API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Document Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 AI Document Chatbot")
st.markdown("Upload your PDF or TXT documents and ask questions based on their content.")

# View state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for uploading documents
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Uploading and indexing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success("Document processed successfully! You can now ask questions about it.")
                    else:
                        st.error(f"Error processing document: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the backend server. Make sure it is running.")
        else:
            st.warning("Please upload a file first.")
            
    st.divider()
    st.markdown("### How it works")
    st.markdown("1. Upload a document")
    st.markdown("2. The backend extracts text, splits it into chunks, and creates vector embeddings.")
    st.markdown("3. Ask questions in the main chat interface.")
    st.markdown("4. The most relevant chunks are retrieved and sent to an LLM to formulate an answer.")

# Main Chat Interface

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input block
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Generating response... _(Fetching from database)_")
        
        try:
            response = requests.post(f"{API_URL}/chat", json={"question": prompt})
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer received.")
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Error generating response: {response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Could not connect to the backend server. Make sure it is running."
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
