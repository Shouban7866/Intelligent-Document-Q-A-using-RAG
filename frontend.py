import requests
import streamlit as st
import uuid

st.title("Intelligent Document Q&A using RAG")

BACKEND_URL = "http://127.0.0.1:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

file = st.sidebar.file_uploader(
    "Upload file",
    type=["pdf", "txt"]
)


if file:
    if st.sidebar.button("Process File"):
        with st.spinner("Processing and indexing document..."):
            files = {"file": (file.name,file.getvalue(),file.type)}

            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files,
                params={"session_id": st.session_state.session_id},
                timeout=300
            )
            if response.status_code == 200:
                st.sidebar.success("File processed successfully!")
                st.session_state.uploaded_filename = file.name

                st.session_state.messages = []

            else:
                st.error(response.text)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


query = st.chat_input("Enter your query:")

if query:
    if not st.session_state.uploaded_filename:
        st.error("Please upload and click 'Process File' first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):

        response = requests.post(
            f"{BACKEND_URL}/ask_question",
            params={
                "query": query,
                "session_id": st.session_state.session_id
            },
            timeout=120
        )

    if response.status_code == 200:
        result = response.json()
        answer = result["answer"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        with st.chat_message("assistant"):
            st.write(answer)

    else:

        st.error(response.text)