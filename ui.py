""" Module for Streamlit based UI -- RAG based FIFA Chatbot"""

import streamlit as st
from app import RagChat


@st.cache_resource
def initialize():
    """ Function for initializing the RagChat class from ./app.py"""
    chat = RagChat()
    return chat

st.session_state["chat"] = initialize()
st.title("The FIFA Chatbot")
st.markdown(":blue[A RAG(Retrieval Augmented Generation) based application.]")


# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# init models
if "model" not in st.session_state:
    st.session_state["model"] = ""

# Display chat messages from response history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Please type in your query here..."):

    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner('Generating response..'):
            response = st.session_state["chat"].rag_chat_gen(question=prompt)

            message = response
            st.markdown(message)
      
    st.session_state["messages"].append({"role": "assistant", "content": message})

# end of file