""" Module providing RAG """

import ollama
import streamlit as st
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer, util
embeddings = OllamaEmbeddings(model="mistral")


class RagChat():
    """ The RAG (Retrieval Augmented Generation) class. """

    def __init__(self):
        """ Function initializing the class RagChat. """
        self.chat_history = []

    def format_docs(self,docs):
        """ Function for formatting documents. """
        return "\n\n".join(doc.page_content for doc in docs)

    def load_vector_database(self,):
        """ Function for loading loacal vectorDB and creating retriever. """
        vector_store = Chroma(persist_directory = "./", embedding_function = embeddings)
        retriever = vector_store.as_retriever()
        return retriever

    def ollama_llm(self, question, context):
        """ Function for generating response using ollama (model = mistral). """
        formatted_prompt = f"""Question: {question}\n\n You are a FIFA chatbot. Use the context to
                            answer the question, if context is not enough, use your best knowldege
                            to answer the question\n\n Context: {context}"""
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content':formatted_prompt}])
        return response['message']['content']


    def list_of_retrieved_docs(self, retrieved_docs):
        """ Function for returning an array of retrieved documents. """
        dict_response = {}
        for i, cur_response in enumerate(retrieved_docs):
            if i not in dict_response:
                dict_response[i] = cur_response.page_content.replace("\n", "")
        list_retrieved_docs = list(dict_response.values())
        return list_retrieved_docs

    def valid_docs(self, similarity_scores, max_similarity):
        """ Function for returning an array of indices (of valid documents). """
        valid_doc_index = []
        for i, cur_similarity_score in enumerate(similarity_scores):
            if cur_similarity_score >= 0.6*max_similarity:
                valid_doc_index.append(i)
        return valid_doc_index

    def semantic_search_on_retrieved_docs(self, prompt, list_retrieved_docs, retrieved_docs):
        """ Function for performing semantic search on retrieved documents. """
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        query_embedding = model.encode(prompt)
        retrieved_docs_embedding = model.encode(list_retrieved_docs) 
        similarity_scores = util.dot_score(query_embedding, retrieved_docs_embedding)[0].tolist()
        max_similarity = max(similarity_scores)
        valid_doc_index = self.valid_docs(similarity_scores, max_similarity)
        best_retrieved_docs = []
        for idx in valid_doc_index:
            best_retrieved_docs.append(retrieved_docs[idx])
        return best_retrieved_docs

    def rag_chat_gen(self, question):
        """ Function with the main RAG flow. """
        retriever=self.load_vector_database()
        retrieved_docs = retriever.invoke(question)
        list_retrieved_docs = self.list_of_retrieved_docs(retrieved_docs)
        best_retrieved_docs = self.semantic_search_on_retrieved_docs(question,list_retrieved_docs, retrieved_docs)
        formatted_context = self.format_docs(best_retrieved_docs)
        return self.ollama_llm(question, formatted_context)


    def clear_history(self,):
        """ Function for clearing chat history. """
        if 'history' in st.session_state:
            del st.session_state['history']


if __name__ == "__main__":
    chat = RagChat()
    print(chat.rag_chat_gen(question="Who won Fifa world cup 2010?"))

# end of file
