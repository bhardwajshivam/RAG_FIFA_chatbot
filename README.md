
# RAG-FIFA chatbot
Ollama, RAG, ChromaDB 

## Overview
This project implements a Retrieval Augmented Generation (RAG) based chatbot leveraging Ollama for large language models (LLMs) and ChromaDB as the vector database. It enables users to input Wikipedia links, creates a local vector database of the text scraped from these links, invokes a RAG chain, and sets up a chatbot for question and answer sessions on the relevant text.

## Concept of RAG 
The concept of a Retrieval-Augmented Generation (RAG) refers to a hybrid approach in natural language processing that combines retrieval-based and generation-based methods to enhance the quality and relevance of text output in tasks such as question answering, conversation, or document summarization. This approach addresses the limitations of purely generative models by dynamically integrating external information from a relevant database.

### How RAG Works:
1. **Retrieval Step**: Upon receiving a query (like a question or prompt), the RAG model first retrieves relevant documents or data snippets from a large corpus or database. This retrieval is conducted using a vector space model where both the query and the documents are encoded into embeddings. Techniques like approximate nearest neighbor search are employed to find the most relevant documents.
2. **Augmentation and Context Integration**: The retrieved documents are then used as additional context for the generation model. This context augmentation helps the generative model to be more informed and precise, allowing it to produce answers that are not only based on its pre-trained knowledge but also on specific information contained in the external documents.
3. **Generation Step**: Leveraging a language model, the RAG system generates a response by conditioning on both the initial query and the retrieved documents. The generation model synthesizes information from these sources to construct a coherent and contextually appropriate response.

### Key Benefits of RAG:
- **Enhanced Accuracy and Relevance**: By using external sources of information, RAG systems can provide responses that are more accurate and relevant to the specific context of the query.
- **Scalability**: Since the knowledge is not stored within the model but retrieved from an external database, RAG systems can scale their knowledge base simply by expanding the database without retraining the model.
- **Flexibility and Adaptability**: RAG can be adapted to various domains and applications by changing the database it retrieves from, making it highly versatile.


# RAG + Semantic Search 

## 1. Calling RAG

The `rag_chat_gen()` function is used to initiate the RAG (Retrieval-Augmented Generation) with semantic search capabilities. It requires two arguments:
- **Prompt**: The input text or query for which relevant information is sought.
- **Retriever**: The mechanism responsible for retrieving relevant documents from the VectorDB (Chroma).

Example usage:
```python
result = rag_chain_test_semantic(prompt, retriever)
```

## 2. Retrieval from VectorDB (Chroma)

The RAG retrieves the top documents from the VectorDB, known as Chroma. These documents are fetched based on their relevance to the provided prompt.

## 3. Semantic Search

After retrieving documents from the VectorDB, a semantic search is performed on these documents. This search aims to identify documents with similarity scores falling within a specified range. The range is defined as `[0.6*max_similarity_score, max_similarity_score]`, where the similarity scores are calculated between the retrieved documents and the prompt.

## 4. Formatting and Passing Context

Once relevant documents are identified, their context is formatted for further processing. This formatted context is then passed to the `ollama_llm` function.

## 5. Generating Response with Ollama LLM

The `ollama_llm` function takes two inputs:
- **Prompt**: The initial query or context.
- **Context**: The formatted context obtained from the previous step.

This function utilizes the Mistral LLM (Large Language Model) to generate a response relevant to the given prompt and context.

Example usage:
```python
response = ollama_llm(prompt, formatted_context)
```

## Code for RAG
```python

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
        vector_store = Chroma(persist_directory = "./vector_store", embedding_function = embeddings)
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

```

## User Guide

* Use python=3.11
* Use sqlite3 > 3.35.0
* To load links of your choice update links in load_data.py
* Install pytest to run unit tests: pip install -U pytest


* Install Ollama from: https://ollama.com
```
1. Clone this repository: git clone https://github.com/bhardwajshivam/RAG-chatbot.git

2. Create a virtual environment (e.g., venv): virtualenv -p python3 venv "python=3.11"

3. Activate your virtual environment: source venv/bin/activate

4. Install dependencies using: pip install -r requirements.txt 

5. Run the application with: streamlit run ui.py
```


