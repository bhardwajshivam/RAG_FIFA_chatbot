""" Module for loading vector database """
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
embeddings = OllamaEmbeddings(model="mistral")


links = ['https://en.wikipedia.org/wiki/2002_FIFA_World_Cup',
         'https://en.wikipedia.org/wiki/2006_FIFA_World_Cup',
         'https://en.wikipedia.org/wiki/2010_FIFA_World_Cup',
         'https://en.wikipedia.org/wiki/2014_FIFA_World_Cup',
         'https://en.wikipedia.org/wiki/2018_FIFA_World_Cup',
         'https://en.wikipedia.org/wiki/2022_FIFA_World_Cup',
         'https://en.wikipedia.org/wiki/2026_FIFA_World_Cup']


loader = WebBaseLoader(links)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model="mistral")
vector_store = Chroma.from_documents(documents = splits, embedding = embeddings, persist_directory="./vector_store")

#end of file