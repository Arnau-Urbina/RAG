import streamlit as st
import tiktoken

from langchain_community.document_loaders import BSHTMLLoader, PyMuPDFLoader #UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ai21 import AI21SemanticTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings #from langchain.embeddings import HuggingFaceEmbeddings

#from pdt_core.doc_transformers import model_to_parser, Parser
#from pdt_core import file_to_doc
#from typing import List
#from langchain_community.document_loaders import Blob
#from langchain_core.documents import Document

import os 
import config as config

#os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
#os.environ['AI21_API_KEY'] = config.AI21_API_KEY
#os.environ['GROQ_API_KEY'] = config.GROQ_API_KEY

def tokens(string:str) -> int:
  """Return the number of token in a text string"""
  encoding = tiktoken.get_encoding("cl100k_base")
  num_tokens = len(encoding.encode(string))
  return num_tokens
    
def data_loader(uploaded_file):
  if uploaded_file[-4:] == 'html': 
    loader = BSHTMLLoader(uploaded_file, open_encoding='utf-8')
    doc = loader.load()
  elif uploaded_file[-3:] == "pdf":
    loader = PyMuPDFLoader(uploaded_file)
    doc = loader.load()
  return doc 

def chunks(type, doc, size = 0, overlap = 0, embed = 'BAAI/bge-small-en-v1.5'):
  if type == 'CharacterTextSplitter': 
    chunks = CharacterTextSplitter(chunk_size = size, chunk_overlap = overlap).split_documents(doc)
  elif type == 'RecursiveCharacterTextSplitter':
    chunks = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap = overlap, length_function = len ).split_documents(doc)
  elif type == 'SemanticChunker':
    embeddings = HuggingFaceEmbeddings(model_name = embed, model_kwargs = {'device':'cpu'})
    Semantic_Chunker = SemanticChunker(embeddings, breakpoint_threshold_type = 'percentile')
    chunks = Semantic_Chunker.create_documents([doc[0].page_content])
  else: #SemanticTextSplitter
    semantic_text_splitter = AI21SemanticTextSplitter()
    chunks = semantic_text_splitter.split_text_to_documents(doc[0].page_content) 
  return chunks


def printer_chunks(chunk, chunks_gen): 
  chunks_gen.append(chunk)
  st.markdown(chunk.page_content)
  st.write(f"**Token**: {tokens(chunk.page_content)}")
  st.write(f"**Characters:** {len(chunk.page_content)} ")
  st.write('---')  

def tab_chunk():
  st.header('Split documents')
  uploaded_files = st.file_uploader("Load HTML documents", accept_multiple_files = True, key = "documents_Lang")
  chunk_model = st.selectbox("Select chunking model:", ["Character Splitter", "Recursive Text Splitter", 
                                                      "Semantic Chunker", "Semantic Text Splitter", "Header"], key = "splitter_Lang")
  
  if chunk_model in ["Character Splitter", "Recursive Text Splitter"]:
    tipe = 'RecursiveCharacterTextSplitter' if chunk_model == "Recursive Text Splitter" else 'CharacterTextSplitter'
    embed_model = 'BAAI/bge-small-en-v1.5'
    chunk_size = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 1200, key = "chunk_size_Lang")
    overlap = st.number_input("Overlap:", min_value = 0, max_value = 90000, value = 30, key = "overlap_Lang")
  elif chunk_model in ["Semantic Chunker", "Semantic Text Splitter"]:
    tipe = 'SemanticChunker' if chunk_model == "Semantic Chunker" else 'SemanticTextSplitter'
    chunk_size = 0
    overlap = 0
    if chunk_model == "Semantic Chunker": 
     embed_model = st.selectbox("Select embedding model:", ["text-embedding-3-large", "WhereIsAI/UAE-Large-V1", "thenlper/gte-base-en-v1.5", "BAAI/bge-base-en-v1.5", 
                                                             "nomic-ai/nomic-embed-text-v1.5", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"], 
                                                             key = "embedding model chunking_Lang")
    else: 
      embed_model = 'BAAI/bge-small-en-v1.5'
  else: #chunk_model == "Header": 
    tipe = 'Jordi'

  if st.button("Run", key = "chunking_test_Lang"):

    if uploaded_files is not None:
      chunks_gen = []
      documents = []

      for uploaded_file in uploaded_files:
        file_path = os.path.join('..','Data', 'input', uploaded_file.name)
        doc = data_loader(file_path)
        documents.append(doc)

        if tipe == "Jordi": 
          pass
          #splits = header(uploaded_file = file_path)
          #st.header(f"Document: {uploaded_file.name}", divider = 'grey')
          #for chunk in splits:  
            #printer_chunks(chunk, chunks_gen) 
        else: 
          splits = chunks(tipe, doc, chunk_size, overlap, embed = embed_model)
          for i in splits: i.metadata['title'] = uploaded_file.name
          st.header(f"Document: {uploaded_file.name}", divider = 'grey')
          for chunk in splits:
            empty = 0
            if chunk.page_content == '': 
              empty =+ 1
            else: 
              printer_chunks(chunk, chunks_gen)
          st.write(f"**{empty} empty chunks have been removed from this document.**")
    st.session_state['chunks_gen'] = chunks_gen
    st.session_state['chunk_model'] = chunk_model
    st.session_state['chunk_size'] = chunk_size
    st.session_state['overlap'] = overlap
    st.session_state['documents'] = documents




"""
def header(llm = ChatOpenAI(), uploaded_file = None):
  
  #This way of chunking uses functions from the pdt library. First, it transforms the tables to a List,
  #then it transform the HTML tags to markdown and finally it splits the text by HTML header. 
  
  header = { #Combination of three functions for separating documents by headers
    "name": "by header",
    "pipeline": [
    {
      "module": "pdt_core.doc_transformers.table_to_list",
      "name": "TableToListTransformer",
      "params": {}
    },
    {
      "module": "pdt_core.doc_transformers.htmlTagsToMarkdown",
      "name": "HtmlTagsToMarkdownTransformer",
      "params": {
        "tags": ["pre"]
      }
    },
    {
      "module": "pdt_core.doc_transformers.lc.htmlHeaderTextSplitter",
      "name": "HTMLHeaderTextSplitter",
      "params": {
        "skip_first_doc": True,
        "add_last_title": True,
        "headers_to_split_on": [
          "h1, Header 1",
          "h2, Header 2"
        ]
      }
    }
  ]}

  # Create the parser out of the JSON definition
  parser:Parser = model_to_parser(header)
  chunks: List[Document]

  with open(uploaded_file, encoding = "utf8") as f:
    # convert the file content into a Blob
    blob = Blob.from_data(
      data = f.read(),
      path = f.name,
      mime_type = 'text/html',
    )

  # convert the blob to a langchain Document
  doc: Document = file_to_doc(blob)

  # chunk the file using the pipeline definition
  chunks = parser.get_docs([doc], llm) 

  for c in chunks: ## Add to the metadata the title tag with the document name
    if c.metadata.get('title') is None: 
      c.metadata['title'] = c.metadata['file'][:-5]
    else: 
      pass

    return chunks
"""