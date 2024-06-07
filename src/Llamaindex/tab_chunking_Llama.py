import streamlit as st
from pathlib import Path
import tiktoken

from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_parse import LlamaParse

from llama_index.core.node_parser import HTMLNodeParser
from llama_index.core.text_splitter import CodeSplitter
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

import os
import config
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
os.environ['GROQ_API_KEY'] = config.GROQ_API_KEY
os.environ['LLAMA_CLOUD_API_KEY'] = config.LLAMA_CLOUD_API_KEY

def tokens(string:str) -> int:
    """Return the number of token in a text string"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def check_var(nombre_var):
    if nombre_var not in st.session_state:
        st.session_state[nombre_var] = 0 
    return st.session_state[nombre_var]


def reader_document(uploaded_file):
    if uploaded_file[-4:] == 'html': 
        reader = FlatReader()
        docs = reader.load_data(Path(uploaded_file))
    elif uploaded_file[-3:] == "pdf":
        parser = LlamaParse(
            result_type = "markdown", # "markdown" and "text" are available
            max_timeout = 5000
        )
        docs = parser.load_data(uploaded_file) #["./my_file1.pdf", "./my_file2.pdf"]
    return docs 




def tab_chunk_Lla():
    st.header('Split the documents')
    uploaded_files = st.file_uploader("Load the documents, currently this demo supports PDF and HTML: ", accept_multiple_files = True, key = "documents_Llama")
    chunk_model = st.selectbox("Select chunking model:", ["Simple File Node Parser", "HTML Node Parser", "Code Splitter", "Sentence Splitter",
                                                         "Semantic Splitter Node Parser", "Token Text Slpitter", "Hierarchical Node Parser"], key = "splitter_Llama")
    
    if chunk_model in ["Sentence Splitter", "Token Text Slpitter"]:
        #tipe = 'SentenceSplitter' if chunk_model == "Sentence Splitter" else 'TokenTextSplitter' 
        chunk_size = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 100, key="chunk_size_Llama")
        overlap = st.number_input("Overlap:", min_value = 0, max_value = 90000, value = 0, key = "overlap_Llama")
    elif chunk_model == 'Code Splitter':
        #tipe = "CodeSplitter"
        chunk_size = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 100, key="chunk_size_Llama")
        overlap = st.number_input("Overlap:", min_value = 0, max_value = 90000, value = 0, key = "overlap_Llama")
        max_chars = st.number_input("Max chars:", min_value = 10, max_value = 100000, value = 100, key="max_chars_Llama")
        language = st.text_input("Select language: ", key = 'language_Llama')
    elif chunk_model in ["Semantic Splitter Node Parser"]:
      #tipe = 'SemanticSplitterNodeParser'
      embed_model = st.selectbox("Select embedding model:", ["text-embedding-3-large", "WhereIsAI/UAE-Large-V1", "thenlper/gte-base-en-v1.5",
                                                             "BAAI/bge-base-en-v1.5", "nomic-ai/nomic-embed-text-v1.5", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"], 
                                                             key = "embedding model chunking_Llama") 
      buffer_size = st.number_input("Buffer size:", min_value = 10, max_value = 100000, value = 100, key="buffer_Llama")
    elif chunk_model in ["Hierarchical Node Parser"]:
        #tipe = 'HierarchicalNodeParser'
        chunk_size = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 2048, key="chunk_size_Llama")
        chunk_size2 = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 512, key="chunk_size2_Llama")
        chunk_size3 = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 128, key="chunk_size3_Llama")
    else: #Simple File Node Parser and HTML Node Parser
        #tipe = 'HTMLNodeParser' if chunk_model == "HTML Node Parser" else 'SimpleFileNodeParser' 
        chunk_size = 0
        overlap = 0
        
    if st.button("Run", key = "chunking_test_Llama"):
        if uploaded_files is not None:
            chunks_gen = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join('..','Data', 'input', uploaded_file.name)
                docs = reader_document(file_path)

                if chunk_model == 'Simple File Node Parser': 
                    nodes = SimpleFileNodeParser().get_nodes_from_documents(docs)
                elif chunk_model == 'HTML Node Parser':
                    nodes = HTMLNodeParser(include_metadata = True, include_prev_next_rel = True).get_nodes_from_documents(docs)
                elif chunk_model == 'Code Splitter':
                    nodes  = CodeSplitter(language = language, chunk_lines = chunk_size, chunk_lines_overlap = overlap, max_chars = max_chars).get_nodes_from_documents(docs)
                elif chunk_model == 'Sentence Splitter': 
                    nodes = SentenceSplitter( chunk_size = chunk_size, chunk_overlap = overlap).get_nodes_from_documents(docs)
                elif chunk_model == 'Semantic Splitter Node Parser': 
                            if embed_model == 'text-embedding-3-large': 
                                embeddings = OpenAIEmbedding(model = 'text-embedding-3-large')
                            else: 
                                embeddings = HuggingFaceEmbedding(model_name = embed_model)
                            nodes = SemanticSplitterNodeParser(buffer_size = buffer_size, breakpoint_percentile_threshold = 95, embed_model = embeddings).get_nodes_from_documents(docs)
                elif chunk_model == 'Token Text Slpitter': 
                    nodes = TokenTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap, separator = " ").get_nodes_from_documents(docs)
                else: ## chunk_model == Hierarchical Node Parser
                    nodes = HierarchicalNodeParser.from_defaults(chunk_sizes = [chunk_size, chunk_size2, chunk_size3]).get_nodes_from_documents(docs)
                    leaf_nodes = get_leaf_nodes(nodes)
                    st.session_state['leaf_nodes'] = leaf_nodes
    
                st.header(f"Document: {uploaded_file.name[:-5]}", divider = 'grey')
                for chunk in nodes:
                    empty = 0 ## Counter for empty chunks in a document 
                    if chunk.text == '':
                        empty =+ 1 
                    else:  
                        chunks_gen.append(chunk)
                        st.markdown(chunk.text)
                        st.write(f"**Tokens:** {tokens(chunk.text)}")
                        st.write("---")
                st.write(f"**{empty} empty chunks have been removed from this document.**")
            st.session_state['chunks_gen'] = chunks_gen
        st.session_state['chunk_model'] = chunk_model

