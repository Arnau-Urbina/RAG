import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from tab_chunking import tokens

import os 
import config
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

def embedding(model_name): 
  encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
  if model_name == "text-embedding-3-large":
     return OpenAIEmbeddings(model = model_name)
  return HuggingFaceBgeEmbeddings(model_name, encode_kwargs = encode_kwargs)


def retriever(chunks, model, embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large'), k = 1):
  vector = FAISS.from_documents(chunks, embeddings) #VectorStore 
  if model == 'Similarity Search':
    return vector.as_retriever(search_kwargs = {"k": k})
  elif model == 'MMR': 
    return vector.as_retriever(search_type="mmr", search_kwargs = {"k": k, "fetch_k": 10})
  else: #model == "Hybrid Search":
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    faiss_retriever = vector.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])#initialize the ensemble retriever



def tab_retrievers():
    st.header('Retrievals')
    st.write('On this page we choose the retrieval model we want to use.')
    retrieval_model = st.selectbox("Select retrieval model:", ["Similarity Search","MMR", "Hybrid Search"], 
    key = "retrieval")

    k = st.number_input("K: ", min_value = 1, max_value = 20, value = 1, key = 'k')

    if st.button("Run", key = "contexts"):
        questions = st.session_state['editable_questions'] 
        ground_truths = st.session_state['editable_ground_truths'] 
        document = st.session_state['document']
        chunks_gen = st.session_state['chunks_gen']
        embed_model = st.session_state['embed_model']

        embeddings = embedding(embed_model)

        contexts = []
        for i in range(len(questions)):
            retrieved_chunks = retriever(chunks = chunks_gen, model = retrieval_model, embeddings = embeddings).invoke(questions[i])
            st.write(f"**Query:** {questions[i]}")
            st.write(f"**Ground Truth:** {ground_truths[i]}")
            st.write(f"**Document Ground Truth:** {document[i]}")
            st.markdown(f"**Retrieval:** {retrieved_chunks[0].page_content}")
            st.write(f"**Retrieval tokens:** {tokens(retrieved_chunks[0].page_content)}")
            st.write(f"**Document Retrieval:** {retrieved_chunks[0].metadata['title'][:-5]}")
            st.write("---")
            contexts.append(retrieved_chunks[0].page_content)

        st.session_state['context'] = contexts
        st.session_state['retrieval_model'] = retrieval_model
