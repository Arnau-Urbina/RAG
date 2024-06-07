import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings #from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import torch

from langchain_community.vectorstores import FAISS 
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

### Other Pages Functions 
from Langchain.tab_chunking_LangChain import tokens

llm = ChatGroq(temperature = 0.7, model_name = "Llama3-70b-8192")

def embedding(name_model):
  if name_model == 'text-embedding-3-large': 
    model = OpenAIEmbeddings(model = 'text-embedding-3-large')
  else:
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    model = HuggingFaceEmbeddings(model_name = name_model, model_kwargs = model_kwargs) 
    #model = SentenceTransformer(name_model) ## == HuggingFaceBgeEmbeddings
    return model
  

def retrievers(chunks, model, embeddings = None, k = 2): #HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-small-en-v1.5', encode_kwargs = {'normalize_embeddings': True})
  vector = FAISS.from_documents(chunks, embeddings) #VectorStore 
  if model == 'Similarity Search':
    return vector.as_retriever(search_kwargs = {"k": k})
  elif model == 'MMR': 
    return vector.as_retriever(search_type="mmr", search_kwargs = {"k": k, "fetch_k": 10})
  else: #model == "Hybrid Search":
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    faiss_retriever = vector.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers = [bm25_retriever, faiss_retriever], weights = [0.7, 0.3])

def ParentDocument(documents, embeddings, k = 1):
  """
  The ParentDocumentRetriever strikes that balance by splitting and storing small chunks of data. 
  During retrieval, it first fetches the small chunks but then looks up the parent ids for those chunks and returns those larger documents.
  """

  parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000)
  child_splitter = RecursiveCharacterTextSplitter(chunk_size = 400)
  vector = FAISS.from_texts([doc[0].page_content for doc in documents], embeddings) #[doc.page_content for doc in documents]
  store = InMemoryStore()

  retriever = ParentDocumentRetriever(
    vectorstore = vector,
    docstore = store, 
    child_splitter = child_splitter,
    parent_splitter = parent_splitter,
    search_kwargs = {'k': k}
  )
  for doc in documents:
    retriever.add_documents(doc, ids = None)

  return retriever 

def MultiQuery(chunks, embeddings):
  """
  The MultiQueryRetriever automates the process of prompt tuning by using an LLM to generate multiple queries from different perspectives for a given user input query.
  For each query, it retrieves a set of relevant documents and takes the unique union across all queries to get a larger set of potentially relevant documents.
  """

  vector = FAISS.from_documents(chunks, embeddings)
  retriever = MultiQueryRetriever.from_llm(
    retriever = vector.as_retriever(), llm = llm
  )
  return retriever

def reranker(model, chunks, k = 2, retriever = None, query = None):
  if k == 1:
    return chunks
  else: 
    if model == 'LongContextReorder': 
      retrieved = LongContextReorder().transform_documents(chunks)
    elif model == 'JinaRerank':
      compressor = JinaRerank(top_n = k)
      retrieved = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = retriever).invoke(query)
    else: #model == 'FlashrankRerank'
      compressor = FlashrankRerank(model = 'ms-marco-MiniLM-L-12-v2', top_n = k)
      retrieved = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = retriever).invoke(query)

  return retrieved

def printer(k, query, ground_truth, document_gt, retrieved, contexts):
  if k == 1: 
    st.write(f"**Query:** {query}")
    st.write(f"**Ground Truth:** {ground_truth}")
    st.write(f"**Document Ground Truth:** {document_gt}")
    st.markdown(f"**Retrieval:** {retrieved[0].page_content}")
    st.write(f"**Retrieval tokens:** {tokens(retrieved[0].page_content)}")
    st.write(f"**Document Retrieval:** {retrieved[0].metadata['title'][:-5]}")
    st.write("---")
    contexts.append(retrieved[0].page_content)
  else: 
    st.write(f"**Query:** {query}")
    st.write(f"**Ground Truth:** {ground_truth}")
    st.write(f"**Document Ground Truth:** {document_gt}")
    for i in range(k): 
      st.markdown(f"**Retrieval {i}:** {retrieved[i].page_content}")
      st.write(f"**Retrieval {i} tokens:** {tokens(retrieved[i].page_content)}")
      st.write(f"**Document Retrieval {i}:** {retrieved[i].metadata['title'][:-5]}")
    st.write("---")
    contexts.append([i.page_content for i in retrieved])




def tab_retrievers():
  st.header('Retrievals')
  st.write('On this page we choose the retrieval model we want to use.')
  retrieval_model = st.selectbox("Select Retrieval model:", ["Similarity Search","MMR", "Hybrid Search", "Parent Document", "Multi-Query"], 
  key = "retrieval_Lang")
  k = st.number_input("K: ", min_value = 1, max_value = 20, value = 2, key = 'k_Lang')
  rerank_model = st.selectbox("Select Rerank method", ["LongContextReorder", "JinaRerank", "FlashrankRerank"], key = 'reranker_Lang')

  if st.button("Run", key = "contexts_Lang"):
    questions = st.session_state['editable_questions'] 
    ground_truths = st.session_state['editable_ground_truths'] 
    documents = st.session_state['documents'] #The page_content of the differents documents (all the raw text)
    document = st.session_state['document'] #Documents containning the questions 
    chunks_gen = st.session_state['chunks_gen']
    embed_model = st.session_state['embed_model']

    embeddings = embedding(embed_model)
    contexts = []
    for i in range(len(questions)):
      if retrieval_model in ["Similarity Search","MMR", "Hybrid Search"]:
        retriever = retrievers(chunks = chunks_gen, model = retrieval_model, embeddings = embeddings, k = k)
        retrieved_chunks = retriever.invoke(questions[i])
        retrieved_chunks = reranker(rerank_model, retrieved_chunks,  k, retriever = retriever, query = questions[i])
      elif retrieval_model == "Parent Document": 
        retriever = ParentDocument(documents, embeddings, k = k)
        retrieved_chunks = retriever.invoke(questions[i]) #vectorstore.similarity_search(query) --> retrieve the small chunk
        retrieved_chunks = reranker(rerank_model, retrieved_chunks, k, retriever = retriever, query = questions[i])
      else: #retrieval_model == 'Multi-Query' 
        retriever = MultiQuery(chunks_gen, embeddings, k = k)
        retrieved_chunks = retriever.get_relevant_documents(questions[i])
        retrieved_chunks = reranker(rerank_model, retrieved_chunks, k, retriever = retriever, query = questions[i])

      printer(k, questions[i], ground_truths[i], document[i], retrieved_chunks, contexts)

    st.session_state['context'] = contexts
    st.session_state['retrieval_model'] = retrieval_model
    st.session_state['k'] = k
    st.session_state['rerank_model'] = rerank_model

