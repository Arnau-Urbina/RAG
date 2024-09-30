import streamlit as st
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex, SimpleKeywordTableIndex, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from typing import List 

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever

from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank 
from llama_index.core.postprocessor import LongContextReorder
from llama_index.postprocessor.colbert_rerank import ColbertRerank 

from llama_index.core.node_parser import SentenceSplitter
### Other Pages Functions 
from Llamaindex.tab_chunking_Llama import tokens

def load_llm(model_name):
  if model_name[:3] == 'gpt': 
    llm = AzureOpenAI(model = model_name)
  else: 
    llm = Groq(temperature = 0.7, model_name = model_name)
  return llm

def embeddings(name_model):
  if name_model == 'text-embedding-3-large':
      model = OpenAIEmbedding(model = 'text-embedding-3-large')
  else: 
      model = HuggingFaceEmbedding(name_model)
  return model 


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
    
def SimpleHybridSearch(nodes, embedding, k = 2):
  storage_context = StorageContext.from_defaults()
  storage_context.docstore.add_documents(nodes)
  #service_context = ServiceContext.from_defaults(llm = llm, embed_model = "local:BAAI/bge-base-en-v1.5")
  vector_index = VectorStoreIndex(nodes, storage_context = storage_context, embed_model = embedding)
  keyword_index = SimpleKeywordTableIndex(nodes, storage_context = storage_context)

  vector_retriever = VectorIndexRetriever(index = vector_index, similarity_top_k = k)
  keyword_retriever = KeywordTableSimpleRetriever(index = keyword_index)
  custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

  return custom_retriever


def SimpleFusionRetriever(nodes, LLM, embedding, k = 2, nq = 4): 
  storage_context = StorageContext.from_defaults()
  storage_context.docstore.add_documents(nodes)

  vector_index = VectorStoreIndex(nodes, storage_context = storage_context, embed_model = embedding)

  retriever = QueryFusionRetriever(
    [vector_index.as_retriever()],
    similarity_top_k = k,
    num_queries = nq,  # set this to 1 to disable query generation
    use_async = True,
    verbose = True,
    llm = LLM
    # query_gen_prompt="...",  # we could override the query generation prompt here
  )
  return retriever


def ReciprocalRerankFusionRetriever(nodes, embedding, LLM, k = 2, nq = 4): 
  storage_context = StorageContext.from_defaults()
  storage_context.docstore.add_documents(nodes)

  index = VectorStoreIndex(nodes, storage_context = storage_context, embed_model = embedding)

  vector_retriever = index.as_retriever(similarity_top_k = k)

  bm25_retriever = BM25Retriever.from_defaults(
      docstore = index.docstore, similarity_top_k = k
  )
  
  retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k = k,
    num_queries = nq,  # set this to 1 to disable query generation
    mode = "reciprocal_rerank",
    use_async = True,
    verbose = True, 
    llm = LLM
    # query_gen_prompt="...",  # we could override the query generation prompt here
  )
  return retriever
     

def AutoMergingRetriever_func(nodes_AMR, leaf_nodes, embedding, k = 2): #ONLY WE USE WITH HIERARCHICAL NODE PARSER ??? 
  #leaf_nodes = get_leaf_nodes(nodes_AMR)

  # insert nodes into docstore
  docstore = SimpleDocumentStore() 
  docstore.add_documents(nodes_AMR)

  # define storage context (will include vector store by default too)
  storage_context = StorageContext.from_defaults(docstore = docstore) 

  base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context = storage_context,
    embed_model = embedding
  )

  base_retriever = base_index.as_retriever(similarity_top_k = k)
  retriever = AutoMergingRetriever(base_retriever, storage_context, ) # verbose = True

  return retriever


def RecursiveRetrieverNodeReferences(documents, embedding = "local:BAAI/bge-small-en", k = 2): #### UPDATE ??
  base_nodes = SentenceSplitter(chunk_size = 1024).get_nodes_from_documents(documents)

  for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"

  #RECURSIVE RETRIEVER + NODE REFERENCES 
  embed_model = resolve_embed_model(embedding)

  sub_chunk_sizes = [128, 256, 512]
  sub_node_parsers = [
      SentenceSplitter(chunk_size = c, chunk_overlap = 20) for c in sub_chunk_sizes ### PROVE WITH DIFFERENTE PARSER FUNCTION ??
  ]

  all_nodes = []
  for base_node in base_nodes:
      for n in sub_node_parsers:
          sub_nodes = n.get_nodes_from_documents([base_node])
          sub_inodes = [
              IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
          ]
          all_nodes.extend(sub_inodes)

      # also add original node to node
      original_node = IndexNode.from_text_node(base_node, base_node.node_id)
      all_nodes.append(original_node)
  all_nodes_dict = {n.node_id: n for n in all_nodes}
  vector_index_chunk = VectorStoreIndex(all_nodes, embed_model = embed_model)
  vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k = k)

  retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict = {"vector": vector_retriever_chunk},
    node_dict = all_nodes_dict,
    verbose = True,
  )
  return retriever_chunk
    

def print_add_context(contexts, retriever, questions, ground_truths, document, k, reranker_model, embed_model, LLM = Groq(temperature = 0.7, model = "Llama3-70b-8192")):
  for i in range(len(questions)):
    retrieval_chunk = retriever.retrieve(questions[i])
    retrieval_chunk = reranker(reranker_model, k, retrieval_chunk, LLM = LLM, embed_model = embed_model)
    st.write(f"**Query:** {questions[i]}")
    st.write(f"**Ground Truth:** {ground_truths[i]}")
    st.write(f"**Document Ground Truth:** {document[i]}")
    for i in range(k):
      st.write(f"**Retrieval {i + 1}: {retrieval_chunk[i].text}")
      st.write(f"**Score:** {retrieval_chunk[i].score}")
      st.write(f"**Retrieval {i + 1} tokens:** {tokens(retrieval_chunk[i].text)}")
      st.write(f"**Document Retrieval {i + 1}:** {retrieval_chunk[i].metadata['filename'][:-5]}")
    st.write("---")
  contexts.append(retrieval_chunk[0].text)

 

def reranker(model, k, chunks, LLM, embed_model = "local:BAAI/bge-base-en-v1.5"):
  if k == 1:
    return chunks
  else: 
    if model == 'LongContextReorder':
      postprocessor = LongContextReorder()
    elif model == 'LLMRerank':
      service_context = ServiceContext.from_defaults(llm = LLM, embed_model = embed_model)
      postprocessor = LLMRerank(top_n = k, service_context = service_context)
    elif model == 'JinaRerank':  
      postprocessor = JinaRerank(top_n = k, model = "jina-reranker-v1-base-en", api_key = "jina_eba96965c69b4fd894f57138b933d4195yirZCsUcmqh6-7DeqSi1Uu9DaHg")
    else:
      postprocessor = ColbertRerank(top_n = 5, model = "colbert-ir/colbertv2.0", tokenizer = "colbert-ir/colbertv2.0", keep_retrieval_score = True)

  return postprocessor.postprocess_nodes(chunks)
#https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/ColbertRerank/


def tab_retrievers_Lla():
  st.header('Retrievals')
  st.write('On this page we choose the retrieval model we want to use.')
  st.write("NOTE: Use the Auto Merge Retriever only if you use the Hierarchical Node Parser for chunking.")
  retrieval_model = st.selectbox("Select retrieval model:", ["Simple Hybrid Search","Simple Fusion Retriever",
                                                              "Reciprocal Rerank Fusion Retriever", 'Auto Merging Retriever',
                                                              "RecursiveRetrieverNode"], key = "retrieval Llama")
  st.write("Select the number of relevants contexts you want to obtain.")
  k = st.number_input("K: ", min_value = 1, max_value = 20, value = 1, key = 'k_Llama')
  rerank_model = st.selectbox("Select Rerank method", ["LongContextReorder", "JinaRerank", "LLMRerank", "ColbertRerank"], key = 'reranker_Llama')
  llm_model_lla = st.selectbox("Select the llm model: ", ["Llama3-70b-8192", "Llama-3.1-70b-Versatile", "Llama-3.2-90b-Text-Preview", "gpt-35-turbo-1106", "gpt-4o-mini-2024-07-18"], key = 'llm_llama')
  st.session_state['llm_model_lla'] = llm_model_lla
  llm = load_llm(llm_model_lla)

  if st.button("Run", key = "contexts_Llama"):
    questions = st.session_state['editable_questions'] 
    ground_truths = st.session_state['editable_ground_truths'] 
    document = st.session_state['document']
    chunks_gen = st.session_state['chunks_gen']
    embed_model = st.session_state['embed_model']
    chunk_model = st.session_state['chunk_model']
    if chunk_model == 'Hierarchical Node Parser':
      leaf_nodes = st.session_state['leaf_nodes']
    embedding = embeddings(embed_model)

    contexts = []
    if retrieval_model == 'Simple Hybrid Search':
      retriever = SimpleHybridSearch(chunks_gen, embedding, k)
      print_add_context(contexts, retriever, questions, ground_truths, document, k, rerank_model, embed_model, llm)
    elif retrieval_model == 'Simple Fusion Retriever': 
      retriever = SimpleFusionRetriever(chunks_gen, llm, embedding, k)
      print_add_context(contexts, retriever, questions, ground_truths, document, k, rerank_model, embed_model, llm)
    elif retrieval_model == 'Reciprocal Rerank Fusion Retriever':
      retriever = ReciprocalRerankFusionRetriever(chunks_gen, embedding, llm, k)
      print_add_context(contexts, retriever, questions, ground_truths, document, k, rerank_model, embed_model, llm)
    elif retrieval_model == 'Auto Merging Retriever':
      retriever = AutoMergingRetriever_func(chunks_gen, leaf_nodes , embedding, k)
      print_add_context(contexts, retriever, questions, ground_truths, document, k, rerank_model, embed_model, llm)
    else: #retrieval_model == 'RecursiveRetrieverNode'
      retriever = RecursiveRetrieverNodeReferences(document, embed_model, k)
      print_add_context(contexts, retriever, questions, ground_truths, document, k, rerank_model, embed_model, llm)

    st.session_state['retriever_Llama'] = retriever
    st.session_state['context_Llama'] = contexts
    st.session_state['retrieval_model'] = retrieval_model
    st.session_state['llm_model_lla'] = llm_model_lla

