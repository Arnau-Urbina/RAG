import tiktoken
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

import config 
import os 
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
os.environ['AI21_API_KEY'] = config.AI21_API_KEY
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
##Jordi tool: 
from pdt_core.doc_transformers import model_to_parser, Parser
from typing import List
from langchain_community.document_loaders import Blob
from langchain_core.documents import Document
from pdt_core import file_to_doc

def tokens(string:str) -> int:
    """Return the number of token in a text string"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_tokens(documents): #.page_content
    i = 1
    for t in documents:
        print('Document', i, ":")
        print('Chunk size: ', len(t)) 
        print('Tokens: ', tokens(t))
        print('--------------------')
        i += 1

   
def chunks(type, uploaded_file, size = 0, overlap = 0, embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')):
    loader = BSHTMLLoader(uploaded_file, open_encoding='utf-8')
    doc = loader.load()
    if type == 'CharacterTextSplitter': 
        chunks = CharacterTextSplitter(chunk_size = size, chunk_overlap = overlap).split_documents(doc)
    elif type == 'RecursiveCharacterTextSplitter':
        chunks = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap = overlap, length_function = len ).split_documents(doc)
    elif type == 'SemanticChunker':
            Semantic_Chunker = SemanticChunker(embeddings, breakpoint_threshold_type = 'percentile')
            chunks = Semantic_Chunker.create_documents([doc[0].page_content])
    else: #SemanticTextSplitter
        semantic_text_splitter = AI21SemanticTextSplitter()
        chunks = semantic_text_splitter.split_text_to_documents(doc[0].page_content) 
    return chunks


def embed(name_model, text):
  if name_model == 'text-embedding-3-large': 
    model = OpenAIEmbeddings(model = 'text-embedding-3-large')
    embeddings = model.embed_documents(text)
  else: 
    model = SentenceTransformer(name_model)
    embeddings = model.encode(text, normalize_embeddings = True)
  return embeddings 

def model_embed(name_model):
  if name_model == 'text-embedding-3-large': 
    model = OpenAIEmbeddings(model = 'text-embedding-3-large')
  else: 
    model = SentenceTransformer(name_model)
  return model 


### To graph embeddings with UMAP
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


def retriever(chunks, query, model, embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large'), k = 1):
  vector = FAISS.from_documents(chunks, embeddings)
  if model == 'Similarity Search':
    retriever = vector.as_retriever(search_kwargs = {"k": k})
    ret = retriever.invoke(query)
  elif model == 'MMR': 
    retriever = vector.as_retriever(search_type="mmr", search_kwargs = {"k": k, "fetch_k": 10})
    ret = retriever.invoke(query)
  else: #model == "Hybrid Search":
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    faiss_retriever = vector.as_retriever(search_kwargs={"k": k})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])#initialize the ensemble retriever
    ret = ensemble_retriever.invoke(query)
  return ret

#############################################################################################################
#############################################################################################################

PROMPT_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. Try to provide a complete but concise answer.
Question: {question}
Context:{context}

Answer: 
"""
prompt_template_aux = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def chains(query, retriever, prompt_template = prompt_template_aux, llm = ChatOpenAI()):
    chain = (
        {
            'context': retriever, 
            'question': RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
            )
    res = chain.invoke(query)
    return res


### Dict to do the evaluation with ragas 
def data_dict(retriever, query, answer):
    result = retriever.invoke(query)
    
    data_samples = {'question': [query],
                    'answer': [answer],
                    'contexts': [[doc.page_content for doc  in result]]}
    rag_dict = Dataset.from_dict(data_samples)

    return rag_dict

### View metric evaluations 
def plot_metrics(metrics_dict, title = 'RAG Metrics'):
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize = (10, 6))  
    bars = plt.bar(names, values, color = sns.color_palette("viridis", len(names)))
    
    # Adding the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,  # x-position
                 height + 0.01,  # y-position
                 f'{height:.4f}',  # value
                 ha='center', va='bottom')
    
    plt.ylabel('Score')
    plt.title(title)
    plt.ylim(0, 1.2)  # Setting the y-axis limit to be from 0 to 1
    plt.show()

def plot_compare_rags(metrics_dict, tokens, ax, title='RAG Metrics'):
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    bars = ax.bar(names, values, color=sns.color_palette("viridis", len(names)))
    # Adding the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, # x-position
                height + 0.01, # y-position
                f'{height:.4f}', # value
                ha='center', va='bottom')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_ylim(0, 1.1) # Setting the y-axis limit to be from 0 to 1
    # Adding tokens info
    ax.text(0.5, 0.95, f'Tokens: {tokens}', transform=ax.transAxes, ha='center', va='top')

def plot_compare_rags2(metrics, title = 'RAG model comparison', xtitle = 'Models'):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(metrics).T.reset_index().melt('index')

    # Create the figure and axes
    plt.figure(figsize = (12, 6))

    # Create a bar chart with Sea born
    sns.barplot(x  ='index', y = 'value', hue = 'variable', data = df, palette ="muted")

    # Adding titles and tags
    plt.title(title, fontsize=20)
    plt.xlabel(xtitle, fontsize = 15)
    plt.ylabel('Score', fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    # Adjust the y-axis limits
    plt.ylim(0, 1.3)

    # Add a legend
    plt.legend(title = 'Metrics', title_fontsize = '13', loc = 'upper right')

    # Adding grid lines
    plt.grid(axis = 'y', alpha = 0.5)

    # Adding data tags
    for bar in plt.gca().patches:
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom')
        
    # Adjust spacing
    plt.tight_layout()

    # Show the graph
    plt.show()



### Chunking with jordi's library
def jordi_tool_header(llm = ChatOpenAI(), uploaded_file = None):
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
            "tags": [
              "pre"
            ]
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
      ]
    }

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