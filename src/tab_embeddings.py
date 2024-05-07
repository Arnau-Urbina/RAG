import streamlit as st
import pandas as pd
import numpy as np 
from tqdm import tqdm
import umap
import plotly.express as px

from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings


def embed(name_model, text):
  if name_model == 'text-embedding-3-large': 
    model = OpenAIEmbeddings(model = 'text-embedding-3-large')
    return model.embed_documents(text)
  else: 
    model = SentenceTransformer(name_model)
    return model.encode(text, normalize_embeddings = True) 

### To graph embeddings with UMAP
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def tab_embed():
    st.header("Select the embedding model")
    st.markdown("These are the available embedding models and some of their characteristics, this information is taken from [Hugging Face.](https://huggingface.co/spaces/mteb/leaderboard)")
    table = pd.DataFrame(data=[
            ["text-embedding-3-large", "", "3072", "8191", "64.59", "59.16", "55.44"],
            ["WhereIsAI/UAE-Large-V1", "335", "1024", "512", "64.64", "59.88", "54.66"],
            ["thenlper/gte-base-en-v1.5", "137", "768", "8192", "64.11", "57.66", "54.09"],
            ["BAAI/bge-base-en-v1.5", "109", "768", "512", "63.55", "58.86", "53.25"],
            ["nomic-ai/nomic-embed-text-v1.5", "137", "768", "8192", "62.28", "55.78", "53.01"],
            ["BAAI/bge-small-en-v1.5", "33", "384", "512", "62.17", "58.36", "51.68"],
            ["sentence-transformers/all-MiniLM-L6-v2", "23", "384", "512", "56.26", "58.04", "41.95"]
    ], 
    columns = [
        "Model", "Model Size (Million Parameters)", 
        "Embedding Dimensions", "Max Tokens",
          "Average (56 datasets)", "Reranking Average (4 datasets)", "Retrieval Average (15 datasets)"])

    my_table = st.table(table)
    st.write('If you have chunked the documents, you can choose an embedidng model and see the representation of the chunks: ')
    embed_model = st.selectbox("Select embedding model:", ["text-embedding-3-large","BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", 
    "sentence-transformers/all-MiniLM-L6-v2", "WhereIsAI/UAE-Large-V1", "nomic-ai/nomic-embed-text-v1.5", "thenlper/gte-base-en-v1.5"], 
    key = "embedding model")
    st.session_state['embed_model'] = embed_model
    if st.button("Run", key = "umap"):
        chunks_gen = st.session_state['chunks_gen']
        embeddings = embed(embed_model, [i.page_content for i in chunks_gen])
        umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
        projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
        # Create a DataFrame with the data
        df = pd.DataFrame(projected_dataset_embeddings, columns=['x', 'y'])
        df['title'] = [chunk.metadata['title'][:-5] for chunk in chunks_gen]
        # Creating a scatter plot with Plotly
        fig = px.scatter(df, x = 'x', y = 'y', color = 'title', hover_data=['title'])
        # Add title and axis labels
        fig.update_layout(
            title = "UMAP projection of the document embeddings",
            xaxis_title = "UMAP dimension 1",
            yaxis_title = "UMAP dimension 2",
            hovermode = "closest"
        )
        # Show the plot
        st.plotly_chart(fig)
        
