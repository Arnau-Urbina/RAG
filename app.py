import streamlit as st
from tab_chunking import tab_chunk
from tab_embeddings import tab_embed
from tab_queries import tab_query
from tab_retrieval import tab_retrievers
from tab_generations import tab_generation
from tab_metrics import tab_metric

st.set_page_config(
    page_title = "RAG",
    page_icon = "icon.png",
    layout = "wide",
    initial_sidebar_state = "expanded",
    menu_items={
        "Get help": "https://github.axa.com/arnau-urbina-external/RAG",
        "Report a bug": "https://github.axa.com/arnau-urbina-external/RAG",
        "About": """
            ## Evaluation RAG 
            
            **GitHub**: https://github.axa.com/arnau-urbina-external/RAG
            
            This demo provides the different steps of RAG and does them separately so that you
            can see what the RAG model returns at each step. This way we can check if the chunk
            returned by the retriever is a chunk that can really answer the question we pass to the model. 
        """
    }
)


# Tab settings
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Chunk", "Embeddings","Query", "Retrieval", "Generation", "Metrics"])


with tab1:
    tab_chunk()



with tab2: 
    tab_embed()



with tab3:
    tab_query()



with tab4:
    tab_retrievers()



with tab5:
    tab_generation()
    


with tab6: 
    tab_metric()