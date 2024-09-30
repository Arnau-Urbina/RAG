import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from enum import Enum

import streamlit as st
from streamlit_option_menu import option_menu

from Langchain.tab_chunking_LangChain import tab_chunk
from Langchain.tab_embeddings_LangChain import tab_embed
from Langchain.tab_queries import tab_query
from Langchain.tab_retrieval_LangChain import tab_retrievers
from Langchain.tab_generations_LangChain import tab_generation
from Langchain.tab_metrics_LangChain import tab_metric
from Langchain.tab_results_LangChain import tab_comparision

from Llamaindex.tab_chunking_Llama import tab_chunk_Lla
from Llamaindex.tab_embeddings_Llama import tab_embed_Lla
from Llamaindex.tab_retrieval_Llama import tab_retrievers_Lla
from Llamaindex.tab_generations_Llama import tab_generation_Lla
#from Llamaindex.tab_metrics_Llama import tab_metric_Lla

def apis(api, leng, beg, name_api):
    if api == '':
        pass
    else: 
        if (api[:3] == beg and len(api) == leng):
            try: 
                os.environ[name_api] = api
                st.write(f"{name_api} was loaded correctly")
            except:  
                st.write(f"**An incorrect {name_api}, please try to write the api again.**")



class PAGE(Enum):
    API = 'APIs'
    LangChain = "LangChain"
    LlamaIndex = "LlamaIndex"


def init(): 
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
    with st.sidebar:
        selected = option_menu(
            menu_title = 'Libraries',
            options = [p.value for p in PAGE]
        )

    if selected == PAGE.API.value: 
        st.header('This page is for posting APIs: ')
        st.write("Set the APIs you want to use in the demo: ")

        open = st.text_input("OPEN AI:", placeholder = 'sk-.......')
        groq = st.text_input("GROQ", placeholder = 'gsk_....' )
        hf = st.text_input("HUGGING FACE", placeholder = "hf_.....")
        llama = st.text_input("LLAMA CLOUD API", placeholder = "llx-...")

        if st.button('Save', key = 'APIs'):
            apis(open, 51, 'sk-', 'OPENAI_API_KEY')
            apis(hf, 37, 'hf_', 'HUGGINGFACEHUB_API_TOKEN')
            apis(groq, 56, 'gsk', 'GROQ_API_KEY')
            apis(llama, 52 , 'llx-', 'LLAMA_CLOUD_API_KEY')

    if selected == PAGE.LangChain.value:
        # Tab settings
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Chunk", "Embeddings","Query", "Retrieval", "Generation", "Metrics", "Comparision"])
        
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
            # align material UI icons
            st.markdown('<style>.main .element-container > iframe { margin-top:-17px }</style>', unsafe_allow_html=True)
            st.json(body=st.session_state, expanded=False)

        with tab7: 
            tab_comparision() 

    if selected == PAGE.LlamaIndex.value: 
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Chunk", "Embeddings", "Query", "Retrieval", "Generation", "Metrics"])
        
        with tab1: 
            tab_chunk_Lla()

        with tab2: 
            tab_embed_Lla()

        with tab3: 
            tab_query()

        with tab4: 
            tab_retrievers_Lla()

        with tab5: 
            tab_generation_Lla()

        with tab6:
            pass #UPDATE !! 
            #tab_metric_Lla()
    
        #### ADD THE TAB OF COMPARISION, USE THE SAME CSV OR DIFFERENT ? 




if __name__ == "__main__":
    # start_api()
    init()


    
