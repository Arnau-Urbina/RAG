import streamlit as st
import pandas as pd


def tab_comparision():
    st.header("Comparision")
    st.write("In this tabe we can compare the results of a diferentes RAGs models.")

    if st.button("Run", key = 'Comparision_Lang'):
        chunk_model = st.session_state['chunk_model']
        chunk_size = st.session_state['chunk_size']
        overlap = st.session_state['overlap']
        embed_model = st.session_state['embed_model']
        retrieval_model = st.session_state['retrieval_model']
        questions = st.session_state['editable_questions'] 
        scores = st.session_state['scores']
        k = st.session_state['k']
        rerank_model = st.session_state['rerank_model']
        
        data = {
            "Chunk Model": [chunk_model],
            "Chunk Size": [chunk_size],
            "Overlap": [overlap],
            "Embed model": [embed_model],
            "Retrieval Model": [retrieval_model],
            "Reranker Model": [rerank_model],
            "k": [k],
            "Number of queries": [len(questions)],
            "Faithfulness": [scores['faithfulness']],
            "Answer relevancy": [scores['answer_relevancy']],
            "Context precision": [scores['context_precision']],
            "Framework": ['LangChain'],
        }

        table = pd.DataFrame(data)

        final_data = pd.read_csv('../Data/output/ComparisionRAGs.csv')
        merg = pd.concat([final_data, table], axis = 0)
        st.write(merg)
        merg.to_csv('../Data/output/ComparisionRAGs.csv', index = False)


