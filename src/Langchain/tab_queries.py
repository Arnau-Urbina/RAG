import streamlit as st
import pandas as pd


def tab_query():
    """
    Load the query and ground truth
    """
    st.header('Load the query and ground truth')
    
    # 1. Option to upload a csv or xlsx file
    data_file = st.file_uploader("Upload the CSV or XLSX file with the questions and the ground truth:", type=["csv", "xlsx"], key = "input_uploader")
    if data_file is not None:
        df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
        questions = df['questions'].tolist()
        ground_truths = df['ground_truths'].tolist()
        document = df['document'].tolist()
        st.session_state['editable_questions'] = questions
        st.session_state['editable_ground_truths'] = ground_truths
        st.session_state['document'] = document

    else: # 2. Option to create question pairs and ground truth manually
        query = st.text_input("Query:",key = "question")
        ground_truth = st.text_input("Ground truth:", key = 'GroundTruth')
        file = st.text_input("Document:", key='documnet')
        questions = []
        ground_truths = []
        document = []
        if st.button("Add"):
            questions.append(query)
            ground_truths.append(ground_truth)
            document.append(file)
            st.session_state['editable_questions'] = questions
            st.session_state['editable_ground_truths'] = ground_truths
            st.session_state['document'] = document
    
    # Visualisation of data
    st.subheader('Database')
    data = {'Query': questions, 'Ground Truth': ground_truths, 'Document': document}
    df = pd.DataFrame(data)
    st.table(df)
