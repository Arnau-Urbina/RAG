import streamlit as st
import pandas as pd


def tab_query():
    st.header('Load the query and ground_truth')
    
    data_file = st.file_uploader("Upload the CSV or XLSX file with the questions and the ground truth:", type=["csv", "xlsx"], key = "input_uploader")
    
    # 1. Option to upload a csv or xlsx file
    if data_file is not None:
        df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
        questions = df['questions'].tolist()
        ground_truths = df['ground_truths'].tolist()
        document = df['document'].tolist()
        st.session_state['editable_questions'] = questions
        st.session_state['editable_ground_truths'] = ground_truths
        st.session_state['document'] = document

    else: # 2. Option to create question pairs and ground truth manually
        questions = st.session_state.get('editable_questions', [""])
        ground_truths = st.session_state.get('editable_ground_truths', [""])
        questions[-1] = st.text_input("Query:", questions[-1], key = "question")
        ground_truths[-1] = st.text_area("Ground Truth:", ground_truths[-1], key = "ground_truth")
        document = 'Manual'
        if st.button("Add"):
            questions.append("")
            ground_truths.append("")
            st.session_state['editable_questions'] = questions
            st.session_state['editable_ground_truths'] = ground_truths
            st.session_state['document'] = document
    
    # Visualisation of data
    st.subheader('Database')
    data = {'Query': questions, 'Ground Truth': ground_truths}
    df = pd.DataFrame(data)
    st.table(df)
