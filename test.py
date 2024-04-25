import streamlit as st
import pandas as pd
from rag_func import chunks, tokens, embed, retriever, project_embeddings, jordi_tool_header, model_embed
import umap
import plotly.express as px


# Tab settings
tab1, tabs, tab2, tab3 = st.tabs(["Chunk", "Embeddings","Query", "Retrieval"])



with tab1:
    st.header('Split the documents')
    uploaded_files = st.file_uploader("Load HTML documents", accept_multiple_files=True, key="html_uploader")
    chunk_model = st.selectbox("Select chunking model:", ["Character Splitter", "Recursive Text Splitter", 
                                                         "Semantic Chunker", "Semantic Text Splitter", "Header"], key="model_select")
    
    if chunk_model == "Character Splitter":
        tipe = 'CharacterTextSplitter'
        chunk_size = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 100, key="chunk_size")
        overlap = st.number_input("Overlap:", min_value = 0, max_value = 90000, value = 0, key = "overlap")
    elif chunk_model == "Recursive Text Splitter":
        tipe = 'RecursiveCharacterTextSplitter'
        chunk_size = st.number_input("Chunk size:", min_value = 10, max_value = 100000, value = 100, key="chunk_size")
        overlap = st.number_input("Overlap:", min_value = 0, max_value = 90000, value = 0, key = "overlap")
    elif chunk_model == "Semantic Chunker":
        tipe = 'SemanticChunker'
        chunk_size = 0
        overlap = 0
    elif chunk_model == "Semantic Text Splitter":
        tipe= 'SemanticTextSplitter' 
        chunk_size = 0
        overlap = 0
    else: #chunk_model == "Header": 
        tipe = 'Jordi'

    if st.button("Chunking test", key="chunking_test"):
        if uploaded_files is not None:
            chunks_gen = []
            for uploaded_file in uploaded_files:
                if tipe == "Jordi": 
                    split = jordi_tool_header(uploaded_file = uploaded_file.name)

                else: 
                    split = chunks(tipe,  uploaded_file.name, chunk_size, overlap)
                    for i in split: i.metadata['title'] = uploaded_file.name
                #st.write(f"Document: {uploaded_file.name[:-5]}")
                st.header(f"Document: {uploaded_file.name[:-5]}", divider = 'grey')
                for chunk in split:
                    chunks_gen.append(chunk)
                    st.markdown(chunk.page_content)
                    st.write(f"**Tokens:** {tokens(chunk.page_content)}")
                    st.write("---")

            st.session_state['chunks_gen'] = chunks_gen






with tabs: 
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
    key="embedding_select")

    st.session_state['embed_model'] = embed_model

    if st.button("Test", key="umap"):
        chunks_gen = st.session_state['chunks_gen']

        embeddings = embed(embed_model, [i.page_content for i in chunks_gen])
        umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
        projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

        # Create a DataFrame with the data
        df = pd.DataFrame(projected_dataset_embeddings, columns=['x', 'y'])
        df['title'] = [chunk.metadata['title'][:-5] for chunk in chunks_gen]

        # Creating a scatter plot with Plotly
        fig = px.scatter(df, x='x', y='y', color='title', hover_data=['title'])

        # Add title and axis labels
        fig.update_layout(
            title="UMAP projection of the document embeddings",
            xaxis_title="UMAP dimension 1",
            yaxis_title="UMAP dimension 2",
            hovermode="closest"
        )

        # Show the plot
        st.plotly_chart(fig)






with tab2:
    st.header('Load the query and ground_truth')
    
    # 1. Option to upload a csv or xlsx file
    data_file = st.file_uploader("Upload the CSV or XLSX file with the questions and the ground truth:", type=["csv", "xlsx"], key="data_uploader")
    if data_file is not None:
        df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
        questions = df['questions'].tolist()
        ground_truths = df['ground_truths'].tolist()
        document = df['document'].tolist()
        st.session_state['editable_questions'] = questions
        st.session_state['editable_ground_truths'] = ground_truths
        st.session_state['document'] = document
    else:
        # 2. Option to create question pairs and ground truth manually
        questions = st.session_state.get('editable_questions', [""])
        ground_truths = st.session_state.get('editable_ground_truths', [""])
        questions[-1] = st.text_input("Query:", questions[-1], key="question")
        ground_truths[-1] = st.text_area("Ground Truth:", ground_truths[-1], key="ground_truth")
        if st.button("Add"):
            questions.append("")
            ground_truths.append("")
            st.session_state['editable_questions'] = questions
            st.session_state['editable_ground_truths'] = ground_truths
    
    # 3. Visualisation of data
    st.subheader('Database')
    data = {'Query': questions, 'Ground Truth': ground_truths}
    df = pd.DataFrame(data)
    st.table(df)






with tab3:
    st.header('Retrievals')
    st.write('On this page we choose the retrieval model we want to use.')
    retrieval_model = st.selectbox("Select retrieval model:", ["Similarity Search","MMR", "Hybrid Search"], 
    key = "retrieval")

    if st.button("Results", key="load_results"):
        questions = st.session_state['editable_questions'] 
        ground_truths = st.session_state['editable_ground_truths'] 
        document = st.session_state['document']
        chunks_gen = st.session_state['chunks_gen']
        embed_model = st.session_state['embed_model']

        embeddings1 = model_embed(embed_model)

        for i in range(len(questions)):
            retrieved_chunks = retriever(chunks_gen, questions[i], retrieval_model, embeddings1)
            st.write(f"**Query:** {questions[i]}")
            st.write(f"**Ground Truth:** {ground_truths[i]}")
            st.write(f"**Document:** {document[i]}")
            st.markdown(f"**Retrieval:** {retrieved_chunks[0].page_content}")
            st.write(f"**Retrieval tokens:** {tokens(retrieved_chunks[0].page_content)}")
            st.write(f"**Document:** {retrieved_chunks[0].metadata['title'][:-5]}")
            st.write("---")






# Instructions for Use
st.sidebar.header('Instructions for Use')
st.sidebar.write("""
1. Chunks
2. Embeddings
3. Queries and Ground Truth
4. Retrieval 
""")

