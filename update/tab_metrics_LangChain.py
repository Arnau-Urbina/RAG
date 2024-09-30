import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from ragas.metrics import (faithfulness, context_precision, answer_relevancy)
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate

from Langchain.tab_retrieval_LangChain import embedding
from Langchain.tab_retrieval_LangChain import load_llm #llm

### Dictonary to do the evaluation with ragas 
def data_dict(query, ground_truth, answer, context, k): 
    if k == 1: 
        contexts = []
        for i in context: 
            contexts.append([i])

        data_samples = {'question': query, 
            'answer': answer,
            'contexts': contexts, 
            'ground_truth': ground_truth
        }
        rag_dict = Dataset.from_dict(data_samples)
    else: 
        data_samples = {'question': query, 
            'answer': answer,
            'contexts': context, 
            'ground_truth': ground_truth
        }
        rag_dict = Dataset.from_dict(data_samples)

    return rag_dict


def ragas_metrics(data, llm, embeddings):
  vllm = LangchainLLMWrapper(langchain_llm = llm)
  faithfulness.llm = vllm
  context_precision.llm = vllm
  answer_relevancy.llm = vllm
  answer_relevancy.embeddings = embeddings

  score = evaluate(data, metrics = [faithfulness, answer_relevancy, context_precision],
                    raise_exceptions = False)
  return score


### View metric evaluations 
def plot_metrics(scores, title = 'RAG Metrics'):
    # Create the plot
    plt.figure(figsize=(10, 6))
    names = list(scores.keys())
    values = list(scores.values())
    bars = plt.bar(names, values, color = sns.color_palette("viridis", len(names)))

    # Adding the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,  # x-position
                height + 0.01,  # y-position
                f'{height:.4f}',  # value
                ha = 'center', va = 'bottom')

    plt.ylabel('Score')
    plt.title('RAG Metrics')
    plt.ylim(0, 1.2)  # Setting the y-axis limit to be from 0 to 1
    #plt.show()

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())




def tab_metric():
    st.header("RAG Metrics")
    st.write("Once we have made the RAG, we are going to evaluate it with different metrics from the ragas library.")

    if st.button("Run", key = "Metrics_Lang"):
        questions = st.session_state['editable_questions'] 
        ground_truths = st.session_state['editable_ground_truths']
        context = st.session_state['context']  
        answer = st.session_state['answer']
        embed_model = st.session_state['embed_model']
        k = st.session_state['k']
        llm_model = st.session_state['llm_model']
        llm = load_llm(llm_model)

        embeddings = embedding(embed_model)
        db =  data_dict(questions, ground_truths, answer, context, k)
        
        st.write('Scores: ')
        scores = ragas_metrics(db, llm, embeddings)
        st.write(scores)
        st.write('We can see the metrics in a graph:')
        plot_metrics(scores = scores)

        st.write('Finally, we show the table with the different queries and their groundtruth, answer, context and metrics.')
        obs = pd.DataFrame(db)
        met = pd.DataFrame(scores.scores)
        table = pd.concat([obs, met], axis=1)
        st.write(table)

        st.session_state['scores'] = scores




