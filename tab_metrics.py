import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from ragas.metrics import faithfulness, context_relevancy, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate

from tab_retrieval import embedding
from tab_generations import llm


### Dict to do the evaluation with ragas 
def data_dict(query, ground_truth, answer, context):
    data_samples = {'question': query, 
                    'answer': answer,
                    'contexts': context, #If 2 o more context: [context]
                    'ground_truth': ground_truth
                    }
    rag_dict = Dataset.from_dict(data_samples)

    return rag_dict


def data_dict2(retriever, query, ground_truth, answer):
  rag_dataset = []
  for i in range(len(query)):
    ans = retriever.invoke({"question" : query[i]})
    rag_dataset.append(
        {"question" : query[i],
         "answer" : answer[i],
         "contexts" : [context.page_content for context in ans["context"]],
         "ground_truths" : [ground_truth[i]]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def ragas_metrics(data, llm, embeddings):
  vllm = LangchainLLMWrapper(langchain_llm = llm)
  faithfulness.llm = vllm
  context_relevancy.llm = vllm
  answer_relevancy.llm = vllm
  answer_relevancy.embeddings = embeddings

  score = evaluate(data, metrics = [faithfulness, answer_relevancy, context_relevancy])
  return score


### View metric evaluations 
def plot_metrics(scores, title = 'RAG Metrics'):
    names = list(scores.keys())
    values = list(scores.values())
    
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




def tab_metric():
    st.header("RAG Metrics")
    st.write("Once we have made the RAG, we are going to evaluate it with different metrics from the ragas library.")

    if st.button("Run", key = "Metrics"):
        questions = st.session_state['editable_questions'] 
        ground_truths = st.session_state['editable_ground_truths']
        context = st.session_state['context']  
        answer = st.session_state['answer']
        embed_model = st.session_state['embed_model']

        embeddings1 = embedding(embed_model)

        dataset =  data_dict(questions, ground_truths, answer, context)
        st.write('First, we show the table with the different queries and their groundtruth, answer and context.')
        st.write(pd.DataFrame(dataset))
        st.write('Scores: ')
        scores = ragas_metrics(dataset, llm, embeddings1)
        st.write(pd.DataFrame(scores))
        #Add the scores in the table. Add the possibility to save the table. 
        #Graph metrics 

