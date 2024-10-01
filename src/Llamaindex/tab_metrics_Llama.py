import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from llama_index.core.evaluation import BatchEvalRunner, CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
from tab_generations_Llama import llm_Llama 

#def metrics_Llama(llm, qe, queries):
#    # Initialize the evaluators
#    correctness_evaluator = CorrectnessEvaluator(llm = llm) # Useful for measuring if the response is correct against a reference answer
#    faithfulness_evaluator = FaithfulnessEvaluator(llm = llm) # Useful for measuring if the response is hallucinated
#    relevancy_evaluator = RelevancyEvaluator(llm = llm) # Useful for measuring if the query is actually answered by the response
#    runner = BatchEvalRunner(
#    {
#        "correctness": correctness_evaluator,
#        "faithfulness": faithfulness_evaluator,
#        "relevancy": relevancy_evaluator,
#    }, show_progress = True)
#    # Run the asynchronous evaluation
#    eval_result = await runner.aevaluate_queries(
#        query_engine = qe, #index.as_query_engine(),
#        queries = [question for question in queries]
#    )
#
#return eval_result



### View metric evaluations 
def plot_metrics(scores, title = 'RAG Metrics'):
    # Create the plot
    plt.figure(figsize=(10, 6))
    names = list(scores.keys())
    values = list(scores.values())
    bars = plt.bar(names, values, color=sns.color_palette("viridis", len(names)))

    # Adding the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,  # x-position
                height + 0.01,  # y-position
                f'{height:.4f}',  # value
                ha='center', va='bottom')

    plt.ylabel('Score')
    plt.title('RAG Metrics')
    plt.ylim(0, 1.2)  # Setting the y-axis limit to be from 0 to 1
    #plt.show()

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())




def tab_metric_Lla():
    st.header("RAG Metrics")
    st.write("Once we have made the RAG, we are going to evaluate it with different metrics from the ragas library.")

    if st.button("Run", key = "Metrics_Llama"):
        questions = st.session_state['editable_questions'] 
        #ground_truths = st.session_state['editable_ground_truths']
        context = st.session_state['context']  
        answer = st.session_state['answer']
        retriever = st.session_state['retriever_Llama']

        st.write('Scores: ')
        #scores = metrics_Llama(llm_Llama, retriever, questions)
        st.write(scores)
        #st.write('We can see the metrics in a graph:')
        #plot_metrics(scores = scores)
        
        #st.write('Finally, we show the table with the different queries and their groundtruth, answer, context and metrics.')
        #obs = pd.DataFrame(db)
        #met = pd.DataFrame(scores.scores)
        #table = pd.concat([obs, met], axis=1)
        #st.write(table)

