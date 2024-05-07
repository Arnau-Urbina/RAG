# RAG 
This demo is designed to test RAG and see that the chunk returned by the retriever contains information to answer the question. In this demo there are 6 tabs: Chunks, Embeddings, Query, Retrieval, Generation and Metrics. 

## Chunks  
In this first tab we load all the HTML documents we want to use and choose the method and parameters we want to chunking. Then the chunks of each document and their tokens will be printed in order. 
There are 5 methods: 

- **Character separator**: Split text based on a user-defined character. One of the simplest methods.
- **Recursive Text Splitter**: Recursively splits text. Splitting text recursively serves the purpose of trying to keep related pieces of text next to each other. This is the recommended way to start splitting text.
- **Semantic chunker**: First splits on sentences. Then combines ones next to each other if they are semantically similar enough.
- **Semantic** text separator: Identifies distinct topics that form coherent pieces of text and splits along those.
- **Header**: Separates the document by headers, this method is created from Jordi's library.
 
Below is an example of what a pair of chunks would look like using the *Headers* method: 
![image](https://github.com/Arnau-Urbina/RAG/assets/163839495/9a7edd59-613d-4df2-8995-725a223e2483)

## Embeddings

At the top of the tab there is a table with all the models that are available in the demo with some of their characteristics and with some metrics. This information is taken from [Hugging Face](https://huggingface.co/spaces/mteb/leaderboard), all the models except the Open AI model are open source models.
If we select an embedding model and click on the button, we will create a graphical representation with the UMAP library of the chunks created in the previous tab. It can be useful to visualise the relationship between different chunks in a lower dimensional space.

Now let's show how the graphical representation of all the uploaded documents would look like with the openai embedding model and using the *Headers* method to make the chunks. 



## Query

In this tab we have two options: 

- We upload a `.csv` or `.xlsx` file with questions and ground truth. 

- Manually create the dataset of questions and ground truth. 

We load the dataset we have in the repository: 
![image](https://github.com/Arnau-Urbina/RAG/assets/163839495/59760d5e-2309-4068-a43e-5b8717c913e0)


## Retrieval 

Once we have selected the method of chunking, the embedding model and loaded the dataset, we only have to choose which method we are going to use for retrieval. 

In this case we have three retrievers options: 

- **Similarity Search** 
- **MMR**
- **Hybrid Search**: Combines Similarity Search and BM25. 

All options use FAISS as vectorstore. 

We will test using Similarity Search. And following the pattern of the previous examples, the chunking will be done with *Headers* and the embedding model will be the Openai model. 

  
