# LLM Rag

This project is to build a RAG (Retriever Augmented Generation System) for NUS students to help select their modules.  There are 16 thousand modules in semester 2025/2026. Sometimes it is difficult for a student to find a module that can help them learn. A RAG system will allow students to chat to a vector database of all modules. Example questions will be things like "What modules can I take to learn more about generative AI?"


## Getting module info
The module info are found from https://api.nusmods.com/v2/.
An example for retrieving would be ``get_data_and_store.ipynb``. The modules are then stored in a vector base locally using chroma, with the description being embedded, and the rest as metadata. In this example, Huggingface ``sentence-transformers/all-mpnet-base-v2`` is used. 


## LLM and embedding
The LLM used will be from gemni, gemini-2.5-flash, and the embedding model used is from huggingface, sentence-transformers/all-mpnet-base-v2. 
In order to use this, set up your ``.env`` file to have 
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```


## To run the backend 
```commandline
uvicorn main:app --reload
```
## To runs streamlit frontend
```commandline
streamlit run frontend.py
```

## RAG
The jupyter notebook ``LLM RAG example.ipynb`` shows how to query the RAG system, and it's results. 
