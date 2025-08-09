# LLM Rag

This project is to build a RAG (Retriever Augmented Generation System) for NUS students to help select their modules.  There are 16 thousand modules in semester 2025/2026. Sometimes it is difficult for a student to find a module that can help them learn. A RAG system will allow students to chat to a vector database of all modules. Example questions will be things like "What modules can I take to learn more about generative AI?"


## Getting module info
The module info are found from https://api.nusmods.com/v2/.
An example for retrieving would be ``get_data_and_store.ipynb``. The modules are then stored in a vector base locally using chroma, with the description being embedded, and the rest as metadata. In this example, Huggingface ``sentence-transformers/all-mpnet-base-v2`` is used. 


## LLM
The LLM used will be from Ollama, running gemma3. To do so locally, make sure you have downloaded Ollama and simply run the command on command prompt or terminal ``ollama run gemma3``. Any LLM can be used


## RAG
``LLM RAG example.ipynb`` 
This notebook shows how to query the RAG system, and it's results. 