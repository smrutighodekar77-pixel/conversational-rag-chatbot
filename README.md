# Conversational RAG Chatbot

A conversational question-answering chatbot built using Retrieval-Augmented Generation (RAG). It answers questions based on Lilian Weng's article, *LLM-powered Autonomous Agents*, and supports follow-up questions using conversation history.

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq LLM (`openai/gpt-oss-20b`)
- Hugging Face Embeddings (`all-MiniLM-L6-v2`)
- ChromaDB
- Python-dotenv

## Features

- Loads and processes an article from the web
- Splits the article into chunks
- Creates embeddings and stores them in ChromaDB
- Retrieves relevant context for questions
- Supports conversational follow-up questions
- Provides a simple Streamlit chat interface

## Project Structure

- `app.py` — Streamlit application
- `rag_pipeline.py` — document processing, retrieval, and answer generation
- `requirements.txt` — Python dependencies
- `.env.example` — example environment variable names

## Setup

1. Clone this repository.
2. Create and activate a Python environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Create a `.env` file containing your own API credentials.
5. Run `streamlit run app.py`.


## Knowledge Source

Lilian Weng — LLM-powered Autonomous Agents:  
https://lilianweng.github.io/posts/2023-06-23-agent/

## Future Improvements

- Evaluate retrieval and answer quality
- Improve handling of questions outside the source article
- Deploy the application
