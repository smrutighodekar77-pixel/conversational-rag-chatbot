# Conversational RAG Chatbot

A conversational question-answering chatbot built using Retrieval-Augmented Generation (RAG). It answers questions based on Lilian Weng's article, *LLM-powered Autonomous Agents*, and supports follow-up questions by using conversation history to maintain context.

## Tech Stack

- **Python** — Core programming language
- **Streamlit** — Chat interface
- **LangChain** — RAG pipeline and conversational logic
- **Groq LLM** — `openai/gpt-oss-20b`
- **Hugging Face Embeddings** — `all-MiniLM-L6-v2`
- **ChromaDB** — Vector storage and similarity search
- **Python-dotenv** — Environment variable management

## Features

- Loads and processes an article from the web
- Splits the document into smaller chunks
- Generates embeddings and stores them in ChromaDB
- Retrieves relevant document context for user questions
- Supports follow-up questions using conversation history
- Provides an interactive chat interface with Streamlit

## Project Structure

- `app.py` — Streamlit application and chat interface
- `rag_pipeline.py` — Document processing, retrieval, and answer generation
- `requirements.txt` — Project dependencies
- `.env.example` — Example environment variable names

## Setup and Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your own API credentials.
5. Run the application:

   ```bash
   streamlit run app.py
   ```

## Knowledge Source

Lilian Weng — *LLM-powered Autonomous Agents*

https://lilianweng.github.io/posts/2023-06-23-agent/

## Future Improvements

- Evaluate retrieval accuracy and answer quality
- Improve handling of questions beyond the source article
- Deploy the application for public access
