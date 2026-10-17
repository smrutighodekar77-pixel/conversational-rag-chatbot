
import os
from functools import lru_cache

import bs4
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()

ARTICLE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
DB_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "agent_article"


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


@lru_cache(maxsize=1)
def get_llm():
    """Initialize the Groq chat model."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-20b",
    )


def load_documents():
    """Load the source article."""
    loader = WebBaseLoader(
        web_paths=(ARTICLE_URL,),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    return loader.load()


@lru_cache(maxsize=1)
def get_vectorstore():
    """Load Chroma or create its index if empty."""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIRECTORY,
    )

    if vectorstore._collection.count() == 0:
        documents = load_documents()

        if not documents:
            raise RuntimeError(
                "No documents were loaded from the article."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = splitter.split_documents(documents)
        vectorstore.add_documents(documents=splits)

    return vectorstore


@lru_cache(maxsize=1)
def get_rag_chain():
    """Build the conversational RAG chain."""
    llm = get_llm()
    retriever = get_vectorstore().as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference the chat history, formulate a "
        "standalone question that can be understood without "
        "the chat history. Do not answer the question. "
        "If it is already standalone, return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    qa_system_prompt = (
        "You are an assistant answering questions about "
        "Lilian Weng's article 'LLM-powered Autonomous Agents'. "
        "Answer using only the retrieved article context. "
        "Use chat history only to understand follow-up questions, "
        "not as a source of factual information. "
        "If the answer is not supported by the context, say "
        "you don't know based on the article. "
        "Keep the answer concise, using at most three sentences.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )

    return create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )


def get_answer(question, chat_history):
    """Answer a question using retrieval and conversation history."""
    rag_chain = get_rag_chain()

    response = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history,
        }
    )

    return response["answer"]
