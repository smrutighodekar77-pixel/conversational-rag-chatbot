
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from rag_pipeline import get_answer


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Conversational RAG Assistant")
st.caption(
    "Ask questions about LLM-Powered Autonomous Agents "
    "with RAG and conversation history."
)

# Keep the conversation available across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar controls.
with st.sidebar:
    st.header("Settings")
    st.write("Model: `openai/gpt-oss-20b`")
    st.write("Embeddings: `all-MiniLM-L6-v2`")
    st.write("Vector store: ChromaDB")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Render existing messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input.
question = st.chat_input("Ask a question about the article...")

if question:
    # Display and store the user's question.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the article and generating an answer..."):
                answer = get_answer(
                    question,
                    st.session_state.chat_history,
                )

            st.markdown(answer)

        # Save both sides of the exchange for future follow-ups.
        st.session_state.chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as exc:
        st.error(
            "The chatbot could not generate a response. "
            "Check your API key, internet connection, and dependencies."
        )
        # Useful for local debugging; avoid exposing secrets.
        st.exception(exc)
