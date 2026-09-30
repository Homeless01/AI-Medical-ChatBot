import os
import streamlit as st

from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =========================
# Load environment variables
# =========================
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="AI Medical ChatBot",
    page_icon="🩺",
    layout="centered"
)


# =========================
# Load Vector Store
# =========================
@st.cache_resource
def get_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# =========================
# Custom Prompt
# =========================
CUSTOM_PROMPT_TEMPLATE = """
You are a medical information assistant.

Use ONLY the information provided in the context to answer the user's question.

If the answer is not available in the context, say:

"I don't know based on the provided medical information."

Do not make up information.
Do not provide information outside the given context.

Context:
{context}

Question:
{question}

Answer:
"""


prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# =========================
# Format retrieved documents
# =========================
def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# =========================
# Main App
# =========================
def main():

    st.title("🩺 AI Medical ChatBot")

    st.caption(
        "Ask questions based on the medical information in the knowledge base."
    )

    # -------------------------
    # Initialize chat history
    # -------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -------------------------
    # Display previous messages
    # -------------------------
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -------------------------
    # User input
    # -------------------------
    user_query = st.chat_input(
        "Ask your medical question..."
    )

    if user_query:

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        try:

            # -------------------------
            # Load vector store
            # -------------------------
            vectorstore = get_vectorstore()

            # -------------------------
            # Create retriever
            # -------------------------
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            # -------------------------
            # Load Groq LLM
            # -------------------------
            llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.5,
            groq_api_key=os.environ["GROQ_API_KEY"]
            )

            # -------------------------
            # Create RAG chain
            # -------------------------
            qa_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # -------------------------
            # Generate response
            # -------------------------
            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    response = qa_chain.invoke(user_query)

                st.markdown(response)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")


# =========================
# Run application
# =========================
if __name__ == "__main__":
    main()