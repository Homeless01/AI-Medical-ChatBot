
import os

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings
)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS


# ============================================================
# Step 1: Setup Hugging Face LLM
# ============================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-72B-Instruct"


def load_llm(huggingface_repo_id):

    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

    chat_model = ChatHuggingFace(
        llm=llm
    )

    return chat_model


# ============================================================
# Step 2: Custom Prompt
# ============================================================

CUSTOM_PROMPT_TEMPLATE = """
You are a medical information assistant.

Use ONLY the information provided in the context
to answer the user's question.

If the answer is not available in the context, say:

"I don't know based on the provided medical information."

Do not make up information.

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


# ============================================================
# Step 3: Load FAISS Database
# ============================================================

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# ============================================================
# Step 4: Create Retriever
# ============================================================

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# Step 5: Format Retrieved Documents
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# Step 6: Load Chat LLM
# ============================================================

llm = load_llm(HUGGINGFACE_REPO_ID)


# ============================================================
# Step 7: Create RAG Chain
# ============================================================

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# Step 8: Ask User Query
# ============================================================

user_query = input("Write Query Here: ")

response = qa_chain.invoke(user_query)


# ============================================================
# Step 9: Display Result
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(response)
