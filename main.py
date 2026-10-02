import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
import cohere

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    api_key=GROQ_API_KEY
)

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY
)

INDEX_NAME = "document-chatbot"

def uploaded_file(file_path):
    file_path = str(file_path).lower()

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")

    else:
        raise ValueError("Only PDF and TXT files are supported")

    return loader.load()

# QUERY REWRITING
rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query rewriting assistant.

Rewrite the user's question into a clear and specific search query
for document retrieval.

Rules:
- Keep the original meaning.
- Do not answer the question.
- Do not add information that is not present.
- Return only the rewritten query.

User Question:
{question}

Rewritten Query:
""")

def rewrite_query(query):
    chain = rewrite_prompt | llm
    result = chain.invoke({"question": query})
    rewritten = result.content.strip()
    return rewritten


# VECTOR STORE + BM25
def vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    namespace = f"doc_{uuid.uuid4().hex}"

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

    sim = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
        }
    )

    BM = BM25Retriever.from_documents(chunks)
    BM.k = 5

    return sim, BM, namespace

def RRF(sim, BM, query):

    sim_results = sim.invoke(query)
    bm_results = BM.invoke(query)

    score = {}
    doc_map = {}

    k = 60

    all_results = [sim_results, bm_results]

    for results in all_results:
        for rank, doc in enumerate(results, start=1):
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content
            )
            if key not in score:
                score[key] = 0
                doc_map[key] = doc

            score[key] += 1 / (k + rank)

    sorted_keys = sorted(
        score,
        key=score.get,
        reverse=True
    )
    fused_results = [
        doc_map[key]
        for key in sorted_keys
    ]

    documents = [
        doc.page_content
        for doc in fused_results
    ]

    rerank = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=min(3, len(documents))
    )

    if not rerank.results:
        return fused_results[:3]

    return [
        fused_results[result.index]
        for result in rerank.results
    ]


# FINAL ANSWER PROMPT
prompt = ChatPromptTemplate.from_template("""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information provided
in the document context.

If the answer is not available in the context, clearly say:

"I could not find the answer in the uploaded document."

Do not make up information.
Keep the answer clear, accurate, and relevant.

Document Context:
{context}

User Question:
{question}

Answer:
""")
