from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
# OLD (deprecated)
# from langchain.embeddings import HuggingFaceEmbeddings

# NEW
from langchain_huggingface import HuggingFaceEmbeddings


import os

# ✅ Load and split text
loader = TextLoader("C:/Users/MONA/UniversityChatbot/university_fhse_scraped_programs.txt",encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# ✅ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Vector store
db = FAISS.from_documents(docs, embeddings)

# ✅ Load language model



llm = HuggingFaceHub(
    repo_id="google/flan-t5-base", 
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

# ✅ Chatbot loop
while True:
    query = input("Ask me anything about the university (or type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        break

    relevant_docs = db.similarity_search(query, k=1)
    context = relevant_docs[0].page_content

    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

    response = llm.invoke(prompt)
    print("\nAnswer:", response, "\n")
