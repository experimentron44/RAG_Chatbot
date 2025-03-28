from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Set your HuggingFace Hub API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Step 1: Load the text file
loader = TextLoader("scraped_data.txt")  # Replace with your file path
documents = loader.load()

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Step 3: Use local embeddings model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 4: Create FAISS index
docsearch = FAISS.from_documents(chunks, embeddings)

# Step 5: Use a HuggingFace model as your LLM
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})

# Step 6: Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

def get_response(query):
    return qa_chain.run(query)
