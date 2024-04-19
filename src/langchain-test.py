import os
import getpass
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

os.environ['OPENAI_API_KEY'] = getpass.getpass('langchain-test:')
# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('../dataset/data-set-test.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())
print("Hi")
query = "How does Karna's low birth affect his life and actions throughout the Mahabharata?"
docs = db.similarity_search(query)
print(docs[0].page_content)


