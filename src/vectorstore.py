from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.transcript import get_transcript


class E5Embeddings(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts):
        return self.embedding_model.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text):
        return self.embedding_model.embed_query(f"query: {text}")


def create_vectorstore(video_id):
    """
    Creates a FAISS vector store from the transcript of a YouTube video.

    Args:
        video_id (str): The YouTube video ID.
    Returns:
        FAISS: A FAISS vector store containing the transcript chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = splitter.create_documents([get_transcript(video_id)])
    print(f"Transcript split into {len(chunks)} chunks.")

    embeddings = E5Embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

