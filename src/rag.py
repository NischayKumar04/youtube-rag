import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.vectorstore import create_vectorstore


load_dotenv()


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_chain(vectorstore, model_name="zai-org/GLM-5.2", k=4):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use ONLY the transcript context below to answer. "
                   "If the answer cannot be found in the context, reply: "
                   "\"I don't know based on the transcript.\""),
        ("human", "Transcript Context:\n{context}\n\nQuestion:\n{question}"),
    ])

    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            max_new_tokens=512,
            temperature=0.2,
        )
    )

    parser = StrOutputParser()

    return (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | parser
    )


def build_chain_from_video_id(video_id, model_name="zai-org/GLM-5.2", k=4):
    vectorstore = create_vectorstore(video_id)
    return build_chain(vectorstore, model_name=model_name, k=k)