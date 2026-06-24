import streamlit as st

from src.rag import build_chain
from src.vectorstore import create_vectorstore


st.set_page_config(page_title="YouTube RAG", page_icon="🎥", layout="wide")

st.title("YouTube Video RAG")
st.caption("Ask questions from a video's transcript using Hugging Face embeddings and GLM-5.2 via the Hugging Face Inference API.")

with st.sidebar:
    st.header("Settings")
    video_id = st.text_input("YouTube Video ID", placeholder="J5_-l7WIO_w")
    k = st.slider("Top K chunks", min_value=1, max_value=10, value=4)
    load_clicked = st.button("Load video", type="primary")
    st.divider()
    st.write("Required environment variables:")
    st.code("HF_TOKEN", language="text")


@st.cache_resource(show_spinner=False)
def get_chain(video_id_value, top_k):
    vectorstore = create_vectorstore(video_id_value)
    return build_chain(vectorstore, k=top_k)


if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = ""
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if load_clicked:
    if not video_id.strip():
        st.error("Enter a video ID first.")
    else:
        try:
            with st.spinner("Creating vector store and loading chain..."):
                st.session_state.chain = get_chain(video_id.strip(), k)
                st.session_state.active_video_id = video_id.strip()
                st.session_state.messages = []
            st.success(f"Loaded video {video_id.strip()}.")
        except Exception as exc:
            st.session_state.chain = None
            st.error(f"Failed to load the video: {exc}")

if st.session_state.chain is None:
    st.info("Load a video from the sidebar to start chatting.")
else:
    st.write(f"Current video: `{st.session_state.active_video_id}`")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about the transcript")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
