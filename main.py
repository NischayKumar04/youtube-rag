from src.rag import build_chain
from src.vectorstore import create_vectorstore

current_video_id = None
vectorstore = None
chain = None

while True:
    video_id = input("\nEnter Video ID (or 'exit'): ")

    if video_id.lower() == "exit":
        break

    if video_id != current_video_id:
        print("Creating vectorstore...")
        vectorstore = create_vectorstore(video_id)
        chain = build_chain(vectorstore)
        current_video_id = video_id
        print("Vectorstore ready!")

    while True:
        question = input("\nAsk a question ('change' for new video): ")

        if question.lower() == "change":
            break

        answer = chain.invoke(question)
        print("\nAnswer:")
        print(answer)
