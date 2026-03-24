import os
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="📺")
st.title("📺 Chat with YouTube Video")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Get your key from [OpenAI](https://platform.openai.com/)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "loaded_video" not in st.session_state:
    st.session_state.loaded_video = ""

def build_chain(video_id: str, api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key

    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id, languages=["en"])
    transcript = " ".join(snippet.text for snippet in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g. wrcQwMpAirQ")

if st.button("Process Video"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not video_id.strip():
        st.warning("Please enter a YouTube Video ID.")
    else:
        with st.spinner("Loading transcript and building index..."):
            try:
                chain = build_chain(video_id.strip(), openai_api_key)
                st.session_state.qa_chain = chain
                st.session_state.loaded_video = video_id.strip()
                st.session_state.messages = []
                st.success(f"✅ Video `{video_id}` is ready! Ask your questions below.")
            except (NoTranscriptFound, TranscriptsDisabled):
                st.error("❌ No English transcript found for this video.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

if st.session_state.loaded_video:
    st.caption(f"Active video ID: `{st.session_state.loaded_video}`")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about the video..."):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif st.session_state.qa_chain is None:
        st.warning("Please process a YouTube video first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke(prompt)
                except Exception as e:
                    response = f"❌ Error: {e}"
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})