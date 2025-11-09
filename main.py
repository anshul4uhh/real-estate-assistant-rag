import streamlit as st
import time
from rag import process_urls, answer_query


st.set_page_config(
    page_title="Real Estate RAG Assistant",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>
.chat-container {
    padding: 10px 20px;
    max-width: 900px;
    margin: auto;
}
.user-msg {
    background-color: #CFF5D4;
    text-align: right;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    float: right;
    clear: both;
}
.bot-msg {
    background-color: #F1F0F0;
    text-align: left;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    float: left;
    clear: both;
}
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #e0e0e0;
}
footer {visibility: hidden;}
.block-container {padding-bottom: 60px !important;}
</style>
""", unsafe_allow_html=True)



st.sidebar.title("📘 Build Knowledge Base")
st.sidebar.markdown("Add up to **3 URLs**, build the vector DB, then chat.")

url1 = st.sidebar.text_input("🔗 URL 1", placeholder="https://example.com/page1")
url2 = st.sidebar.text_input("🔗 URL 2 (optional)", placeholder="https://example.com/page2")
url3 = st.sidebar.text_input("🔗 URL 3 (optional)", placeholder="https://example.com/page3")

urls = [u.strip() for u in [url1, url2, url3] if u.strip()]

if st.sidebar.button("Build Vector DB"):
    if not urls:
        st.sidebar.error("Please enter at least one URL!")
    else:
        st.sidebar.info(f"Processing {len(urls)} URL(s)...")
        progress_box = st.sidebar.empty()

        for msg in process_urls(urls):
            progress_box.write(f"{msg}")
            time.sleep(0.25)

        st.sidebar.success("Vector DB ready! Start chatting →")


st.sidebar.markdown("---")
st.sidebar.caption("After building the vector DB, use the chat window on the right.")


st.title("🏡 Real Estate RAG Chatbot")
st.markdown("<p style='text-align:center; color:gray;'>Ask questions based on your knowledge base.</p>",
            unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.markdown('<div class="chat-container">', unsafe_allow_html=True)


for chat in st.session_state.chat_history:

    if chat["role"] == "user":
        st.markdown(
            f"""
            <div class="user-msg">
                <b>User:</b> {chat["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        clean_sources = []
        if "sources" in chat:
            for s in chat["sources"]:
                if s and s not in clean_sources:
                    clean_sources.append(s)

        if clean_sources:
            src_html = "<br><b>Sources:</b><br>"
            for s in clean_sources:
                src_html += f"""• <a href="{s}" target="_blank" style="color:#0077cc">{s}</a><br>"""
        else:
            src_html = ""

        st.markdown(
            f"""
            <div class="bot-msg">
                <b>Assistant: </b> {chat["content"]}
                {src_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("</div>", unsafe_allow_html=True)



user_query = st.chat_input("💬 Ask a question...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.rerun()



if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    with st.spinner("🤔 Thinking..."):
        result = answer_query(st.session_state.chat_history[-1]["content"], k=3)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

    st.rerun()



st.markdown("""
<br><br><hr style="margin-top:40px;">
<center>
    <p style='color: gray; font-size: 14px;'>
        🏠 <b>Real Estate RAG Assistant</b><br>
        Powered by <i>Groq + Chroma + LangChain</i>
    </p>
</center>
""", unsafe_allow_html=True)
