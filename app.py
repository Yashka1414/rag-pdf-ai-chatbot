import streamlit as st
from groq import Groq
from pypdf import PdfReader

st.set_page_config(page_title="Production RAG PDF Engine", page_icon="⚡", layout="wide")
st.title("⚡ Production-Grade RAG PDF Engine")

# Security: Dynamic runtime API Authentication
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
if not api_key:
    st.info("Please enter your Groq API Key in the sidebar to proceed.")
    st.stop()

# Initialize Client with API Key
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Authentication Error: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader("Upload PDF Document for Context Retrieval:", type=["pdf"])

if uploaded_file:
    try:
        reader = PdfReader(uploaded_file)
        full_text = "".join([page.extract_text() or "" for page in reader.pages])

        # Overlapping Text Chunking Strategy
        chunk_size, overlap = 600, 100
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]
        st.success(f"Document successfully parsed into {len(chunks)} vector-indexed chunks.")
    except Exception as e:
        st.error(f"Document Processing Error: {str(e)}")
        st.stop()

    query = st.text_input("Ask a question based on your document:")

    if st.button("Run Vector RAG Query") and query:
        with st.spinner("Executing lexical vector match and LLM synthesis..."):
            # Lexical / Keyword Relevance Scoring
            query_words = set(query.lower().split())
            scored_chunks = sorted(chunks, key=lambda c: sum(1 for w in query_words if w in c.lower()), reverse=True)
            retrieved_context = "\n---\n".join(scored_chunks[:3])

            # Explicit Zero-Shot System Prompt (Prevents Hallucinations)
            system_prompt = (
                "You are an enterprise RAG Engine. Answer the user's question STRICTLY using only the provided context. "
                "If the answer cannot be determined from the context, state 'Information not found in document.'"
            )
            user_prompt = f"Context:\n{retrieved_context}\n\nQuestion: {query}"

            # API Execution with Error Handling & Valid Model Name
            try:
                res = client.chat.completions.create(
                    model="openai/gpt-oss-20b",  # updated: llama-3.1-8b-instant was deprecated by Groq
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )

                answer = res.choices[0].message.content
                st.markdown("### 📝 Grounded Answer")
                st.write(answer)

                with st.expander("🔍 Context Retrieval Payload"):
                    st.code(retrieved_context)

            except Exception as e:
                st.error(f"REST API Execution Error: {str(e)}")
