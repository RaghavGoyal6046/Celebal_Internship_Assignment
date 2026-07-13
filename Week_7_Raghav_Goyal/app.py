import streamlit as st
import time
from pipeline import (
    load_embedding_model, load_llm_model, load_document,
    split_documents, create_vector_store, build_retriever,
    generate_answer, compile_metrics
)

st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="📄")
st.title("📄 Retrieval-Augmented Generation (RAG) Chatbot")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ---- Cache heavy models (loaded once per session) ----
@st.cache_resource
def get_embeddings():
    return load_embedding_model(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_llm():
    return load_llm_model(LLM_MODEL_NAME)

# ---- Session state init ----
for key in ["vector_store", "all_chunks", "retriever", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---- Sidebar: source selection + settings ----
with st.sidebar:
    st.header("⚙️ Configuration")

    source_choice = st.radio("Document source", ["Upload file (PDF/TXT)", "Hugging Face dataset"])

    uploaded_file = None
    hf_dataset_name = None
    max_records = 200

    if source_choice == "Upload file (PDF/TXT)":
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    else:
        hf_dataset_name = st.text_input(
            "Hugging Face dataset name",
            placeholder="e.g. squad, wikitext, imdb"
        )
        max_records = st.number_input("Max records to load", value=200, min_value=1, step=50)

    st.markdown("### Advanced Settings")
    chunk_size = st.number_input("Chunk size", value=400, step=50)
    chunk_overlap = st.number_input("Chunk overlap", value=50, step=25)
    top_k = st.slider("Top K chunks to retrieve", 1, 10, 3)
    hybrid = st.checkbox("Enable Hybrid search (BM25 + Vector)", value=False)

    process_clicked = st.button("🚀 Process Document")

    ready_to_process = process_clicked and (
        uploaded_file is not None or (hf_dataset_name and hf_dataset_name.strip())
    )

    if process_clicked and not ready_to_process:
        st.warning("Please upload a file or enter a Hugging Face dataset name first.")

    if ready_to_process:
        with st.spinner("Loading models (first run downloads Qwen2.5-1.5B, may take a while)..."):
            embeddings, emb_dim = get_embeddings()
            llm = get_llm()

        with st.spinner("Ingesting & indexing document..."):
            if uploaded_file is not None:
                # save upload to a temp path since loaders need a filepath
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                docs = load_document(temp_path)
            else:
                docs = load_document(hf_dataset_name.strip(), source_type="hf_dataset", max_records=max_records)

            all_chunks = split_documents(docs, chunk_size, chunk_overlap)
            vector_store = create_vector_store(all_chunks, embeddings)
            retriever = build_retriever(vector_store, all_chunks, top_k, hybrid)

            st.session_state.vector_store = vector_store
            st.session_state.all_chunks = all_chunks
            st.session_state.retriever = retriever
            st.session_state.embeddings = embeddings
            st.session_state.emb_dim = emb_dim
            st.session_state.llm = llm
            st.session_state.chat_history = []

        st.success(f"Indexed {len(all_chunks)} chunks successfully! ✅")

# ---- Main chat area ----
if st.session_state.retriever is None:
    st.info("👈 Please choose a document source in the sidebar and click **Process Document** to begin.")
else:
    question = st.chat_input("Ask something about the processed document...")

    if question:
        start = time.time()
        answer, retrieved_chunks = generate_answer(
            st.session_state.retriever, st.session_state.llm, question
        )
        latency = round(time.time() - start, 2)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "latency": latency,
            "chunks": retrieved_chunks,
        })

    # Render chat history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(f"⏱️ Generation Latency: {turn['latency']} seconds")
            with st.expander("🔍 Show Retrieved Reference Chunks"):
                for i, doc in enumerate(turn["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — Metadata: `{doc.metadata}`")
                    st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

    # ---- Metrics panel ----
    with st.sidebar:
        if st.session_state.chat_history and st.button("📊 Show Current Session Metrics"):
            metrics = compile_metrics(
                chunk_size, chunk_overlap, st.session_state.all_chunks,
                EMBEDDING_MODEL_NAME, st.session_state.emb_dim,
                hybrid, st.session_state.vector_store, LLM_MODEL_NAME
            )
            st.json(metrics)