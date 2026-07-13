import argparse
import time
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace

from langchain_community.vectorstores import FAISS

# Safe, robust imports that avoid the broken 'langchain.retrievers' module path:
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# strict system prompt — answers only from retrieved context, refuses otherwise
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Retrieval-Augmented Generation assistant.

Answer ONLY from the provided context.

If the answer is not present, reply exactly:

I don't have enough information to answer that question.

Do not make up information.

Keep the answer concise.
""",
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
""",
        ),
    ]
)

# Embedding Model Ingestion
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Loads the sentence-transformer model and returns embeddings and dimension size."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"Loaded embedding model: {model_name}")
    embedding_dimension = len(embeddings.embed_query("dimension Probe"))
    return embeddings, embedding_dimension

# LLM Model Ingestion
def load_llm_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """Loads a causal LLM and wraps it with LangChain ChatHuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        temperature=0,
        do_sample=False,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=llm)

    print(f"Loaded LLM model: {model_name}")
    return chat_model

# Step 1: Document Loader
def load_document(file_path, source_type="auto", max_records=200):
    """Loads documents from PDF, TXT, or Hugging Face dataset."""
    if source_type == "auto":
        if file_path.lower().endswith(".pdf"):
            source_type = "pdf"
        elif file_path.lower().endswith(".txt"):
            source_type = "txt"
        else:
            source_type = "hf_dataset"
    
    if source_type == "pdf":
        docs = PyPDFLoader(file_path).load()
    elif source_type == "txt":
        docs = TextLoader(file_path).load()
    elif source_type == "hf_dataset":
        from datasets import load_dataset
        dataset = load_dataset(file_path, split="train")
        n = min(max_records, len(dataset))
        field = "text" if "text" in dataset.column_names else dataset.column_names[0]
        docs = [
            Document(page_content=str(dataset[i][field]),
                     metadata={"source": file_path, "record_index": i})
            for i in range(n)
        ]
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    
    print(f"Loaded {len(docs)} documents from {file_path}")
    return docs

# Step 2: Document Splitting
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Splits documents into smaller text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", "", "."]
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Split documents into {len(split_docs)} chunks")
    return split_docs

# Step 3: Vector Store Creation
def create_vector_store(docs, embeddings, vector_store=None):
    """Initializes or adds documents to FAISS vector index."""
    if vector_store is None:
        vector_store = FAISS.from_documents(docs, embeddings)
    else:
        vector_store.add_documents(docs)
    print(f"Vector store now holds {vector_store.index.ntotal} vectors.")
    return vector_store

# Step 4: Retriever Builder
def build_retriever(vector_store, chunks, top_k=4, hybrid=False):
    """Builds a retriever (Pure Vector Similarity or Hybrid Vector + BM25)."""
    vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    if hybrid:
        bm25_retriever = BM25Retriever.from_documents(chunks, k=top_k)
        retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.6, 0.4])
    else:
        retriever = vector_retriever
   
    return retriever

# Step 5: Answer Generation
def generate_answer(retriever, llm, question):
    """Retrieves relevant context and generates answer using strict prompting."""
    retrieved_chunks = retriever.invoke(question)
    context = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer.strip(), retrieved_chunks

# Compile Metrics
def compile_metrics(chunk_size, chunk_overlap, all_chunks, embedding_model_name, 
                    embedding_dim, hybrid, vector_store, llm_model_name):
    """Generates the metadata tracking object for system analysis."""
    return {
        "chunking": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks_indexed": len(all_chunks),
            "splitter": "RecursiveCharacterTextSplitter",
        },
        "embedding_model": {
            "name": embedding_model_name,
            "dimension": embedding_dim,
        },
        "vector_store": {
            "backend": "LangChain FAISS",
            "hybrid_keyword_search": hybrid,
            "vectors_stored": vector_store.index.ntotal if vector_store else 0,
        },
        "llm": {
            "name": llm_model_name,
            "task": "text2text-generation",
        },
    }

# Main Pipeline Runner
def run_rag_pipeline(
    file_path=None,
    question=None,
    source_type="auto",
    chunk_size=400,        
    chunk_overlap=50,
    top_k=3,               
    hybrid=False,
    max_records=200,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct", 
    embeddings=None,
    llm=None,
    vector_store=None,
    all_chunks=None,
):
    # Load Embedding Model
    if embeddings is None:
        embeddings, embedding_dim = load_embedding_model(embedding_model_name)
    else:
        embedding_dim = len(embeddings.embed_query("dimension Probe"))

    # Load LLM Model
    if llm is None:
        llm = load_llm_model(llm_model_name)

    # Ingestion Stage
    if file_path is not None:
        docs = load_document(file_path, source_type, max_records)
        all_chunks = split_documents(docs, chunk_size, chunk_overlap)
        vector_store = create_vector_store(all_chunks, embeddings, vector_store)

    # Inference Stage
    if question is not None:
        if vector_store is None:
            raise ValueError("Vector store is not initialized.")
            
        retriever = build_retriever(vector_store, all_chunks, top_k, hybrid)
        answer, retrieved_chunks = generate_answer(retriever, llm, question)

        metrics = compile_metrics(
            chunk_size, chunk_overlap, all_chunks, embedding_model_name,
            embedding_dim, hybrid, vector_store, llm_model_name
        )
        return answer, retrieved_chunks, metrics

    return embeddings, llm, vector_store, all_chunks

# CLI Interface
def run_cli():
    parser = argparse.ArgumentParser(description="Simple RAG Chat CLI")
    parser.add_argument("--file", required=True, help="Path to PDF, TXT, or HF dataset")
    args = parser.parse_args()

    print(f"\n🚀 Ingesting and indexing: {args.file}...")
    
    embeddings, llm, vector_store, all_chunks = run_rag_pipeline(file_path=args.file)

    print("\n✅ Setup complete! Ask questions below. Type 'exit' or 'quit' to stop.\n")
    
    log_lines = []
    latest_metrics = None
    
    while True:
        try:
            query = input("🤖 You: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("\nShutting down pipeline and saving reports...")
                break
            
            start_time = time.time()
            
            answer, retrieved_chunks, metrics = run_rag_pipeline(
                question=query,
                embeddings=embeddings,
                llm=llm,
                vector_store=vector_store,
                all_chunks=all_chunks,
            )
            
            latency = round(time.time() - start_time, 2)
            latest_metrics = metrics
            
            print(f"💡 AI: {answer}")
            print(f"⏱️  (Latency: {latency}s)\n")
            
            # Build validation log
            log_lines.append(f"Q: {query}")
            log_lines.append(f"A: {answer}")
            log_lines.append(f"Latency: {latency}s")
            log_lines.append("Retrieved chunks:")
            for doc in retrieved_chunks:
                log_lines.append(f"  - {doc.metadata} :: {doc.page_content[:100]}...")
            log_lines.append("-" * 60)
            
        except KeyboardInterrupt:
            print("\nShutting down pipeline and saving reports...")
            break

    # Save Evaluation Artifacts On Exit
    if log_lines and latest_metrics:
        os.makedirs("outputs", exist_ok=True)
        
        # 1. Write Validation Logs
        validation_path = os.path.join("outputs", "validation_log.txt")
        with open(validation_path, "w") as f:
            f.write("\n".join(log_lines))
        print(f"📝 Saved: {validation_path}")

        # 2. Write System Metrics Report
        metrics_path = os.path.join("outputs", "metrics_report.md")
        with open(metrics_path, "w") as f:
            f.write("# RAG Pipeline Metrics Report\n\n")
            f.write("## Chunking Profile\n")
            for k, v in latest_metrics["chunking"].items(): 
                f.write(f"- **{k}**: {v}\n")
            f.write("\n## Embedding Model\n")
            for k, v in latest_metrics["embedding_model"].items(): 
                f.write(f"- **{k}**: {v}\n")
            f.write("\n## Vector Store\n")
            for k, v in latest_metrics["vector_store"].items(): 
                f.write(f"- **{k}**: {v}\n")
            f.write("\n## Language Model\n")
            for k, v in latest_metrics["llm"].items(): 
                f.write(f"- **{k}**: {v}\n")
        print(f"📊 Saved: {metrics_path}")
    else:
        print("No questions were evaluated; skipping log generation.")

if __name__ == "__main__":
    run_cli()