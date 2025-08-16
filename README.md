PDF Q&A and Summarization with LangGraph

An AI-powered interactive PDF assistant built with LangChain, LangGraph, FAISS, HuggingFace Embeddings, and Streamlit.

This tool allows you to:

📂 Upload any PDF document

❓ Ask natural language questions about the content

📝 Generate concise, context-aware answers with page references

📑 Summarize entire PDFs (hierarchical chunk-based summarization)

🔄 Automatically rephrase/refine unclear queries for better retrieval

✅ Grade retrieved documents to ensure only relevant results are used

🚀 Features
🔹 PDF Processing

Loads PDFs using PyPDFLoader (fallback: UnstructuredPDFLoader)

Cleans and normalizes text for better chunking

Splits into overlapping chunks (RecursiveCharacterTextSplitter)

🔹 Embeddings & Vector Store

Uses HuggingFace nomic-embed-text-v1 embeddings

Stores chunks in a FAISS vector store

Provides similarity & MMR-based retrieval

🔹 Intelligent RAG Workflow (LangGraph)

Retriever → Finds relevant chunks from the PDF

Retriever Grader → Filters irrelevant results with an LLM-based relevance check

Question Rewriter → Converts conversational queries into standalone questions

Refine Question → Iteratively improves queries if retrieval fails

Summarizer → Multi-step chunk summarization + hierarchical merging for large PDFs

Generate Answer → Produces clear, concise responses with references

Cannot Answer → Gracefully handles unanswered queries

🔹 User Interface (Streamlit)

Intuitive chat-like interface (st.chat_input)

Displays both answers and referenced page numbers

PDF Upload + Question/Answer in real-time

🛠️ Tech Stack

LangChain – Orchestrates LLM pipelines

LangGraph – State-based reasoning workflow

FAISS – Vector similarity search

HuggingFace Embeddings – Dense vector embeddings

Groq Llama 3.1 – Fast inference LLM for Q&A & summarization

Streamlit – Interactive UI for chatting with PDFs

Python 3.10+

📂 Project Structure
📦 pdf-qa-langgraph
├── app.py                # Main Streamlit app (the code you provided)
├── requirements.txt       # Dependencies
├── faiss_index/           # FAISS vector index storage
├── .env                   # API keys / environment variables
└── README.md              # Documentation

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/pdf-qa-langgraph.git
cd pdf-qa-langgraph

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

5️⃣ Run the App
streamlit run app.py

💡 Usage

Upload a PDF document

Type your question in the chat box

Example: “Summarize the PDF in 5 bullet points”

Example: “What does section 2.3 talk about?”

Get concise answers with page references

🧠 Workflow Explanation
flowchart TD
    A[User Question] --> B[Task Router]
    B -->|If Summarization| C[Summarize PDF]
    B -->|Else| D[Question Rewriter]
    D --> E[Retriever]
    E --> F[Retrieval Grader]
    F -->|Relevant Docs| G[Generate Answer]
    F -->|No Docs & Retries < 2| H[Refine Question]
    F -->|No Docs & Retries ≥ 2| I[Cannot Answer]
    H --> E
    C --> B
    G --> J[Final Answer with Page References]
    I --> J

🔮 Future Improvements

 Support for multiple file uploads

 Multi-modal retrieval (PDF + images)

 Persistent FAISS storage across sessions

 Fine-grained summarization (per section/chapter)

 Export chat history & summaries as PDF

🤝 Contributing

Contributions are welcome! Please fork this repo, make changes, and submit a pull request.

