
import nest_asyncio
nest_asyncio.apply()
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import streamlit as st
import asyncio
import nest_asyncio
nest_asyncio.apply()
import os
from dotenv import load_dotenv

load_dotenv()  # works locally
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# If not found, fall back to Streamlit Cloud secrets
if not GROQ_API_KEY:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not huggingface_token:
    huggingface_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# Set environment for HuggingFace
if huggingface_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
FAISS_PATH = "faiss_index"

st.set_page_config(page_title="PDF Q&A with LangGraph", layout="wide")
st.title("PDF Q&A and Summarization")
st.markdown("""
Welcome to the **PDF Q&A and Summarization** tool!  

- Upload a PDF using the uploader below.  
- Ask any question related to the content of the PDF.  
- You can also ask for a summary of the PDF.  
- The assistant will provide concise answers and relevant page references.  
""")
# PDF upload
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
pdf_path = None
if uploaded_pdf:
    pdf_path = "uploaded.pdf"  
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())


docs = []
if pdf_path:
    def load_pdf(path: str):
        """Load a PDF using the best available loader."""
        try:
            loader = PyPDFLoader(file_path=path)
            return loader.load()
        except Exception:
            loader = UnstructuredPDFLoader(file_path=path)
            return loader.load()

    docs = load_pdf(pdf_path)
    docs = [
        Document(page_content=" ".join(d.page_content.split()), metadata=d.metadata)
        for d in docs
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True} 
    )

    if chunks:   
        vector_store = FAISS.from_documents(chunks, embedding=embed)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 3})
    else:
        st.warning("No chunks could be created from the uploaded PDF.")


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatGroq(model="llama-3.1-8b-instant")

template = """You are a smart, precise assistant. Answer the user's question **clearly and concisely** using the provided context and chat history. 
Only provide as much information as necessary. Do **not** give extra explanations or details unless the user explicitly asks for them.

Chat history: {history}

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatGroq(model="llama-3.1-8b-instant")
rag_chain = prompt | llm
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage


class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )

def task_router_node(state: AgentState):
    
    return state

def task_router_condition(state: AgentState):
    question_text = state["question"].content.lower()
    if "summarize" in question_text and "pdf" in question_text:
        return "summarize_pdf"
    return "question_rewriter"

def summarize_pdf(state: AgentState):
    print("Entering summarize_pdf")

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    llm = ChatGroq(model="llama-3.1-8b-instant")

    # Step 1: Summarize each chunk individually
    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.page_content.strip()
        if not text:
            continue

        # Safety: truncate each chunk if too long
        if len(text) > 4000:
            text = text[:4000]

        try:
            system_msg = SystemMessage(
                content="You are an expert summarizer. Summarize this text concisely."
            )
            human_msg = HumanMessage(content=text)
            response = llm.invoke([system_msg, human_msg])
            summary = (response.content or "").strip()
            if summary:
                chunk_summaries.append(summary)
                print(f"Chunk {i} summarized.")
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
            continue

    if not chunk_summaries:
        state["messages"].append(AIMessage(content="PDF Summary: No content to summarize."))
        return state

    # Step 2: Combine chunk summaries safely
    combined_text = "\n\n".join(chunk_summaries)
    # Hierarchical summarization if too long
    while len(combined_text) > 6000:
        print("Combined summaries too long, performing hierarchical summarization...")
        partial_summaries = []
        sub_chunks = [combined_text[i:i+4000] for i in range(0, len(combined_text), 4000)]
        for sub in sub_chunks:
            try:
                system_msg = SystemMessage(
                    content="You are an expert summarizer. Summarize this text concisely."
                )
                human_msg = HumanMessage(content=sub)
                resp = llm.invoke([system_msg, human_msg])
                partial_summaries.append((resp.content or "").strip())
            except Exception as e:
                continue
        combined_text = "\n\n".join(partial_summaries)

    # Step 3: Final overall summary
    try:
        system_msg_final = SystemMessage(
            content="Combine the following summaries into a single coherent, concise overall summary."
        )
        human_msg_final = HumanMessage(content=combined_text)
        final_resp = llm.invoke([system_msg_final, human_msg_final])
        final_summary = (final_resp.content or "").strip()

        if not final_summary:
            final_summary = "Unable to produce final summary."

        state["messages"].append(AIMessage(content=f"PDF Summary:\n\n{final_summary}"))
        print("Final summary generated successfully.")

    except Exception as e:
        print(f"Error generating final summary: {e}")
        state["messages"].append(AIMessage(content="PDF Summary: An error occurred during summarization."))

    return state



def question_rewriter(state: AgentState):
    print(f"Entering question_rewriter with following state: {state}")

    # Reset state variables except for 'question' and 'messages'
    state["documents"] = []
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["question"].content
        messages = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
            )
        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        llm = ChatGroq(model="llama-3.1-8b-instant")
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f"question_rewriter: Rephrased question: {better_question}")
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = state["question"].content
    return state



def retrieve(state: AgentState):
    print("Entering retrieve")
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    documents = retriever.invoke(state["rephrased_question"])
    print(f"retrieve: Retrieved {len(documents)} documents")
    state["documents"] = documents
    return state


class GradeDocument(BaseModel):
    score: str = Field(
        description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
    )

def retrieval_grader(state: AgentState):
    print("Entering retrieval_grader")

    # System-level instruction for grading
    system_message = SystemMessage(
        content=(
            "You are a grader assessing the relevance of a retrieved document "
            "to a user question. Only answer with 'Yes' or 'No'.\n\n"
            "If the document contains information relevant to the user's question, respond with 'Yes'. "
            "Otherwise, respond with 'No'."
        )
    )

    # LLM with structured output to GradeDocument schema
    llm = ChatGroq(model="llama-3.1-8b-instant")
    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []

    for doc in state["documents"]:
        # Human input for the grader
        human_message = HumanMessage(
            content=f"User question: {state['rephrased_question']}\n\nRetrieved document:\n{doc.page_content}"
        )
        try:
            result = structured_llm.invoke([system_message, human_message])
            # print(f"Grading document: {doc.page_content[:30]}... Result: {result.score.strip()}")
        except Exception as e:
            print(f"Error grading document: {e}")
            continue

        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)

    # Update state
    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0
    print(f"retrieval_grader: proceed_to_generate = {state['proceed_to_generate']}")

    return state


def proceed_router(state: AgentState):
    print("Entering proceed_router")
    rephrase_count = state.get("rephrase_count", 0)
    if state.get("proceed_to_generate", False):
        print("Routing to generate_answer")
        return "generate_answer"
    elif rephrase_count >= 2:
        print("Maximum rephrase attempts reached. Cannot find relevant documents.")
        return "cannot_answer"
    else:
        print("Routing to refine_question")
        return "refine_question"
    
def refine_question(state: AgentState):
    print("Entering refine_question")
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        print("Maximum rephrase attempts reached")
        return state
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
    Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatGroq(model="llama-3.1-8b-instant")
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()
    print(f"refine_question: Refined question: {refined_question}")
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state

def generate_answer(state: AgentState):
    print("Entering generate_answer")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    response = rag_chain.invoke(
        {"history": history, "context": documents, "question": rephrased_question}
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content=generation))
    print(f"generate_answer: Generated response: {generation}")
    return state

def cannot_answer(state: AgentState):
    print("Entering cannot_answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for."
        )
    )
    return state


from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
# Workflow
workflow = StateGraph(AgentState)
workflow.add_node("question_rewriter", question_rewriter)

workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("refine_question", refine_question)
workflow.add_node("cannot_answer", cannot_answer)
workflow.add_node("summarize_pdf", summarize_pdf)

workflow.add_edge("summarize_pdf", "task_router")
workflow.add_edge("question_rewriter", "retrieve")

workflow.add_edge("retrieve", "retrieval_grader")
workflow.add_conditional_edges(
    "retrieval_grader",
    proceed_router,
    {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer",
    },
)
workflow.add_node("task_router", task_router_node)
workflow.add_conditional_edges(
    "task_router",
    task_router_condition,
    {
        "summarize_pdf": "summarize_pdf",
        "question_rewriter": "question_rewriter",
    }
)

workflow.add_edge("refine_question", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
# workflow.add_edge("off_topic_response", END)
workflow.set_entry_point('task_router')
graph = workflow.compile(checkpointer=checkpointer)
user_input = st.chat_input("Ask something about your PDF...")
if user_input:
    # Show the user's message
    with st.chat_message("user"):
        st.write(user_input)

    input_data = {"question": HumanMessage(content=user_input)}

    # Assistant's message placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        page_labels_displayed = set()
        final_content = ""

        # Stream output from LangGraph
        for event in graph.stream(input=input_data, config={"configurable": {"thread_id": 2}}):
            for node_name, node_state in event.items():
                
                # If documents exist, display page labels
                for doc in node_state.get("documents", []):
                    page_label = doc.metadata.get("page_label", "Unknown")
                    page_labels_displayed.add(page_label)

                # Stream AI messages
                if "messages" in node_state and node_state["messages"]:
                    last_msg = node_state["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        final_content = last_msg.content
                        
                        # Display page labels (if any) above final answer
                        labels_text = (
                            f"**Page(s): {', '.join(sorted(page_labels_displayed))}**\n\n"
                            if page_labels_displayed else ""
                        )
                        placeholder.markdown(f"{labels_text}{final_content}")

else:
    st.info("Please upload a PDF to start chatting.")

