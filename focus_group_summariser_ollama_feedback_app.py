# Import necessary libraries
import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import nltk
from datetime import datetime, timedelta
import os
import csv
import uuid
import matplotlib.pyplot as plt

# Download the tokenizer for sentence splitting
nltk.download('punkt')

# Initialize the Ollama Llama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
    num_predict=256
)

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the prompt template
template = """Use the context below to answer the question or provide a summary:
Context:
{context}
Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)

# Feedback storage file path
FEEDBACK_FILE = "feedback.csv"

# Function to initialize feedback file
def initialize_feedback_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Query", "Response", "Score", "Comment", "Answer ID"])

initialize_feedback_file(FEEDBACK_FILE)

# Function to save feedback locally
def save_feedback(query, response, score, comment, answer_id):
    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), query, response, score, comment, answer_id])

# Function to process uploaded files
@st.cache_resource
def process_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if "transcript" not in df.columns or "topic_name" not in df.columns:
            st.error("CSV must contain 'transcript' and 'topic_name' columns.")
            return None, None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        documents = [
            Document(page_content=chunk, metadata={"topic_name": row["topic_name"]})
            for _, row in df.iterrows()
            for chunk in text_splitter.split_text(row["transcript"])
        ]

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory="./chromadb"
        )
        return vectorstore, df
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        return None, None

# Function to create the RAG chain
def get_rag_chain(vectorstore, query, topic_name):
    retriever = get_retriever_for_topic(vectorstore, topic_name)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to create retriever
def get_retriever_for_topic(vectorstore, topic_name=None):
    search_kwargs = {"k": 3}
    if topic_name:
        search_kwargs["filter"] = {"topic_name": topic_name}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

# Main header
st.title("Deliberation Insight Assistant Demo")
st.subheader("Powered by Llama 3.2")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File with Deliberation Data", type=["csv"])
if uploaded_file is not None:
    vectorstore, df = process_uploaded_file(uploaded_file)

    # Dropdown for selecting a topic (optional)
    topic_options = [""] + sorted(df["topic_name"].unique())
    selected_topic = st.selectbox("Select Topic (optional)", topic_options)
    topic_name = selected_topic if selected_topic != "" else None

    # Text input for the query
    query = st.text_input("Enter your question or request for a summary")

    if st.button("Get Answer"):
        # Clear feedback when a new query is submitted
        st.session_state.feedback_score = None

        # Generate a unique answer ID for this query-response interaction
        st.session_state.answer_id = str(uuid.uuid4())

        rag_chain = get_rag_chain(vectorstore, query, topic_name)

        # Generate response
        result = ""
        with st.spinner("Generating response..."):
            for chunk in rag_chain.stream(query):
                result += chunk

        # Store the result and query in session state
        st.session_state.result = result
        st.session_state.query = query

        # Display the result
        st.write("**Response (Summary):**", result)

# Feedback collection
if "result" in st.session_state:
    st.write("### Feedback")

    # Feedback selection with thumbs up/thumbs down
    score = st.radio("Rate this answer:", ("👍 Thumbs Up", "👎 Thumbs Down"), key=f"feedback_{st.session_state.answer_id}")

    # Collect optional comment
    st.session_state.feedback_comment = st.text_area("Optional: Provide additional feedback")

    # Submit feedback
    if st.button("Submit Feedback"):
        feedback_score = score if score is not None else "No Response"
        save_feedback(
            st.session_state.query,
            st.session_state.result,
            feedback_score,
            st.session_state.feedback_comment,
            st.session_state.answer_id  # Add `answer_id` to the feedback
        )
        st.success("Thank you for your feedback!")

# Initialize the 'show_feedback' state if it doesn't exist
if "show_feedback" not in st.session_state:
    st.session_state["show_feedback"] = False

# Add a button to toggle feedback visibility
if st.button("Show/Hide Collected Feedback and Visualization"):
    st.session_state["show_feedback"] = not st.session_state["show_feedback"]

if st.session_state["show_feedback"]:
    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        st.write("### Collected Feedback")
        st.dataframe(feedback_df, height=300)

        # Extract date and filter data for the last 4 weeks
        feedback_df["Date"] = pd.to_datetime(feedback_df["Timestamp"]).dt.date
        today = pd.to_datetime("today").date()
        last_4_weeks = today - pd.Timedelta(weeks=4)
        recent_feedback = feedback_df[feedback_df["Date"] >= last_4_weeks]

        # Calculate thumbs up and thumbs down counts by date
        feedback_summary = recent_feedback.groupby("Date")["Score"].value_counts().unstack(fill_value=0)

        # Ensure columns for thumbs up (1) and thumbs down (0) exist
        if 1 not in feedback_summary.columns:
            feedback_summary[1] = 0
        if 0 not in feedback_summary.columns:
            feedback_summary[0] = 0

        # Sort feedback_summary by date for consistent x-axis
        feedback_summary = feedback_summary.sort_index()

        # Plotting the visualization
        st.write("### Feedback Breakdown (Last 4 Weeks)")
        plt.figure(figsize=(10, 6))

        # Bar plots for thumbs up and thumbs down
        plt.bar(feedback_summary.index, feedback_summary[1], label="👍 Thumbs Up", color="green", zorder=3, width=0.6)
        plt.bar(feedback_summary.index, feedback_summary[0], bottom=feedback_summary[1], label="👎 Thumbs Down", color="red", zorder=3, width=0.6)

        # Customizations for the chart
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title("Feedback Breakdown (Thumbs Up/Down) - Last 4 Weeks")
        plt.legend(loc="upper right")

        # Removing the top and right borders
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Adding dashed grid lines at major tick points on the y-axis
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)

        # Format the x-axis date labels for better readability
        ax.set_xticks(feedback_summary.index)
        ax.set_xticklabels(
            [date.strftime('%d-%b-%Y') for date in feedback_summary.index], 
            rotation=45, 
            ha='right'
        )

        # Display the chart
        st.pyplot(plt)
    else:
        st.warning("No feedback has been collected yet.")
