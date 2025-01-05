# Focus Group RAG Summariser

This project provides a Retrieval-Augmented Generation (RAG) system for summarizing and analyzing focus group transcripts. The system uses a combination of LangChain, Ollama's Llama model, and a Chroma vector store to retrieve relevant information and generate summaries based on the input data.

## Features

- **Upload Focus Group Data**: Upload CSV files containing focus group transcripts.
- **Query and Summarization**: Ask questions related to the topics of the transcripts and receive relevant summaries.
- **Feedback Collection**: Collect user feedback on responses to improve the system's accuracy.
- **Visualization**: Visualize the collected feedback with interactive charts.

## Requirements

To run the project, you'll need the following dependencies. Install them using:

```bash
pip install -r requirements.txt
```

## Setup Instructions

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-folder>
```

2. Create a virtual environment:

```bash
python3 -m venv venv
```

3. Activate the virtual environment:
- On MacOS/Linux:
```bash
source venv/bin/activate
```
- On Windows
```bash
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the Streamlit app:
```bash
streamlit run focus_group_summariser_ollama_feedback_app.py
```

This will launch the app in your web browser.

## Usage

- **Upload CSV**: Upload a CSV file containing your focus group transcripts with columns `transcript` and `topic_name`.
- **Select a Topic**: Choose a topic from the dropdown (optional).
- **Ask a Question**: Type in your question or request a summary.
- **Get Feedback**: After receiving an answer, you can rate the response and leave feedback.
- **View Collected Feedback**: Toggle the visibility of feedback and view related statistics in the last 4 weeks.

## Feedback

After interacting with the app, users can provide feedback on the responses to improve the system. Feedback is stored in a CSV file for further analysis and visualization.

## Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For integrating language models with the RAG system.
- **Ollama**: For the Llama model powering the summarization process.
- **Chroma**: A vector store for efficient retrieval of information.
- **Matplotlib**: For feedback visualization.

## Contributing

Feel free to fork this project and submit issues or pull requests. Contributions are welcome!

## Notes

This app requires the download and local installation/hosting of **Ollama** in order to use the Llama model. Readers can access these models for free by visiting [https://ollama.com/](https://ollama.com/).
