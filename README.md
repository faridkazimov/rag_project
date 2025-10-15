#  Document Question-Answering System (RAG Project)

This project is an interactive web application that uses the Retrieval-Augmented Generation (RAG) architecture to answer natural language questions about the content of a single text document.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragproject-a9dq4num5grltk6nxhjcby.streamlit.app/)

---

## 🚀 Key Features

- **Interactive Interface:** A user-friendly web interface built using Streamlit.
- **RAG Architecture:** Prevents OpenAI LLMs from "hallucinating" by producing reliable answers based only on the provided document.
- **Local Embeddings:** Uses Hugging Face's free `all-MiniLM-L6-v2` model to convert texts into vectors, significantly reducing costs.
- **Fast Search:** Employs Facebook AI's FAISS library as the vector database for efficient searching.
- **Cost Control:** Includes a limit of 3 questions per user session.

---

## 🛠️ Technologies Used

- **Python**
- **LangChain:** The main framework for managing the RAG flow.
- **Streamlit:** Used for creating the web interface.
- **OpenAI:** Used for the Generation step (answering the question).
- **Hugging Face Transformers:** Used for text embeddings.
- **FAISS:** Used for the vector database.

---

## ⚙️ Setup and Running Locally

Follow these steps to run the project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/faridkazimov/rag_project]
    cd [rag_project]
    ```
2.  **Create and activate the virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For MacOS/Linux
    # venv\Scripts\activate  # For Windows (Command Prompt or PowerShell)
    ```
3.  **Install the necessary libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your API key:**
    Create a file named `.env` and add your key inside it in the format: `OPENAI_API_KEY="sk-..."`
5.  **Run the application:**
    ```bash
    streamlit run streamlit_app.py
    ```

