# Importing necessary libraries
import streamlit as st
import os
from dotenv import load_dotenv

# Importing LangChain libraries
# NOTE: It is recommended to use the latest package structure (e.g., langchain_community, langchain_openai)
# to avoid deprecation warnings.
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# Load the API key from the .env file
load_dotenv()

# --- HELPER FUNCTIONS AND CACHING (FOR PERFORMANCE) ---

@st.cache_data # This decorator caches the file content, preventing repeated disk reads.
def get_document_text(file_path="information_document.txt"):
    """Reads and returns the content of the text document."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "Error: 'information_document.txt' file not found. Please ensure the file is in the correct location."

@st.cache_resource # This decorator performs costly operations (model loading, DB creation) only once.
def create_retriever(file_path="information_document.txt"):
    """Loads the document, splits it, converts it to vectors, and creates a retriever object."""
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(docs, embeddings)
    
    return db.as_retriever(search_kwargs={'k': 2})

@st.cache_resource
def create_qa_chain(_retriever):
    """Creates the Question-Answering chain, incorporating a custom prompt template."""
    
    # --- NEW SECTION START ---
    
    # Preparing the template containing the instructions for the LLM.
    # This makes the RAG system more robust.
    prompt_template = """Use the provided context snippets to answer the question.
    If you cannot find the answer within the texts, absolutely do not use your own knowledge and strictly state: "The requested information is not available in the provided document."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""

    # Create a LangChain PromptTemplate object from the template
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # --- NEW SECTION END ---
    
    llm = ChatOpenAI(temperature=0)
    
    # We inject our custom prompt using "chain_type_kwargs" when creating the chain.
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT} # This line is newly added!
    )

# --- MAIN STREAMLIT APPLICATION ---

def main():
    # Page configuration (Title and icon)
    st.set_page_config(page_title="Chat with Document", page_icon="📄")

    # Sidebar Design
    with st.sidebar:
        st.title("Project Information")
        st.info("This demo uses the Retrieval-Augmented Generation (RAG) architecture to answer questions about the text document below.")
        
        st.subheader("Document You Can Ask About")
        document_content = get_document_text()
        st.text_area("Document Content", document_content, height=300, disabled=True)

    # Main Page Content
    st.title("📄 Document Q&A Bot")
    st.markdown("Read the document in the sidebar and ask anything you're curious about its content!")

    # Initialize question counter in Session State
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0

    # Get Retriever and QA chain (this will be very fast thanks to caching)
    try:
        # NOTE: Keeping the Turkish file path as per the original code
        retriever = create_retriever(file_path="information_document.txt") 
        qa_chain = create_qa_chain(retriever)
    except Exception as e:
        st.error(f"An error occurred while starting the system: {e}")
        st.stop()

    # Display remaining question quota
    remaining_quota = 3 - st.session_state.question_count
    st.metric(label="Remaining Questions in This Session", value=f"{remaining_quota}/3")

    # Question submission form
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area("Your Question:", height=100, placeholder="E.g.: What was the project's 2025 budget?")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and user_question:
        if st.session_state.question_count < 3:
            st.session_state.question_count += 1
            
            with st.spinner("🧠 Searching and generating answer..."):
                try:
                    # Using .invoke() is recommended for newer LangChain versions, but .__call__ (or just the dict passing) still works for now
                    response = qa_chain({"query": user_question})
                    
                    st.success("Here is Your Answer:")
                    st.markdown(response['result'])
                    
                    with st.expander("📖 Source Text Used for the Answer"):
                        for source in response["source_documents"]:
                            st.write("---")
                            st.write(source.page_content)
                except Exception as e:
                    st.error(f"An error occurred while retrieving the answer: {e}")
        else:
            st.error("🚨 Your question quota is exhausted. Please refresh the browser page to ask new questions.")

if __name__ == '__main__':
    main()