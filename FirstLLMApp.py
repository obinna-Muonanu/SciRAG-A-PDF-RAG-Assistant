#import necessary libraries for LLM app
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile
import os


# Initialize Groq client
# Groq is serving as our high speed engine for running AI models lightning fast.
def init_groq(api_key):
    # Initialize Groq client with the user's API Key
    return Groq(api_key=api_key)

# Create the brain of our conversation RAG system that remembers past interactions
def get_groq_response_with_memory(client, context, question, conversation_history, model_name="llama-3.1-8b-instant"):
    """
    client: The connection to Groq
    context: The relevant chunks of the text FAISS found for the question.
    question: The user's prompt.
    conversation_history: Previous Q&A pairs stored in memory.
    model_name: The LLM to use for Groq inference.
    
    """
    # Give the system its personality and instructions. i.e this is who the RAG system is and what it does even before interaction with the user
    messages = [
        {
            "role": "system",
            "content": """You are a smart and accurate document analysis assistant with outstanding attention to details and a conversation memory. Your capabilities:

1. DOCUMENT GROUNDING: Always base answers on the provided document context
2. CONVERSATION AWARENESS: Remember and reference previous exchanges when necessary
3. REFERENCE RESOLUTION: When users say "that", "it", "the topic", "based on", understand what they're referring to
4. CLARITY: If a reference is ambiguous, ask for clarification
5. ACCURACY: Never make up information not in the document or conversation

You maintain context across the conversation while staying grounded in the document."""
        }
    ]

    # Include the conversation history (up to the last 5 exchanges to manage token limits and manage conversational flow)
    for prev_q, prev_a in conversation_history[-5:]:
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})

    # Add the current question with context(chunks FAISS found relevant to the question)
    current_message = f"""Document chunks relevant to the question:
    {context}

    current question: {question}"""

    messages.append({"role": "user", "content": current_message})

    try:
        response = client.chat.completions.create(
            messages = messages,
            model = model_name,
            temperature = 0.1,
            max_tokens = 1000
        )#makes actual API call to Groq
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
    """
    Basically what the codes above do is to create a conversation RAG system with a personality and instructions
    of its capabilities. It remembers the last 5 exchanges of the conversation and uses the context when the need arises.
    It takes the user's question(this question is appended to the system's prompt) and the context and wraps it
    for the LLM to generate its response."""

@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model for converting texts to embeddings.
    The @st.cache_resource decorator caches the model so it is loaded only once."""
    return SentenceTransformer("all-MiniLM-L6-v2")

# Next, define a function to create a local vector database where our RAG can store and search embeddings using FAISS
class LocalVectorDB:
    """
    This class sets up a local vector database with 4 storage containers:
    1. embedding_model: The SentenceTransformer model used to convert texts to embeddings.
    2. chunks: The text chunks
    3. embeddings: The embeddings of the text chunks.
    4. index: The FAISS index for efficient similarity search.

    It's like setting up a library with a translator, shelves for books(chunks), a catalog system (embeddings), and a search computer (index)
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
        self.index = None

    def add_documents(self, pdfs):
        """This function takes the pdfs, extracts only the page contents in the pdf, chunks them, and converts
        them to embeddings using the embedding model. It then creates a FAISS index for efficient similarity search."""
        
        if pdfs and isinstance(pdfs[0], list):
            self.chunks = [doc.page_content for pdf in pdfs for doc in pdf]
        else:
            self.chunks = [doc.page_content for doc in pdfs]
            #print(type(self.chunks))
        embeddings = self.embedding_model.encode(self.chunks)
        self.embeddings = np.array(embeddings).astype('float32')

        
        # Here, we are getting the dimension of the embeddings (number of columns in the embeddings vector).
        # Then we create a FAISS index.
        # Here's the analogy:
        # FAISS which stands for Facebook AI Similarity search is like an empty big box which understands the shape of
        # the embeddings and how the similarity between embeddings are calculated. It then places these embeddings into the box
        # in a way that makes it easy to find similar embeddings later.
        
        
        dimension = self.embeddings.shape[1] #gets the shape of the embeddings i.e 384 for all-MiniLM-L6-v2 
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def similarity_search(self, query, k=4):
        """This function takes the user's query, converts it to an embedding. This query embedding is then
        used to search the FAISS index for the most similar chunks. It returns the top k similar chunks."""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding, k) #This is where the actual search happens
        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
        return results
    
@st.cache_data #This helps to skip processing the PDF file again if has been processed before
def load_and_split_pdf(uploaded_file):
    """The function takes the uploaded PDF file, saves it temporarily, and then uses PyPDFLoader to load the document.
    It then splits the document into chunks using RecursiveCharacterTextSplitter. Delete the temporary file from the system no matter what happens"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""] #split on paragraphs, new lines, words, and characters
        )
        chunks = text_splitter.split_documents(documents)

        return chunks

    except Exception as e:
        os.unlink(tmp_path)
        return str(e)

def manage_conversation_history(conversation_history, max_exchanges=10):
    """This functiom manages the conversation history by keeping only the last 10 conversations from the user"""
    if len(conversation_history) > max_exchanges:
        return conversation_history[-max_exchanges:]
    return conversation_history


def process_document(uploaded_files, groq_client, embedding_model):
    """
    Main document processing pipeline with conversation memory
    
    This function orchestrates the entire RAG pipeline:
    1. Loads and splits the PDF
    2. Creates embeddings and vector store
    3. Sets up the conversational Q&A interface
    4. Handles user questions with conversation context"""
    #process multiple pdfs
    st.write(f"processing {len(uploaded_files)} document(s)...")

    all_chunks = []

    # Process each PDF file
    for i, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Reading PDF {i+1} of {len(uploaded_files)}...{uploaded_file.name}"):
            chunks = load_and_split_pdf(uploaded_file)
            if chunks:
                all_chunks.extend(chunks)
                st.success(f"{uploaded_file.name} loaded successfully! Found {len(chunks)} chunks.")
            else:
                st.error(f"Could not extract any text from {uploaded_file.name}.")

    if not all_chunks:
        st.error("No text extracted from any of the uploaded PDFs")
        return
    st.success(f"All PDFs loaded successfully! Total chunks found: {len(all_chunks)}")

    # # Load and split the PDF document
    # with st.spinner("Reading PDF..."):
    #     chunks = load_and_split_pdf(uploaded_files)
    # print(type(chunks))

    # if not chunks:
    #     st.error("Could not extract any text from the PDF.")
    #     return
    # st.success("PDF Loaded successfully! Found {} chunks.".format(len(chunks)))

    # create the local vector database
    with st.spinner("Creating Vector Database locally..."):
        vector_db = LocalVectorDB(embedding_model)
        vector_db.add_documents(all_chunks)
    st.success("Vector Database created successfully!")

    # Initialize conversation history
    # The st.session_state simply helps the app to remember conversation history even after refreshing the page
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    st.session_state.vector_db = vector_db
    st.session_state.groq_client = groq_client
    st.session_state.ready_for_questions = True

    #conversation Q&A interface
    if st.session_state.get('ready_for_questions', False):
        st.header("Ask questions about the document")

        #show conversation history
        if st.session_state.conversation_history:
            st.info(f"Conversation memory: {len(st.session_state.conversation_history)} exchanges")

        # Provide example questions to help users get started
        st.write("**Try Asking:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What is the main idea of this document?"):
                st.session_state.question = "What is the main idea of this document?"
            if st.button("What are the key points?"):
                st.session_state.question = "What are the key points?"

        with col2:
            if st.button("What is the conclusion?"):
                st.session_state.question = "What is the conclusion?"
            if st.button("Can you eloborate on the author's perspective?"):
                st.session_state.question = "Can you eloborate on the author's perspective?"

        # Clear conversation history
        if st.session_state.conversation_history:
            if st.button("Clear Conversation History"):
                st.session_state.conversation_history = []
                st.success("Conversation history cleared!")
                st.rerun()

        # Main question input
        question = st.text_input("Your question:",
                                 value = st.session_state.get('question',''), #gets the question from the pre-defined questions when button is clicked
                                 key = "User Question",
                                 placeholder = "Ask anything about the document...")
        
        # preprocess the question
        if question:
            try:
                with st.spinner("Thinking..."):
                    # Find relevant chunks from the vector db using similarity search
                    relevant_chunks = st.session_state.vector_db.similarity_search(question, k=4)
                    if not relevant_chunks:
                        st.warning("No relevant information found in the document for this question.")
                        return
                    
                    # Combine the chunks into context
                    context = "\n\n".join(relevant_chunks)

                    # Get conversation history
                    conversation_history = manage_conversation_history(
                        st.session_state.conversation_history, max_exchanges=10
                    )

                    # Get response from Groq with conversation memory
                    answer = get_groq_response_with_memory(
                        st.session_state.groq_client,
                        context,
                        question,
                        conversation_history,
                        st.session_state.get("model_name", "llama-3.1-8b-instant")
                    )

                    # Append the question and answer to the conversation history
                    st.session_state.conversation_history.append((question, answer))

                # Display answer
                st.write("**Answer:**")
                st.write(answer)

                 # Show conversation history
                if len(st.session_state.conversation_history) > 1:
                    with st.expander(" Conversation History"):
                        for i, (q, a) in enumerate(st.session_state.conversation_history[:-1]):  # Exclude current
                            st.write(f"**Q{i+1}:** {q}")
                            display_answer = a[:200] + "..." if len(a) > 200 else a
                            st.write(f"**A{i+1}:** {display_answer}")
                            st.write("---")
                
                # Show source chunks for transparency and debugging
                with st.expander("📚 View source chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        # Truncate long chunks for readability
                        display_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
                        st.write(display_chunk)
                        st.write("---")
            
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    st.error("Rate limit exceeded. Please wait a monemt and try again later.")
                    st.info("Free tier limits are generous but not unlimited.")

                elif 'context_length' in str(e).lower():
                    st.error("Conversation too long. Clering older messages.")
                    st.session_state.conversation_history = st.session_state.conversation_history[-5:]
                    st.info("Try asking your questions again")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Try simplifying your question or check your API key.")


# Define the main streamlit app
def main():
    """
    Main Streamlit application with conversation memory
    
    This function:
    1. Sets up the page configuration
    2. Creates the sidebar for API key and model selection
    3. Handles file upload
    4. Orchestrates the entire application flow with conversation context
    """

    #set up streamlit page config
    st.set_page_config(
        page_title = "RAG with Conversation Memory",
        page_icon = "🤖",
        layout = "wide"
    )

    st.title("SciRAG: Your go-to Document Analysis Assistant 🤖")
    st.write("Upload a PDF document and have insightful conversations about it!")

    # sidebar for API key and model selection
    st.sidebar.header("🔧 Configurations")
    st.sidebar.write("Get your free Groq API key at: https://console.groq.com")

    # API key input
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type = "password",
        placeholder = "Enter your Groq API key here"
    )

    # Model selection
    model_options = {
        "llama-3.1-8b-instant": "Llama 3.1 8B (Fast & Smart) - 131k context",
        "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Capable) - 131k context",
        "gemma2-9b-it": "Gemma2 9B (Balanced) - 8k context"
    }

    selected_model = st.sidebar.selectbox(
        "Select LLM Model:",
        options = list(model_options.keys()),
        format_func = lambda x: model_options[x], # Display model names instead of keys
        index = 0 # Default to the first model
    )

    # Conversation settings
    st.sidebar.header("Conversation Settings")
    max_history = st.sidebar.slider(
        "Max conversation exchanges to remember:",
        min_value=3,
        max_value=20,
        value=10,
        help="Higher values provide more context but use more tokens"
    )

    # Helpful info if no API key
    if not groq_api_key:
        st.warning(" Get your free Groq API key at: https://console.groq.com. \n\n ⚠️ Do not share your API key publicly!")
        st.info("No credit card required for free tier access. Just sign up and get yours!")

        st.stop()

    # Initialize Groq client and embeding model
    groq_client = init_groq(groq_api_key)
    embedding_model = load_embedding_model()

    # Store selected model in session state
    st.session_state.model_name = selected_model
    st.session_state.max_history = max_history

    # Files upload widget
    uploaded_files = st.file_uploader(
        "Upload a single or multiple PDF document(s)",
        type=['pdf'],
        accept_multiple_files = True,
        label_visibility = "collapsed", # Hide the label for a cleaner look
        help = "Upload one or more PDF document(s) to SciRAG for analysis and have a conversation about your document(s) with SciRAG🤖"
    )

    # Process uploaded document
    if uploaded_files:
        process_document(uploaded_files, groq_client, embedding_model)
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ### 🚀 Getting Started
        1. **Get your free Groq API key** at https://console.groq.com
        2. **Enter your API key** in the sidebar
        3. **Upload one or multiple PDF document(s)** using the file uploader above
        4. **Start asking questions** - the SciRAG🤖 remembers your conversation!
        
        ### 💡 Example Questions to Try
        - "What is the main idea of this document?"
        - "Who are the key points?"
        - "What is the conclusion?"
        - "Can you elaborate on the author's perspective?"
        """)


# Application entry point
if __name__ == "__main__":
    main()