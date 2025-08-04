# SciRAG-A Document Q&A Analysis Assistant 🤖

## What is a RAG?

**Retrieval-Augmented-Generation (RAG)** is an AI framework that allows LLMs to access and incorporate information from specific and relevant data sources when generating text, making their responses more accurate, up-to-date, and contextually relevant.

## What's this project about?

**SciRAG** is a very simple RAG system that allows you to upload PDF document(s) and have conversations about the content of the PDFs. It is built with streamlit, powered by Groq lightning fast LLMs and a conversation memory.

## Features
- Multi-PDF support: Upload and analyze multiple PDFs simultenously
- Conversation Memory: Maintains context across multiple (last 5) questions and answers
- Lightning Fast: Powered by Groq's high speed inference engine.
- Smart Document Search: Uses FAISS (Facebook AI Similarity Search) vector database for efficient similarity search
- Multiple LLM Options: Choose from "llama-3.1-8b-instant", "llama-3.3-70b-versatile", and "gemma2-9b-it".
- Source transparency: View the exact docuent chunks used to generate each answer.
- User Friendly Interface: Clean streamlit web interface.

## Demo
<img width="949" height="474" alt="image" src="https://github.com/user-attachments/assets/05c41e21-d82e-45cd-b184-a7b5ebdf8897" />

Input your Groq API key, select LLM and conversation settings

<img width="176" height="461" alt="image" src="https://github.com/user-attachments/assets/98761b89-6ca4-4ba1-9958-4146ac3dfc08" />

Upload PDF document

<img width="934" height="475" alt="image" src="https://github.com/user-attachments/assets/2f976118-8653-49c4-8506-4ce5ae35cbdc" />

<img width="945" height="476" alt="image" src="https://github.com/user-attachments/assets/8dbcedd3-fd12-4afb-a2b7-33921d09cc6f" />

SciRAG Generating response based on document content

<img width="926" height="422" alt="image" src="https://github.com/user-attachments/assets/094aaf52-0321-47c0-8f00-4a17e9ea37e9" />

## Technology Stack
- Frontend: Streanlit
- LLM Provider: Groq API
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector Databse: FAISS
- Document preprocessing: LangChain (pyPDFLoader, RecursiveCharacterTextSplitter)
- Language: Python 3.13

## Installation
```
git clone https://github.com/obinna-Muonanu/SciRAG-A-PDF-RAG-Assistant.git
cd SciRAG-A-PDF-RAG-Assistant
```
## Create a Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```
## Install dependencies
```
pip install -r requirements.txt
```
## Get your API Key
- Visit [Groq Console](https://console.groq.com/)
- Sign up for a free account
- Generate your API Key (⚠ Do not share your API Key publicly)

## Usage
### Run the application
```
streamlit run FirstLLMApp.py
```
### Configure the app
- Enter your API key on the side bar
- Select your preferred LLM
- Adjust conversation memory settings

### Upload documents
- Use the file uploader to select one or more PDF files
- Wait for processing to complete

### Start asking questions
- Type your questions in the text input
- Use suggested questions to quickly get started
- Reference previous questions naturally

## Requirements
Create a rquirements txt file with:
```
streamlit => 1.47.1
groq => 0.30.0
SentenceTranssformer =>
langchain_community =>
faiss => 1.11.0.post1
numpy => 2.3.2
torch => 2.7.1
```
## How SciRAG works

1. Documents processing: The uploaded PDFs are broken down into smaller bits called chunks.
2. Vectorization: These chunks are then converted to embeddings using the SentenceTransformer (these embeddings carry semantics i.e the context of the text. They are not just mere vectors).
3. Indexing: A vector database is created to store these embeddings. FAISS creates an efficient search index for vector similarity matching.
4. Query Processing: The user question is converted to embeddings using the SentenceTransformer. FAISS looks at the embeddings of the query and finds embeddings(documents) that are similar to the embeddings of the query.
5. Context Generation: Relevant chunks are combined with conversation history.
6. Response Generation: Groq's LLM generates contextual answers based on the document content.

## Configuration Options

### Model Selection
- Llama 3.1 8B: Fast and efficient for most tasks
- Llama 3.3 70B: Most Capable model for complex reasoning
- Gemma2 9B: Balanced performance and speed

### Memory Settings
- Adjustable conversation history (3-20 exchanges)
- Automatic context management for long conversations

## Project Structure
```
scirag-document-assistant/
├── FirstLLMApp.py         # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md            # Project documentation
└── LICENSE
```

### Acknowledgments

- Groq for providing lightning-fast LLM inference
- Streamlit for the amazing web app framework
- LangChain for document processing utilities
- FAISS for efficient similarity search
- SentenceTransformers for text embeddings
- tempfile for storing the PDFs temporarily on your local machine

### Future Enhancements

 - Support for more document formats (Word, PowerPoint, etc.)
 - Advanced citation and reference tracking
 - Export conversation history
 - Muti-lingual support
 - Integration with cloud storage providers
