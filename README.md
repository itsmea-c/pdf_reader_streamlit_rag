# pdf_reader_streamlit_rag
Developed a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content. The system retrieves relevant text chunks from the document and uses a Large Language Model (LLM) to generate context-aware answers

Workflow:
PDF Upload: Upload PDF documents for processing.
Text Extraction: Extract text from PDF pages using PyPDF.
Text Chunking: Split extracted text into smaller, overlapping chunks to improve retrieval.
Text Embeddings: Convert document chunks into numerical vector representations using Sentence Transformers.
Vector Database: Store and retrieve document embeddings using ChromaDB.
Semantic Search: Retrieve relevant document chunks based on user questions.
LLM Integration: Generate answers using the Llama 3.2 model through Ollama.
Interactive Interface: Use Streamlit to interact with the application.
Source References: Display the page numbers associated with retrieved document content.
Input Validation and Error Handling: Handle empty files, blank questions, unsupported PDF content, and common processing errors.

Technologies Used
Python — Application development
Streamlit — Interactive web interface
PyPDF — PDF text extraction
Sentence Transformers — Text embedding generation
ChromaDB — Vector storage and similarity search
Ollama — Local LLM execution
Llama 3.2 — Language model for answer generation

Evaluation:
Raised a question about the content in the pdf uploaded and got the valid response.
Raised a question about the content which is not in the pdf uploaded and got the valid response.

Note:
Uploaded the evidence of above evaluation in the repository
1. Question and Answer.png
2. Source page evidence.png

Uploaded the Python code in the repository - stream.py



