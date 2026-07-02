# Study_Mate  
A Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain designed to turn massive, dense textbooks and research papers into interactive, searchable study guides.  
This project served as the foundational RAG pipeline for my subsequent AI Interview Coach project, focusing heavily on wide-scope text exploration, diverse context chunking, and strict constraint adherence.  

## What It Does  
Document Explanations: Allows users to upload multi-page academic PDF documents or textbooks and ask complex conceptual questions about the material.  

### Diverse Context Retrieval (MMR):   
Unlike standard similarity retrieval, this app uses Maximal Marginal Relevance (MMR) search via a Chroma DB instance. By configuring fetch_k and lambda_mult, the retriever pulls chunks that are highly relevant but visually and contextually diverse, avoiding repetitive answers when exploring broad academic topics.  

### Strict Source Anchoring:  
Powered by Mistral AI, the underlying ChatPromptTemplate enforces a strict rule: the AI can only answer using the provided context blocks. If the concept is not present in the uploaded document, it will explicitly state, "I could not find the answer in the document," eliminating hallucinations.  

### Streamlined RAG Interface:   
Built with a clean Streamlit interface optimized for split-screen reading alongside your physical or digital notes.  

## Local Setup  

1. Clone the Repository  
git clone https://github.com/Adrita03/studymate.git
cd studymate  
2. Set Up a Virtual Environment & Install Dependencies  

python -m venv .venv  
On Windows use: .venv\Scripts\activate
pip install -r requirements.txt  
4. Configure Environment Variables  
Create a .env file in the root directory and add your Mistral API key:  

Code snippet  
MISTRAL_API_KEY=your_api_key_here  
4. Run the App  

streamlit run app.py  
