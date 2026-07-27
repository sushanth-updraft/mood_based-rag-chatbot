# Mood-Aware RAG Chatbot

## Overview
A RAG-based chatbot that responds supportively based on detected facial emotions (happy, sad, angry, disgust, surprise, fear, neutral). Built as part of a larger music-aware emotion detection system.

## How it works
1. Facial emotion is detected (via separate emotion-detection module)
2. The detected emotion is used to retrieve relevant supportive content (music suggestions, coping activities, affirmations) from a curated knowledge base
3. Retrieved context is passed to an LLM (via Groq API) to generate a warm, empathetic response

## Tech stack
- **Retrieval:** sentence-transformers (all-MiniLM-L6-v2) for embedding-based similarity search
- **Generation:** Groq API (llama-3.1-8b-instant) for fast, low-cost LLM inference

## Setup
1. Clone this repo
2. Create a virtual environment:
python -m venv venv
venv\Scripts\activate
3.  Install dependencies:
pip install -r requirements.txt
4.  Set your Groq API key as an environment variable:
$env:GROQ_API_KEY = "your_key_here"
5.  Run: python rag_chatbot_py.py

## Project journey
Originally prototyped using a locally-loaded meta-llama/Llama-3.2-3B-Instruct model via Hugging Face transformers. This worked but was slow on CPU (30-60s per response) and required significant local resources. Switched to Groq's hosted API for faster inference and simpler deployment.

## Notes
- Never commit API keys directly — use environment variables
- .gitignore excludes venv/, cache files, and .env
