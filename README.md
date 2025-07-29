# Local LLM Document Review Application

A self-contained application for reviewing documents and PDFs using open source LLMs.

## Quick Start

1. **Start the application**: Double-click `start_app.bat`
2. **Open browser**: Go to http://localhost:8501
3. **Install a model**: In the sidebar, select and install a model (recommended: llama3.2:3b)
4. **Upload documents**: Upload your PDF, DOCX, TXT, or MD files
5. **Start chatting**: Ask questions about your documents!

## Features

- 📄 Multi-format support (PDF, DOCX, TXT, MD)
- 🤖 Local LLM integration via Ollama
- 🔍 Semantic search with vector embeddings
- 💬 Interactive document chat
- 📋 Document summarization
- 🔒 Complete privacy (everything runs locally)

## Recommended Models

**For beginners:**
- llama3.2:1b (fast, lightweight)
- phi3:mini (efficient)

**For better quality:**
- llama3.2:3b (recommended)
- mistral:7b (high quality)

## System Requirements

- RAM: 4GB minimum (8GB+ recommended)
- Storage: 2-10GB for models
- Windows 10/11

## Troubleshooting

**If Ollama fails to start:**
```
ollama serve
```

**If models fail to download:**
```
ollama pull llama3.2:3b
```

**If Python packages fail:**
```
llm_doc_env\Scripts\activate
pip install -r requirements.txt
```

## Privacy

- All processing happens locally
- No internet required after setup
- Your documents never leave your computer
- Open source models only
