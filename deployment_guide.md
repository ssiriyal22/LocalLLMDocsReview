# 🤖 Local LLM Document Review - Complete Deployment Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation Methods](#installation-methods)
3. [System Requirements](#system-requirements)
4. [Quick Start](#quick-start)
5. [Model Management](#model-management)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

## 🎯 Overview

This application provides a complete solution for reviewing documents using open source Large Language Models (LLMs) that run entirely on your local machine. Features include:

- **Document Processing**: PDF, DOCX, TXT, and Markdown support
- **Local AI**: Uses Ollama for privacy-focused AI processing
- **Semantic Search**: Vector-based document search using embeddings
- **Interactive Chat**: Natural language queries about your documents
- **Document Summaries**: AI-generated comprehensive summaries
- **Complete Privacy**: All processing happens locally - no data leaves your machine

## 🛠 Installation Methods

### Method 1: Native Installation (Recommended)

**Linux/macOS:**
```bash
# Download the application files
git clone <repository-url> # or download and extract zip
cd llm-document-review

# Run the installer
chmod +x install.sh
./install.sh

# Start the application
./start_app.sh
```

**Windows:**
```cmd
# Download and extract the application files
# Double-click install.bat
# OR run from command prompt:
install.bat

# Start the application
start_app.bat
```

### Method 2: Docker Installation

**Prerequisites**: Docker and Docker Compose installed

```bash
# Clone repository
git clone <repository-url>
cd llm-document-review

# Linux/macOS
./docker-run.sh

# Windows
docker-run.bat
```

### Method 3: Manual Installation

If automated installers fail, follow these manual steps:

1. **Install Python 3.8+**
2. **Install Ollama** from https://ollama.ai
3. **Setup Python environment**:
   ```bash
   python -m venv llm_doc_env
   source llm_doc_env/bin/activate  # Linux/macOS
   # OR
   llm_doc_env\Scripts\activate     # Windows
   
   pip install -r requirements.txt
   ```
4. **Start services**:
   ```bash
   # Terminal 1: Start Ollama
   ollama serve
   
   # Terminal 2: Start app
   streamlit run app.py
   ```

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: 5GB free space
- **Python**: 3.8 or higher
- **Internet**: Required for initial setup and model downloads

### Recommended Specifications
- **RAM**: 16GB+ for larger models
- **CPU**: 4+ cores for better performance
- **Storage**: 20GB+ for multiple models
- **GPU**: Optional, but improves embedding performance

### Model Size Requirements
| Model | Size | RAM Usage | Performance |
|-------|------|-----------|-------------|
| llama3.2:1b | 1.3GB | 2-3GB | Fast, basic quality |
| phi3:mini | 2.2GB | 3-4GB | Balanced |
| llama3.2:3b | 2.0GB | 4-5GB | **Recommended** |
| mistral:7b | 4.1GB | 6-8GB | High quality |
| codellama:7b | 3.8GB | 6-8GB | Code/technical docs |

## 🚀 Quick Start

### 1. First Launch
1. Start the application using your chosen method
2. Open browser to http://localhost:8501
3. You should see the main interface

### 2. Install Your First Model
1. In the sidebar under "Model Management"
2. Select a recommended model (start with `llama3.2:3b`)
3. Click "Install Model" and wait for download
4. Select the installed model once available

### 3. Upload Documents
1. Use the "Document Upload" section
2. Drag and drop or browse for files
3. Supported formats: PDF, DOCX, TXT, MD
4. Wait for processing and vector index creation

### 4. Start Chatting
1. Use the chat interface to ask questions
2. Examples:
   - "What are the main points in this document?"
   - "Summarize the financial data"
   - "What does this say about project timelines?"

## 🧠 Model Management

### Recommended Models by Use Case

**General Document Review:**
- `llama3.2:3b` - Best balance of speed and quality
- `mistral:7b` - Higher quality, slower

**Technical Documents:**
- `codellama:7b` - Excellent for code and technical content
- `llama3.2:3b` - Good general technical understanding

**Quick Processing:**
- `llama3.2:1b` - Very fast, good for simple questions
- `phi3:mini` - Microsoft's efficient model

### Installing Additional Models
```bash
# From the command line
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull codellama:7b

# Or use the web interface sidebar
```

### Managing Model Storage
```bash
# List installed models
ollama list

# Remove unused models
ollama rm model_name

# Check model info
ollama show model_name
```

## 📖 Usage Guide

### Document Types and Best Practices

**PDFs:**
- Works with text-based PDFs
- Scanned documents need OCR preprocessing
- Large PDFs are automatically chunked

**Word Documents:**
- DOCX format supported
- Preserves basic formatting context
- Tables converted to text

**Text Files:**
- Plain text and Markdown
- Best performance due to clean format
- Good for transcripts and notes

### Effective Prompting

**Good Questions:**
- "What are the key financial metrics mentioned?"
- "Summarize the methodology section"
- "What risks are identified in this document?"
- "Compare the recommendations from both documents"

**Tips for Better Results:**
- Be specific about what you're looking for
- Reference specific sections if known
- Ask follow-up questions to drill down
- Use document summaries for overviews

### Chat Features

**Context Awareness:**
- System maintains conversation context
- References previous questions and answers
- Can build on earlier discussions

**Multi-Document Analysis:**
- Upload multiple related documents
- Ask comparative questions
- System searches across all uploaded content

## 🔧 Troubleshooting

### Common Issues

#### Ollama Not Starting
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start manually
ollama serve

# Check logs
ollama logs
```

#### Python Package Issues
```bash
# Recreate environment
rm -rf llm_doc_env
python -m venv llm_doc_env
source llm_doc_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Memory Issues
- Use smaller models (1b-3b parameters)
- Close other applications
- Increase virtual memory/swap
- Process smaller document chunks

#### Performance Issues
- Check available RAM during operation
- Use GPU acceleration if available
- Consider document preprocessing
- Split large documents

### Error Messages

**"Model not found"**
- Install model: `ollama pull model_name`
- Check available models: `ollama list`

**"Vector store creation failed"**
- Check document text extraction
- Verify sufficient memory
- Try smaller document chunks

**"Connection refused"**
- Ensure Ollama is running: `ollama serve`
- Check port availability (11434)
- Verify firewall settings

### Performance Optimization

**For Better Speed:**
- Use smaller models
- Reduce chunk size in processing
- Close unnecessary applications
- Use SSD storage for models

**For Better Quality:**
- Use larger models (7b+)
- Increase chunk overlap
- Use multiple context retrievals
- Fine-tune prompts

## ⚙️ Advanced Configuration

### Environment Variables
```bash
# Ollama configuration
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=/custom/model/path

# Application configuration
export STREAMLIT_SERVER_PORT=8501
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=200
```

### Custom Model Installation
```bash
# Install from Hugging Face
ollama create custom_model -f Modelfile

# Example Modelfile
FROM llama3.2:3b
PARAMETER temperature 0.1
PARAMETER top_k 40
SYSTEM "You are a document analysis expert."
```

### Integration Options

**API Mode:**
```python
# Use the document processor programmatically
from app import DocumentProcessor, VectorStore, LLMManager

processor = DocumentProcessor()
vector_store = VectorStore()
llm = LLMManager()

# Process documents via API
text = processor.process_document("document.pdf")
vector_store.create_vector_store([text])
response = llm.generate_response("What is this about?")
```

**Batch Processing:**
```bash
# Process multiple documents
python batch_process.py --input-dir ./documents --output-dir ./summaries
```

### Security Considerations

**Network Security:**
- Application runs on localhost by default
- Ollama API on localhost:11434
- No external network calls for document processing

**Data Privacy:**
- All processing happens locally
- No telemetry or usage tracking
- Documents stored temporarily only

**File System Permissions:**
- Application needs read access to document directory
- Temporary files created and cleaned automatically
- Model storage requires write access

## 📞 Support

### Getting Help
1. Check this documentation first
2. Review the troubleshooting section
3. Check application logs
4. Search for similar issues online

### Log Locations
- **Application logs**: Console output
- **Ollama logs**: `~/.ollama/logs/`
- **Streamlit logs**: Console output

### Reporting Issues
When reporting problems, include:
- Operating system and version
- Python version
- Error messages
- Steps to reproduce
- Document types being processed

---

## 🎉 You're Ready to Go!

Your Local LLM Document Review application is now set up and ready to use. Start by uploading a document and asking questions about it. The AI will help you extract insights, summaries, and answers while keeping everything completely private on your local machine.

Happy document reviewing! 📄🤖