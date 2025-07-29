#!/bin/bash
# install.sh - Installation script for Local LLM Document Review Application

set -e  # Exit on any error

echo "🚀 Installing Local LLM Document Review Application..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
        print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama already installed"
        return
    fi
    
    case $OS in
        "linux"|"macos")
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
        "windows")
            print_warning "Please download and install Ollama for Windows from: https://ollama.ai/download/windows"
            print_warning "Press Enter after installing Ollama..."
            read
            ;;
    esac
    
    # Verify installation
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully"
    else
        print_error "Ollama installation failed"
        exit 1
    fi
}

# Create virtual environment and install Python dependencies
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    $PYTHON_CMD -m venv llm_doc_env
    
    # Activate virtual environment
    case $OS in
        "windows")
            source llm_doc_env/Scripts/activate
            ;;
        *)
            source llm_doc_env/bin/activate
            ;;
    esac
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python packages..."
    pip install -r requirements.txt
    
    print_success "Python environment setup complete"
}

# Create requirements.txt
create_requirements() {
    print_status "Creating requirements.txt..."
    
    cat > requirements.txt << EOF
# Core application dependencies
streamlit>=1.28.0
ollama>=0.1.7

# Document processing
PyPDF2>=3.0.1
python-docx>=0.8.11

# Vector store and embeddings
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
langchain>=0.1.0
langchain-community>=0.0.10

# Additional utilities
numpy>=1.24.0
requests>=2.31.0

# Optional: GPU support for faster embeddings
# torch>=2.0.0
# faiss-gpu>=1.7.4  # Use this instead of faiss-cpu for GPU support
EOF

    print_success "requirements.txt created"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    case $OS in
        "windows")
            cat > start_app.bat << 'EOF'
@echo off
echo Starting Local LLM Document Review Application...

REM Start Ollama in background
start /B ollama serve

REM Wait for Ollama to start
timeout /t 5 /nobreak > nul

REM Activate virtual environment and run app
call llm_doc_env\Scripts\activate
streamlit run app.py --server.port 8501 --server.address localhost

pause
EOF
            print_success "Created start_app.bat"
            ;;
        *)
            cat > start_app.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Local LLM Document Review Application..."

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5
fi

# Activate virtual environment
source llm_doc_env/bin/activate

# Start the application
echo "Starting Streamlit application..."
streamlit run app.py --server.port 8501 --server.address localhost
EOF
            chmod +x start_app.sh
            print_success "Created start_app.sh"
            ;;
    esac
}

# Create README file
create_readme() {
    print_status "Creating README.md..."
    
    cat > README.md << 'EOF'
# Local LLM Document Review Application

A self-contained application for reviewing documents and PDFs using open source Large Language Models (LLMs) running locally via Ollama.

## Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, and Markdown files
- 🤖 **Local LLM Integration**: Uses Ollama for privacy-focused AI
- 🔍 **Semantic Search**: Vector-based document search using embeddings
- 💬 **Interactive Chat**: Ask questions about your documents
- 📋 **Document Summaries**: Generate comprehensive summaries
- 🔒 **Privacy First**: All processing happens locally

## Quick Start

### 1. Installation
Run the installation script:
```bash
# Linux/macOS
./install.sh

# Windows
install.bat
```

### 2. Start the Application
```bash
# Linux/macOS
./start_app.sh

# Windows
start_app.bat
```

### 3. Install Your First Model
1. Open the application in your browser (usually http://localhost:8501)
2. In the sidebar, select a recommended model (e.g., "llama3.2:3b")
3. Click "Install Model" and wait for download to complete
4. Select the installed model

### 4. Upload Documents
1. Upload your PDF, DOCX, TXT, or MD files
2. Wait for processing and vector index creation
3. Start asking questions about your documents!

## Recommended Models

### Lightweight (Good for most documents):
- **llama3.2:1b** - Very fast, good for basic tasks
- **phi3:mini** - Microsoft's efficient model

### Balanced (Best overall performance):
- **llama3.2:3b** - Recommended for most users
- **mistral:7b** - Excellent reasoning capabilities

### Specialized:
- **codellama:7b** - Great for technical documents

## System Requirements

- **RAM**: 4GB minimum (8GB+ recommended for larger models)
- **Storage**: 2-10GB for models (varies by model size)
- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux

## Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama manually
ollama serve

# Pull a model manually
ollama pull llama3.2:3b
```

### Python Environment Issues
```bash
# Recreate virtual environment
rm -rf llm_doc_env
python3 -m venv llm_doc_env
source llm_doc_env/bin/activate  # Linux/macOS
# OR
llm_doc_env\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Performance Tips
- Use smaller models (1b-3b parameters) for faster responses
- Close other applications to free up RAM
- For large documents, consider splitting them into smaller files

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     Ollama      │    │   Vector Store  │
│   Frontend      │◄──►│   LLM Engine    │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └─────────────►│  Document       │◄─────────────┘
                        │  Processor      │
                        └─────────────────┘
```

## Privacy & Security

- **All data stays local** - No external API calls for document processing
- **No internet required** - After initial setup, works offline
- **Open source models** - Transparent and auditable AI
- **No data collection** - Your documents never leave your machine

## Contributing

This is a self-contained application. For issues or improvements:
1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

MIT License - Feel free to modify and distribute
EOF

    print_success "README.md created"
}

# Main installation function
main() {
    echo "🤖 Local LLM Document Review Application Installer"
    echo "=================================================="
    
    detect_os
    check_python
    create_requirements
    install_ollama
    setup_python_env
    create_startup_script
    create_readme
    
    echo ""
    print_success "Installation completed successfully! 🎉"
    echo ""
    echo "Next steps:"
    echo "1. Start the application:"
    case $OS in
        "windows")
            echo "   start_app.bat"
            ;;
        *)
            echo "   ./start_app.sh"
            ;;
    esac
    echo "2. Open http://localhost:8501 in your browser"
    echo "3. Install a model from the sidebar (recommended: llama3.2:3b)"
    echo "4. Upload your documents and start chatting!"
    echo ""
    echo "📖 See README.md for detailed instructions and troubleshooting"
}

# Run main function
main "$@"
