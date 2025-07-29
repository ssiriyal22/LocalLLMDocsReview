@echo off
REM install.bat - Windows Installation script for Local LLM Document Review Application

echo 🚀 Installing Local LLM Document Review Application...
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ from https://python.org
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Create requirements.txt
echo 📄 Creating requirements.txt...
(
echo # Core application dependencies
echo streamlit^>=1.28.0
echo ollama^>=0.1.7
echo.
echo # Document processing
echo PyPDF2^>=3.0.1
echo python-docx^>=0.8.11
echo.
echo # Vector store and embeddings
echo sentence-transformers^>=2.2.2
echo faiss-cpu^>=1.7.4
echo langchain^>=0.1.0
echo langchain-community^>=0.0.10
echo.
echo # Additional utilities
echo numpy^>=1.24.0
echo requests^>=2.31.0
) > requirements.txt

echo ✅ requirements.txt created

REM Create virtual environment
echo 🐍 Creating Python virtual environment...
python -m venv llm_doc_env
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment and install packages
echo 📦 Installing Python packages...
call llm_doc_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python packages
    pause
    exit /b 1
)

echo ✅ Python packages installed

REM Check if Ollama is installed
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Ollama not found. Please install Ollama:
    echo    1. Go to: https://ollama.ai/download/windows
    echo    2. Download and install Ollama for Windows
    echo    3. Press Enter when done...
    pause
    
    REM Check again after user input
    where ollama >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Ollama still not found. Please install it manually.
        pause
        exit /b 1
    )
)

echo ✅ Ollama found

REM Create startup script
echo 🚀 Creating startup script...
(
echo @echo off
echo echo Starting Local LLM Document Review Application...
echo.
echo REM Start Ollama in background
echo start /B ollama serve
echo.
echo REM Wait for Ollama to start
echo timeout /t 5 /nobreak ^> nul
echo.
echo REM Activate virtual environment and run app
echo call llm_doc_env\Scripts\activate
echo streamlit run app.py --server.port 8501 --server.address localhost
echo.
echo pause
) > start_app.bat

echo ✅ Created start_app.bat

REM Create README
echo 📖 Creating README.md...
(
echo # Local LLM Document Review Application
echo.
echo A self-contained application for reviewing documents and PDFs using open source LLMs.
echo.
echo ## Quick Start
echo.
echo 1. **Start the application**: Double-click `start_app.bat`
echo 2. **Open browser**: Go to http://localhost:8501
echo 3. **Install a model**: In the sidebar, select and install a model ^(recommended: llama3.2:3b^)
echo 4. **Upload documents**: Upload your PDF, DOCX, TXT, or MD files
echo 5. **Start chatting**: Ask questions about your documents!
echo.
echo ## Features
echo.
echo - 📄 Multi-format support ^(PDF, DOCX, TXT, MD^)
echo - 🤖 Local LLM integration via Ollama
echo - 🔍 Semantic search with vector embeddings
echo - 💬 Interactive document chat
echo - 📋 Document summarization
echo - 🔒 Complete privacy ^(everything runs locally^)
echo.
echo ## Recommended Models
echo.
echo **For beginners:**
echo - llama3.2:1b ^(fast, lightweight^)
echo - phi3:mini ^(efficient^)
echo.
echo **For better quality:**
echo - llama3.2:3b ^(recommended^)
echo - mistral:7b ^(high quality^)
echo.
echo ## System Requirements
echo.
echo - RAM: 4GB minimum ^(8GB+ recommended^)
echo - Storage: 2-10GB for models
echo - Windows 10/11
echo.
echo ## Troubleshooting
echo.
echo **If Ollama fails to start:**
echo ```
echo ollama serve
echo ```
echo.
echo **If models fail to download:**
echo ```
echo ollama pull llama3.2:3b
echo ```
echo.
echo **If Python packages fail:**
echo ```
echo llm_doc_env\Scripts\activate
echo pip install -r requirements.txt
echo ```
echo.
echo ## Privacy
echo.
echo - All processing happens locally
echo - No internet required after setup
echo - Your documents never leave your computer
echo - Open source models only
) > README.md

echo ✅ README.md created

echo.
echo 🎉 Installation completed successfully!
echo.
echo Next steps:
echo 1. Double-click start_app.bat to start the application
echo 2. Open http://localhost:8501 in your browser
echo 3. Install a model from the sidebar ^(recommended: llama3.2:3b^)
echo 4. Upload your documents and start chatting!
echo.
echo 📖 See README.md for detailed instructions
echo.
pause
