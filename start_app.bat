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
