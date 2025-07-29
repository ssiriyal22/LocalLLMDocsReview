#!/usr/bin/env python3
"""
Troubleshooting script for Local LLM Document Review Application
Run this script to diagnose and fix common issues
"""

import os
import sys
import subprocess
import requests
import json
from pathlib import Path

def print_header(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)

def print_status(message, status="INFO"):
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def check_python():
    """Check Python installation and version"""
    print_header("PYTHON CHECK")
    
    version = sys.version_info
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status("Python 3.8+ required!", "ERROR")
        return False
    else:
        print_status("Python version OK", "SUCCESS")
        return True

def check_virtual_env():
    """Check if virtual environment exists and is activated"""
    print_header("VIRTUAL ENVIRONMENT CHECK")
    
    venv_path = Path("llm_doc_env")
    if venv_path.exists():
        print_status("Virtual environment found", "SUCCESS")
    else:
        print_status("Virtual environment not found", "ERROR")
        print("  Run: python -m venv llm_doc_env")
        return False
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Virtual environment activated", "SUCCESS")
        return True
    else:
        print_status("Virtual environment not activated", "WARNING")
        if os.name == 'nt':  # Windows
            print("  Run: llm_doc_env\\Scripts\\activate")
        else:  # Unix-like
            print("  Run: source llm_doc_env/bin/activate")
        return False

def check_packages():
    """Check if required packages are installed"""
    print_header("PACKAGE CHECK")
    
    required_packages = [
        'streamlit', 'ollama', 'PyPDF2', 'python-docx',
        'sentence-transformers', 'faiss-cpu', 'langchain',
        'langchain-community', 'numpy', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print_status(f"{package} installed", "SUCCESS")
        except ImportError:
            print_status(f"{package} missing", "ERROR")
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        print("  Run: pip install -r requirements.txt")
        return False
    else:
        print_status("All packages installed", "SUCCESS")
        return True

def check_ollama():
    """Check Ollama installation and status"""
    print_header("OLLAMA CHECK")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_status(f"Ollama installed: {result.stdout.strip()}", "SUCCESS")
        else:
            print_status("Ollama not found in PATH", "ERROR")
            print("  Install from: https://ollama.ai")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("Ollama not found", "ERROR")
        print("  Install from: https://ollama.ai")
        return False
    
    # Check if Ollama service is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print_status("Ollama service running", "SUCCESS")
            
            # Check available models
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            if models:
                print_status(f"Available models: {', '.join(models)}", "SUCCESS")
                return True
            else:
                print_status("No models installed", "WARNING")
                print("  Install a model: ollama pull llama3.2:3b")
                return True
        else:
            print_status(f"Ollama service error: {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("Ollama service not running", "ERROR")
        print("  Start service: ollama serve")
        return False
    except Exception as e:
        print_status(f"Ollama service check failed: {e}", "ERROR")
        return False

def check_ports():
    """Check if required ports are available"""
    print_header("PORT CHECK")
    
    ports_to_check = [8501, 11434]
    
    for port in ports_to_check:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            if port == 8501:
                print_status(f"Port {port} (Streamlit) in use", "WARNING")
                print("  Streamlit may already be running")
            elif port == 11434:
                print_status(f"Port {port} (Ollama) in use", "SUCCESS")
        except requests.exceptions.ConnectionError:
            print_status(f"Port {port} available", "SUCCESS")
        except Exception as e:
            print_status(f"Port {port} check failed: {e}", "WARNING")

def check_file_structure():
    """Check if required files exist"""
    print_header("FILE STRUCTURE CHECK")
    
    required_files = [
        'app.py',
        'requirements.txt'
    ]
    
    optional_files = [
        'start_app.sh',
        'start_app.bat',
        'README.md'
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print_status(f"{file} found", "SUCCESS")
        else:
            print_status(f"{file} missing", "ERROR")
            all_good = False
    
    for file in optional_files:
        if Path(file).exists():
            print_status(f"{file} found", "SUCCESS")
        else:
            print_status(f"{file} missing", "WARNING")
    
    return all_good

def suggest_fixes():
    """Suggest common fixes"""
    print_header("QUICK FIXES")
    
    print("🔧 Common solutions:")
    print("\n1. Ollama not running:")
    print("   ollama serve")
    
    print("\n2. Missing packages:")
    print("   pip install -r requirements.txt")
    
    print("\n3. Virtual environment issues:")
    if os.name == 'nt':  # Windows
        print("   llm_doc_env\\Scripts\\activate")
    else:  # Unix-like
        print("   source llm_doc_env/bin/activate")
    
    print("\n4. Install first model:")
    print("   ollama pull llama3.2:3b")
    
    print("\n5. Start application:")
    print("   streamlit run app.py")
    
    print("\n6. Reset everything:")
    print("   rm -rf llm_doc_env")
    print("   python -m venv llm_doc_env")
    if os.name == 'nt':
        print("   llm_doc_env\\Scripts\\activate")
    else:
        print("   source llm_doc_env/bin/activate")
    print("   pip install -r requirements.txt")

def auto_fix_attempt():
    """Attempt to automatically fix common issues"""
    print_header("AUTO-FIX ATTEMPT")
    
    fixes_attempted = []
    
    # Try to start Ollama if not running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
    except requests.exceptions.ConnectionError:
        print_status("Attempting to start Ollama...", "INFO")
        try:
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            fixes_attempted.append("Started Ollama service")
            print_status("Ollama start command issued", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to start Ollama: {e}", "ERROR")
    
    # Try to install missing packages
    missing_packages = []
    required_packages = ['streamlit', 'ollama', 'PyPDF2']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_status("Attempting to install missing packages...", "INFO")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, 
                         check=True, capture_output=True)
            fixes_attempted.append(f"Installed packages: {', '.join(missing_packages)}")
            print_status("Packages installed", "SUCCESS")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to install packages: {e}", "ERROR")
    
    if fixes_attempted:
        print_status("Fixes attempted:", "INFO")
        for fix in fixes_attempted:
            print(f"  • {fix}")
        print_status("Please wait 5 seconds and run the troubleshooter again", "INFO")
    else:
        print_status("No automatic fixes available", "INFO")

def main():
    """Main troubleshooting function"""
    print("🔍 Local LLM Document Review - Troubleshooter")
    print("=" * 50)
    
    checks = [
        ("Python", check_python),
        ("Virtual Environment", check_virtual_env),
        ("Required Packages", check_packages),
        ("Ollama", check_ollama),
        ("Ports", check_ports),
        ("File Structure", check_file_structure)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if result is False:
                all_passed = False
        except Exception as e:
            print_status(f"{check_name} check failed: {e}", "ERROR")
            all_passed = False
    
    if all_passed:
        print_header("DIAGNOSIS COMPLETE")
        print_status("All checks passed! Your system should be ready.", "SUCCESS")
        print_status("If you're still having issues, try restarting the application.", "INFO")
    else:
        print_header("ISSUES FOUND")
        print_status("Some issues were detected. See suggestions below.", "WARNING")
        
        # Ask if user wants auto-fix
        try:
            user_input = input("\n🤖 Attempt automatic fixes? (y/n): ").lower().strip()
            if user_input in ['y', 'yes']:
                auto_fix_attempt()
        except KeyboardInterrupt:
            print("\nSkipping auto-fix.")
        
        suggest_fixes()

if __name__ == "__main__":
    main()
