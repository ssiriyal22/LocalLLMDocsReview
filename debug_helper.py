#!/usr/bin/env python3
"""
Debug helper for the LLM Document Review app
Run this to check the current state and force model selection
"""

import requests
import json

def check_ollama_and_models():
    """Check Ollama status and available models"""
    print("🔍 Checking Ollama status...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            
            print(f"📋 Available models: {len(models)}")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            
            return models
        else:
            print(f"❌ Ollama API error: {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running")
        print("   Start with: ollama serve")
        return []
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return []

def test_model_interaction(model_name):
    """Test if we can interact with a specific model"""
    print(f"\n🧪 Testing model: {model_name}")
    
    try:
        import ollama
        
        response = ollama.generate(
            model=model_name,
            prompt="Hello, please respond with just 'Hello back!'",
            stream=False
        )
        
        print(f"✅ Model response: {response['response']}")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def create_streamlit_config():
    """Create a Streamlit config to help with debugging"""
    config_content = """
[browser]
gatherUsageStats = false

[server]
port = 8501
address = "localhost"

[logger]
level = "debug"

[theme]
base = "light"
"""
    
    import os
    config_dir = os.path.expanduser("~/.streamlit")
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "config.toml")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"✅ Created Streamlit config at: {config_path}")

def main():
    print("🔧 LLM Document Review - Debug Helper")
    print("=" * 40)
    
    # Check Ollama and models
    models = check_ollama_and_models()
    
    if models:
        print(f"\n🎯 Since you have {len(models)} model(s), here's what should happen:")
        print("1. The app should auto-select your model if there's only one")
        print("2. You should see model selection buttons in the sidebar")
        print("3. The model status should show as 'Active'")
        
        # Test the first model
        if models:
            test_model_interaction(models[0])
        
        print(f"\n💡 Quick fix - If the model isn't showing in the UI:")
        print("1. Close the Streamlit app (Ctrl+C)")
        print("2. Clear browser cache or use incognito mode")
        print("3. Restart: streamlit run app.py --server.port 8501")
        
    else:
        print("\n❌ No models found. Install one first:")
        print("   ollama pull llama3.2:3b")
    
    # Create debug config
    print(f"\n🔧 Creating debug-friendly Streamlit config...")
    create_streamlit_config()
    
    print(f"\n📱 Access the app at: http://localhost:8501")
    print(f"🔄 If issues persist, try: Ctrl+F5 to hard refresh the browser")

if __name__ == "__main__":
    main()
