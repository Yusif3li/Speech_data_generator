import google.generativeai as genai
import os

# 1. SETUP: We try to get the key from environment, or you can paste it below for testing
api_key = os.getenv("GEMINI_API_KEY") 

# If you want to hardcode it just to test, uncomment the next line:
# api_key = "PASTE_YOUR_API_KEY_HERE"

if not api_key:
    print("❌ Error: No API Key found.")
else:
    genai.configure(api_key=api_key)
    
    print("🔍 Scanning for available models...")
    try:
        found_any = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ AVAILABLE: {m.name}")
                found_any = True
        
        if not found_any:
            print("❌ No text generation models found. Check your API Key permissions.")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")