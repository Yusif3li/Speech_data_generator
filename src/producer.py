import os
import time
import random
import re
import sys
import struct
import glob
from itertools import cycle
from google import genai
from google.genai import types
from dotenv import load_dotenv

# CONFIGURATION 
STAGING_DIR = "staging"
PROCESSED_DIR = os.path.join(STAGING_DIR, "processed")

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True) # Ensure history folder exists

# MEGA TOPIC DICTIONARY
CS_TOPICS = {
    "Data Structures": [
        "Binary Search Trees", "Hash Maps & Collisions", "Linked Lists vs Arrays", 
        "Heaps and Priority Queues", "Tries & Prefix Trees", "Graph Adjacency Matrix vs List",
        "Stack vs Queue Applications", "B-Trees in Databases"
    ],
    "Algorithms": [
        "Big O Notation", "Merge Sort vs Quick Sort", "Dijkstra's Algorithm", 
        "Dynamic Programming: Knapsack", "Depth First Search (DFS)", "Breadth First Search (BFS)",
        "Binary Search Logic", "A* Pathfinding"
    ],
    "Operating Systems": [
        "Process vs Thread", "Deadlocks & Prevention", "Memory Paging & Segmentation", 
        "CPU Scheduling Algorithms", "Semaphores vs Mutex", "Virtual Memory", "Context Switching"
    ],
    "Networking": [
        "TCP vs UDP Handshake", "HTTP vs HTTPS", "DNS Resolution Process", 
        "OSI Model Layers", "Load Balancing Strategies", "WebSockets vs REST", "CDN Fundamentals"
    ],
    "Databases": [
        "SQL Joins Explained", "ACID Properties", "NoSQL vs SQL", "Database Indexing", 
        "Normalization forms", "Sharding vs Replication", "Redis Caching"
    ],
    "AI & ML": [
        "Neural Networks Backpropagation", "Convolutional Neural Networks (CNN)", 
        "Transformers & Attention", "Supervised vs Unsupervised Learning", "Overfitting vs Underfitting", 
        "Gradient Descent", "Reinforcement Learning Basics"
    ]
}

# KEY MANAGER CLASS 
class KeyManager:
    def __init__(self):
        load_dotenv()
        keys_str = os.getenv("GENAI_API_KEYS", "")
        if not keys_str:
            print("❌ ERROR: GENAI_API_KEYS missing in .env")
            sys.exit(1)
            
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.keys:
            print("❌ ERROR: No valid keys found in GENAI_API_KEYS")
            sys.exit(1)
            
        self.key_cycle = cycle(self.keys)
        self.current_key = next(self.key_cycle)
        self.usage_count = 0
        self.max_usage_per_key = 10 
        
        print(f"🔑 Loaded {len(self.keys)} API Keys. Rotation limit: {self.max_usage_per_key} runs per key.")
        self.client = self._create_client()

    def _create_client(self):
        return genai.Client(api_key=self.current_key)

    def get_client(self):
        if self.usage_count >= self.max_usage_per_key:
            print(f"⚠️ Key limit ({self.max_usage_per_key}) reached. Rotating key...")
            self.rotate_key()
        return self.client

    def rotate_key(self):
        old_key = self.current_key[-4:]
        self.current_key = next(self.key_cycle)
        self.usage_count = 0
        self.client = self._create_client()
        print(f"🔄 Switched Key: ...{old_key} ➔ ...{self.current_key[-4:]}")

    def increment_usage(self):
        self.usage_count += 1
        print(f"   [Key Usage: {self.usage_count}/{self.max_usage_per_key}]")

key_manager = KeyManager()

# HELPER FUNCTIONS 

def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text).replace(" ", "_")

def get_next_episode_number():
    """Scans STAGING AND PROCESSED directories to find the true next number."""
    # 1. Check currently staging files
    staging_files = glob.glob(os.path.join(STAGING_DIR, "G-Ai_Studio_Ep*_*.wav"))
    
    # 2. Check historically processed files
    processed_files = glob.glob(os.path.join(PROCESSED_DIR, "G-Ai_Studio_Ep*_*.wav"))
    
    all_files = staging_files + processed_files
    
    if not all_files:
        return 1
    
    max_num = 0
    for f in all_files:
        try:
            # Extract number: "G-Ai_Studio_Ep005_Topic.wav" -> "Ep005" -> 5
            base = os.path.basename(f)
            parts = base.split("_")
            # We look for the part that starts with "Ep"
            for part in parts:
                if part.startswith("Ep") and part[2:].isdigit():
                    num = int(part[2:])
                    if num > max_num:
                        max_num = num
        except (IndexError, ValueError):
            continue
            
    return max_num + 1

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    bits_per_sample = 16
    rate = 24000
    try:
        parts = mime_type.split(";")
        for param in parts:
            if "rate=" in param:
                rate = int(param.split("=")[1])
    except: pass

    num_channels = 1
    data_size = len(audio_data)
    chunk_size = 36 + data_size
    byte_rate = rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        rate, byte_rate, block_align, bits_per_sample, b"data", data_size
    )
    return header + audio_data

# MAIN GENERATOR 

def generate_episode():
    # Pick a Topic
    category = random.choice(list(CS_TOPICS.keys()))
    topic = random.choice(CS_TOPICS[category])
    
    # Dynamic Episode Number
    ep_num = get_next_episode_number()
    clean_topic = clean_filename(topic)
    
    # "G-Ai studio_EP(Episode number)- the subject"
    file_base_name = f"G-Ai_Studio_Ep{ep_num:03d}_{clean_topic}"
    
    print(f"\n🎬 [Generating Ep {ep_num}] Category: {category} | Topic: {topic}")
    
    client = key_manager.get_client()

    # SCRIPT 
    print(f"   📝 Writing Script...")
    script_prompt = f"""
    أنت كاتب سيناريو لبرنامج "بودكاست تقني" باللهجة المصرية.
    الموضوع: "{topic}" ({category})

    الشخصيات:
    1. Speaker 1 (سارة): المذيع، تتحدث باللهجة المصرية العامية البسيطة.
    2. Speaker 2 (أحمد): الضيف (مهندس برمجيات)، يتحدث بلهجة مصرية مثقفة ويستخدم مصطلحات تقنية بالإنجليزية.

    المطلوب:
    - اكتب حواراً مدته ستة دقائق.
    - **اكتب الحوار بالكامل بحروف عربية بلكنة مصرية عامية (اللغة العربية بلكنة مصري).**
    - استخدم المصطلحات التقنية بالإنجليزية (مثل Algorithm, API) ولكن في سياق جمل عربية.
    - التزم تماماً بهذا التنسيق (مهم جداً لتوليد الصوت):
    Speaker 1: [الكلام هنا]
    Speaker 2: [الكلام هنا]
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=script_prompt
        )
        script_text = response.text
        
        with open(f"{STAGING_DIR}/{file_base_name}_script.txt", "w", encoding="utf-8") as f:
            f.write(script_text)
        print("   ✅ Script Saved.")

    except Exception as e:
        print(f"   ❌ Script Error: {e}")
        if "429" in str(e):
            print("   ⚠️ Quota exceeded during script. Rotating key...")
            key_manager.rotate_key()
            return generate_episode()
        return

    time.sleep(2)

    # AUDIO 
    print(f"   🔊 Generating Audio...")
    
    try:
        client = key_manager.get_client()
        
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 1",
                            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 2",
                            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")),
                        ),
                    ]
                ),
            ),
        )

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=script_text)])]
        full_audio_data = bytearray()
        first_chunk_mime = "audio/wav"

        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash-preview-tts",
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data:
                    full_audio_data.extend(part.inline_data.data)
                    first_chunk_mime = part.inline_data.mime_type
                    print(".", end="", flush=True)

        print("\n   💾 Saving Audio...")
        final_wav = convert_to_wav(full_audio_data, first_chunk_mime)
        
        with open(f"{STAGING_DIR}/{file_base_name}_full.wav", "wb") as f:
            f.write(final_wav)
            
        print(f"   ✅ Audio Saved: {file_base_name}_full.wav")
        key_manager.increment_usage()

    except Exception as e:
        print(f"\n   ❌ Audio Failed: {e}")
        if "429" in str(e):
             print("   ⚠️ Quota exceeded during audio. Rotating key...")
             key_manager.rotate_key()

if __name__ == "__main__":
    print("🚀 Generator Started (Multi-Key Support + History Check).")
    try:
        while True:
            generate_episode()
            print("⏳ Waiting 10 seconds before next episode...")
            time.sleep(10) 
    except KeyboardInterrupt:
        print("\n🛑 Generator Stopped manually.")