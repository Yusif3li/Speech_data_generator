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
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. MEGA TOPIC DICTIONARY
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

def get_used_topics():
    """Scans existing files to see which topics are already done."""
    staging_files = glob.glob(os.path.join(STAGING_DIR, "G-Ai_Studio_Ep*_*.wav"))
    processed_files = glob.glob(os.path.join(PROCESSED_DIR, "G-Ai_Studio_Ep*_*.wav"))
    all_files = staging_files + processed_files
    
    used_clean_topics = set()
    for f in all_files:
        base = os.path.basename(f)
        try:
            parts = base.split("_Ep") 
            if len(parts) > 1:
                rest = parts[1] 
                rest_parts = rest.split("_", 1)
                if len(rest_parts) > 1:
                    topic_part = rest_parts[1] 
                    # Handle new format with duration inside name
                    # Remove duration part if exists like "_Dur360s"
                    topic_clean = re.sub(r"_Dur\d+s", "", topic_part)
                    topic_clean = topic_clean.replace("_full.wav", "")
                    used_clean_topics.add(topic_clean)
        except:
            continue
    return used_clean_topics

def get_next_episode_number():
    staging_files = glob.glob(os.path.join(STAGING_DIR, "G-Ai_Studio_Ep*_*.wav"))
    processed_files = glob.glob(os.path.join(PROCESSED_DIR, "G-Ai_Studio_Ep*_*.wav"))
    all_files = staging_files + processed_files
    
    if not all_files:
        return 1
    
    max_num = 0
    for f in all_files:
        try:
            base = os.path.basename(f)
            parts = base.split("_")
            for part in parts:
                if part.startswith("Ep") and part[2:].isdigit():
                    max_num = max(max_num, int(part[2:]))
        except:
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
    except: 
        pass

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
    # Get History
    used_clean = get_used_topics()
    
    # Filter Available Topics
    available_topics = []
    for cat, topics in CS_TOPICS.items():
        for t in topics:
            if clean_filename(t) not in used_clean:
                available_topics.append((cat, t))
    
    if not available_topics:
        print("🎉 CONGRATULATIONS! You have generated episodes for ALL topics!")
        sys.exit(0)

    category, topic = random.choice(available_topics)
    ep_num = get_next_episode_number()
    clean_topic = clean_filename(topic)
    
    # Initial base name
    temp_base_name = f"G-Ai_Studio_Ep{ep_num:03d}_{clean_topic}"
    
    print(f"\n🎬 [Generating Ep {ep_num}] Category: {category} | Topic: {topic}")
    
    client = key_manager.get_client()

    # SCRIPT 
    print(f"   📝 Writing Script...")
    script_prompt = f"""
    ### الدور والمهمة
    أنت كاتب سيناريو محترف لبرنامج "بودكاست تقني" باللهجة المصرية القاهرية (Cairene Slang).
    مهمتك تكتب حوار طبيعي جداً، دمه خفيف، وفيه روح "الصحوبية" المصرية.

    الجمهور المستهدف: طلبة حاسبات وهندسة (CS Students).

    ### تفاصيل الحلقة
    - الموضوع: "{topic}"
    - التصنيف: {category}

    ### الشخصيات
    1. **Speaker 1 (سارة):** المذيعة (وتمثل دور الطالبة).
       - *شخصيتها:* لسه بتتعلم (Junior)، دمها خفيف، بتسأل الأسئلة "البسيطة" اللي بتيجي في بال أي حد مش فاهم.
       - *دورها:* بتبدأ الحلقة، بتقدم الضيف، ولما بتسمع مصطلح كبير بتوقف أحمد وتقوله "استنى بس فهمنا بالراحة".
    
    2. **Speaker 2 (أحمد):** الضيف (Senior Engineer).
       - *شخصيته:* خبير، رايق، وباله طويل. بيشرح التكنولوجيا بأمثلة من الحياة (أكل، مواصلات، مواقف يومية).

    ### 💎 دليل الأسلوب (Style Guide)
    1. **المقدمة (Intro):** ابدأ الحلقة بشكل طبيعي. سارة ترحب بالمستمعين بحماس، تقول اسم البرنامج، وتقدم موضوع الحلقة والضيف أحمد.
    2. **التعبيرات والقفشات (Humor):**
       - مش ممنوع تستخدم "يا دين النبي" أو "يا خبر أبيض"، بالعكس دي مطلوبة لإضافة روح، **لكن بذكاء!**
       - **القاعدة:** استخدم التعبيرات القوية دي لما تكون المعلومة "صادمة" أو "معقدة بجد". غير كده، نوع في كلامك: (يا نهار ابيض، يا ساتر، إيه الحلاوة دي، لا والله؟، تصدق فكرة).
       - *المهم:* بلاش تكرر نفس الكلمة كل سطرين عشان المستمع ما يزهقش.
    3. **العربي المطعّم بالإنجليزي:** المصطلحات التقنية زي ما هي (API, Deadlock, RAM) وسط الكلام العربي.
    4. **التبسيط:** أحمد لازم يشرح بتشبيهات (Analogies). ممنوع الشرح الأكاديمي الناشف.

    ### 🎭 بنك ردود أفعال سارة (عشان التنويع)
    - **لما تتخض من صعوبة معلومة:** "يا دين النبي! إيه التعقيد ده كله؟" / "يا ساتر.. ده طلع موال كبير".
    - **لما تسأل بفضول:** "طب استنى يا أحمد.. يعني أفهم من كده إن..." / "طب وده لازمته إيه في الحياة؟".
    - **لما تفهم:** "آاااه دلوقتي نورت!" / "تصدق، كده وضحت".
    - **لما تهزر:** "يا عم ارحمنا بقى من المصطلحات دي" / "ده إحنا كنا عايشين في مية البطيخ".

    ### 🚫 الممنوعات (Red Lines)
    - ❌ ممنوع الفصحى نهائياً (لا تكتب: لماذا، نعم، ولكن، هذا).
    - ❌ ممنوع الجمل الطويلة جداً (عشان النفس).
    - ❌ ممنوع الرموز التعبيرية (Emojis).

    ---
    ### 🌟 شكل الحوار المتوقع (Tone Reference)
    
    Speaker 1: أهلاً بيكم يا شباب في حلقة جديدة من بودكاست "كود وكلام". معاكم سارة، والنهارده معانا موضوع عامل قلق لناس كتير.. "الـ Pointers". منورنا كالعادة الباشمهندس أحمد.
    Speaker 2: أهلاً بيكي يا سارة، وأهلاً بكل اللي بيسمعونا. وماتقلقيش، هنفك عقدة الـ Pointers دي خالص النهارده.
    Speaker 1: يا مسهل.. أنا أصلاً أول ما بسمع الكلمة دي بيجيلي هبوط. هي ليه الناس معقداها كده؟
    Speaker 2: هي مش معقدة، هي بس محتاجة تخيل. بصي يا ستي، تخيلي الـ Memory دي عمارة كبيرة، وكل شقة ليها "عنوان".
    Speaker 1: حلو.. العمارة دي هي الرامات يعني؟
    Speaker 2: الله ينور عليكي. المتغير العادي (Variable) هو "السكان" اللي جوه الشقة. إنما الـ Pointer؟ ده بقى ورقة مكتوب فيها "عنوان" الشقة بس.
    Speaker 1: يا دين النبي! يعني هو مش شايل داتا، هو شايل "مكان" الداتا؟ طب وليه اللفة دي؟
    Speaker 2: سؤال في الجون. تخيلي معاكي دولاب وزنه طن، وعايزة توريه لصاحبتك. الأسهل تنقلي الدولاب نفسه، ولا تديها ورقة فيها عنوان الدولاب تروح تشوفه؟
    Speaker 1: لا طبعاً العنوان أسهل.. الدولاب تقيل جداً!
    Speaker 2: أهو ده بالظبط دور الـ Pointers.. السرعة والـ Efficiency عشان مننقلش داتا كتير.
    
    ---

    **المطلوب:** اكتب سكريبت كامل عن "{topic}" بنفس الروح دي.
    طول السكريبت: حوالي 800 كلمة (يكفي لمدة 5 دقائق).
    التزم بتنسيق Speaker 1 و Speaker 2.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=script_prompt
        )
        script_text = response.text
        
        # Save script initially
        script_path = f"{STAGING_DIR}/{temp_base_name}_script.txt"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_text)
        print("   ✅ Script Saved.")

    except Exception as e:
        print(f"   ❌ Script Error: {e}")
        if "429" in str(e):
            print("   ⚠️ Quota exceeded during script. Rotating key")
            key_manager.rotate_key()
            return generate_episode()
        return

    time.sleep(2)

    # AUDIO
    print(f"   🔊 Generating Audio")
    
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
        
        # 24000 Hz * 1 channel * 2 bytes (16-bit) = 48000 bytes per second
        duration_sec = len(full_audio_data) / 48000
        duration_str = f"Dur{int(duration_sec)}s"
        
        # Create the Final Filename with Duration
        final_base_name = f"{temp_base_name}_{duration_str}"
        final_wav_path = f"{STAGING_DIR}/{final_base_name}_full.wav"
        
        # Save Audio with new name
        with open(final_wav_path, "wb") as f:
            f.write(final_wav)
            
        # IMPORTANT: Rename the script file to match the new audio name
        # so the consumer can find it later
        final_script_path = f"{STAGING_DIR}/{final_base_name}_script.txt"
        if os.path.exists(script_path):
            os.rename(script_path, final_script_path)

        print(f"   ✅ Audio Saved: {os.path.basename(final_wav_path)}")
        key_manager.increment_usage()

    except Exception as e:
        print(f"\n   ❌ Audio Failed: {e}")
        if "429" in str(e):
             print("   ⚠️ Quota exceeded during audio. Rotating key")
             key_manager.rotate_key()

if __name__ == "__main__":
    print("🚀 Generator Started.")
    try:
        while True:
            generate_episode()
            print("⏳ Waiting 10 seconds before next episode")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 Generator Stopped manually.")