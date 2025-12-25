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
    مهمتك تكتب حوار يبان طبيعي 100%، كأن اتنين صحاب قاعدين على القهوة بيتكلموا في الشغل، مش روبوتات بتقرأ نشرة أخبار.
    
    الجمهور المستهدف: طلبة حاسبات وهندسة في مصر. ناس بتكره "التكلف" وبتحب الكلام السالك المباشر.

    ### تفاصيل الحلقة
    - الموضوع: "{topic}"
    - التصنيف: {category}

    ### الشخصيات (الدويتو)
    1. **Speaker 1 (سارة):** المذيعة.
       - *شخصيتها:* ذكية، دمها خفيف، وبتلقط المعلومة وهي طايرة.
       - *ردود أفعالها:* **ممنوع الصويت والمبالغة.** لما تتفاجئ بتشغل مخها وتقول حاجات زي: "طب استنى.. يعني تقصد إن..."، "آه، الحتة دي لفت معايا شوية"، "تصدق منطقي".
    
    2. **Speaker 2 (أحمد):** الضيف (Senior Engineer).
       - *شخصيته:* مهندس "رايق" وتقيل. بيشرح أعقد الحاجات ببساطة ومن غير فزلكة.
       - *أسلوبه:* بيستخدم تشبيهات من الحياة (مطبخ، مواصلات، كورة). ودايماً يتأكد إن سارة فاهمة: "واخدة بالك؟"، "مجمعة معايا؟".

    ### 💎 دليل الأسلوب المصري "الأصلي" (قواعد صارمة)
    1. **قاعدة "العقلانية":** ممنوع تماماً استخدام "يا دين النبي" أو "يا خبر أبيض" إلا لو فيه مصيبة. الناس الطبيعية بتقول: "إيه ده بجد؟"، "لا والله؟"، "حلوة دي"، "تصدق فكرة".
    2. **العربي المطعّم بالإنجليزي (Educated Slang):** دخل المصطلحات التقنية في وسط الكلام العربي بتصريف مصري.
       - *غلط:* "سوف أقوم بعمل Deploy."
       - *صح:* "هعمل Deploy"، "عشان نـ Handle الـ Requests دي"، "الـ Server وقع".
    3. **الروابط الكلامية (Fillers):** استخدم دي عشان الكلام ميبقاش ناشف:
       - (بصي يا ستي / بص بقى / يعني / أصل / هو الفكرة إن / فاهمة قصدي؟ / ما هو عشان كده).
    4. **قصر النفس:** الجمل لازم تكون قصيرة ومتقطعة عشان التوليد الصوتي يطلع مظبوط. المذيعين بيقاطعوا بعض بأدب (زي: "بالظبط!"، "الله ينور عليك").

    ### 🎭 بنك ردود الأفعال (نوع في الكلام)
    *بدل ما تكرر نفس الجملة، استخدم دول:*
    - **لما يكون فيه فضول:** "طب قولي..."، "طب إيه علاقة ده بـ..."، "إشجينـا يا سيدي".
    - **لما تتلخبط:** "لا ثواني تهت منك"، "الحتة دي وقعت مني"، "مش مجمعة أوي".
    - **لما تفهم (Aha! Moment):** "آاااه دلوقتي فهمت"، "يعني زي ما يكون..."، "ده طلع حوار كبير بقى".
    - **لما توافق:** "بالظبط كده"، "جبت المفيد"، "عليك نور"، "ده الكلام المظبوط".
    - **استغراب خفيف:** "والله؟"، "بجد؟"، "أول مرة أعرف المعلومة دي".

    ### 🚫 الممنوعات (Red Lines)
    - ❌ ممنوع الفصحى نهائياً (لا تكتب: لماذا، نعم، ولكن، هذا، جداً). استخدم (ليه، أيوه، بس، ده، أوي).
    - ❌ ممنوع مقدمات الراديو القديمة: ما تبدأش بـ "أهلاً بكم أعزائي المستمعين". ادخل في الموضوع علطول بطريقة كاجوال (مثلاً: "النهارده معانا موضوع قالب الدنيا...").
    - ❌ ممنوع الرموز التعبيرية (Emojis): ده سكريبت هيتحول لصوت.

    ---
    ### 🌟 المثال الذهبي (عشان تظبط النغمة زيه)
    
    Speaker 1: بقولك إيه يا أحمد.. أنا كل ما أسمع حد بيتكلم عن الـ "Pointers" بيجيله اكتئاب، هي معقدة للدرجة دي؟
    Speaker 2: لا اكتئاب ولا حاجة.. هي بس سمعتها سابقة شوية. بصي يا ستي، تخيلي الـ Memory دي عمارة كبيرة، وكل شقة ليها "عنوان".
    Speaker 1: حلو.. العمارة دي هي الرامات يعني؟
    Speaker 2: الله ينور عليكي. المتغير العادي (Variable) هو "السكان" اللي جوه الشقة. إنما الـ Pointer؟ ده بقى مش ساكن.. ده ورقة مكتوب فيها "عنوان" الشقة.
    Speaker 1: تصدق قربت أفهم.. يعني الـ Pointer مش شايل داتا، هو شايل "مكان" الداتا؟
    Speaker 2: بالظبط كده! هو بيشاور بس. عشان كده لو العنوان غلط، البرنامج بيضرب منك ويقولك Segmentation Fault.
    Speaker 1: يا ساتر.. ده طلع هو اللي بيعمل المشاكل دي كلها! طب وليه وجع الدماغ ده ما نشتغل بـ Variables عادي؟
    Speaker 2: سؤال في الجون. تخيلي معاكي دولاب (Object) وزنه طن، وعايزة توريه لصاحبتك. الأسهل تنقلي الدولاب نفسه، ولا تديها ورقة فيها عنوان الدولاب تروح تشوفه؟
    Speaker 1: لا طبعاً العنوان أسهل بكتير.. الدولاب تقيل!
    Speaker 2: أهو ده بالظبط دور الـ Pointers.. السرعة والـ Efficiency.
    
    ---

    **المطلوب:** اكتب سكريبت كامل عن "{topic}" بنفس الروح المصرية دي.
    طول السكريبت: حوالي 800 كلمة (يكفي لمدة 5 دقائق).
    التزم تماماً بتنسيق Speaker 1 و Speaker 2.
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