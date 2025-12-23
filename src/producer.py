import os
import time
import random
import re
import sys
import struct
from google import genai
from google.genai import types
from dotenv import load_dotenv

#  SETUP
load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    print("❌ ERROR: API Key missing in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)
STAGING_DIR = "staging"
os.makedirs(STAGING_DIR, exist_ok=True)

# CONFIGURATION
TEXT_MODEL_ID = "gemini-2.5-flash"
AUDIO_MODEL_ID = "gemini-2.5-flash-preview-tts"

# TOPICS
CS_TOPICS = [
    "Binary Search Trees", "TCP vs UDP", "Restful APIs", "Docker Containers",
    "Neural Networks Backpropagation", "SQL Joins", "Big O Notation",
    "Operating System Deadlocks", "Hashing Algorithms", "Git Branching"
]

def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text).replace(" ", "_")

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Adds WAV header to Raw PCM data"""
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

def generate_random_episode(episode_number):
    topic = random.choice(CS_TOPICS)
    clean_topic = clean_filename(topic)
    episode_id = f"Ep{episode_number:03d}_{clean_topic}"
    
    print(f"\n🎬 [Test Episode] Topic: {topic}")

    # STEP 1: GENERATE SCRIPT
    print(f"   📝 Generating Script (using {TEXT_MODEL_ID})...")
    

    script_prompt = f"""
    أنت كاتب سيناريو لبرنامج "بودكاست تقني" باللهجة المصرية.
    الموضوع: "{topic}"

    الشخصيات:
    1. Speaker 1 (سارة): المذيع، تتحدث باللهجة المصرية العامية البسيطة.
    2. Speaker 2 (أحمد): الضيف (مهندس برمجيات)، يتحدث بلهجة مصرية مثقفة ويستخدم مصطلحات تقنية بالإنجليزية.

    المطلوب:
    - اكتب حواراً مدته خمس دقائق.
    - **اكتب الحوار بالكامل بحروف عربية بلكنة مصرية عامية (اللغة العربية بلكنة مصري).**
    - استخدم المصطلحات التقنية بالإنجليزية (مثل Algorithm, API) ولكن في سياق جمل عربية.
    - التزم تماماً بهذا التنسيق (مهم جداً لتوليد الصوت):
    Speaker 1: [الكلام هنا]
    Speaker 2: [الكلام هنا]
    """
    
    try:
        response = client.models.generate_content(
            model=TEXT_MODEL_ID,
            contents=script_prompt
        )
        script_text = response.text
        
        # التأكد من التنسيق قبل الحفظ
        if "Speaker 1:" not in script_text:
            print("   ⚠️ Warning: Script format might be wrong, creating backup...")
        
        with open(f"{STAGING_DIR}/{episode_id}_script.txt", "w", encoding="utf-8") as f:
            f.write(script_text)
        print("   ✅ Script Saved (Arabic).")

    except Exception as e:
        print(f"   ❌ Script Error: {e}")
        return

    time.sleep(2)

    # --- STEP 2: GENERATE AUDIO (NATIVE MULTI-SPEAKER) ---
    print(f"   🔊 Generating Audio (using {AUDIO_MODEL_ID})...")
    
    try:
        # إعدادات الصوت الأصلي (Native Config)
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 1",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Kore"
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 2",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Charon"
                                )
                            ),
                        ),
                    ]
                ),
            ),
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=script_text),
                ],
            ),
        ]

        full_audio_data = bytearray()
        first_chunk_mime = "audio/wav"

        # Stream Generation
        for chunk in client.models.generate_content_stream(
            model=AUDIO_MODEL_ID,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data:
                    full_audio_data.extend(part.inline_data.data)
                    first_chunk_mime = part.inline_data.mime_type
                    print(".", end="", flush=True)

        print("\n   💾 Saving merged audio...")
        
        final_wav = convert_to_wav(full_audio_data, first_chunk_mime)
        
        with open(f"{STAGING_DIR}/{episode_id}_full.wav", "wb") as f:
            f.write(final_wav)
            
        print(f"   ✅ Audio Saved: {episode_id}_full.wav")
        print("   🎉 Pipeline Success!")

    except Exception as e:
        print(f"\n   ❌ Audio Failed: {e}")

if __name__ == "__main__":
    print(f"🚀 Generator Started. Target Model: {AUDIO_MODEL_ID}")
    generate_random_episode(1)