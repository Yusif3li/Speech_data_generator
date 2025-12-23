import os
import time
import shutil
import torch
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import torchaudio

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# CONFIG
STAGING_DIR = "staging"
PROCESSED_DIR = "staging/processed"
DATASET_DIR = "final_dataset"
WAV_OUT = f"{DATASET_DIR}/wavs"

os.makedirs(WAV_OUT, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# LOAD MODELS
print("Loading PyAnnote & Whisper... (Wait ~1 min)")
# Using 'token' based on your library version
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)

# Send model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

print(f"✅ PyAnnote loaded on {device}")
# "medium" is much better for Arabic, "large" is best but slower
whisper_model = whisper.load_model("medium")

def find_diarization_data(output_object):
    """
    Hunts for the actual Annotation object inside the wrapper.
    It checks the object itself, and then all its attributes.
    """
    # 1. Check if the object itself is the annotation
    if hasattr(output_object, "itertracks"):
        return output_object
    
    # 2. Check if it's a dictionary (common in some configs)
    if isinstance(output_object, dict):
        if "annotation" in output_object:
            return output_object["annotation"]
        if "speaker_diarization" in output_object:
            return output_object["speaker_diarization"]
            
    # 3. Scan all attributes of the object (The Wrapper Case)
    # We look for ANY attribute that has the 'itertracks' method
    if hasattr(output_object, "__dict__"):
        for attr_name in dir(output_object):
            if attr_name.startswith("__"): continue
            
            try:
                attr_value = getattr(output_object, attr_name)
                if hasattr(attr_value, "itertracks"):
                    print(f"   🔎 Found diarization data in attribute: '{attr_name}'")
                    return attr_value
            except:
                continue

    # 4. If all else fails, try explicit known names from logs
    if hasattr(output_object, "speaker_diarization"):
        return output_object.speaker_diarization
        
    return None

def smart_split_audio(audio_segment, min_len=5000, max_len=8000):
    if len(audio_segment) <= max_len:
        return [audio_segment]
        
    chunks = split_on_silence(
        audio_segment,
        min_silence_len=300,
        silence_thresh=-40,
        keep_silence=200
    )
    
    final_chunks = []
    current_chunk = AudioSegment.empty()
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < max_len:
            current_chunk += chunk
        else:
            if len(current_chunk) > 1000: 
                final_chunks.append(current_chunk)
            current_chunk = chunk
            
    if len(current_chunk) > 1000:
        final_chunks.append(current_chunk)
        
    return final_chunks

def process_new_file(filepath):
    if not filepath.endswith("_full.wav"): return
    
    file_id = os.path.basename(filepath).replace("_full.wav", "")
    print(f"--> Processing New Podcast: {file_id}")
    
    try:
        # Manual Load (Fix for TorchCodec)
        waveform, sample_rate = torchaudio.load(filepath)
        inputs = {"waveform": waveform, "sample_rate": sample_rate, "uri": file_id}
        
        # Run Pipeline
        output = pipeline(inputs)
        
        # --- THE HUNTER LOGIC ---
        diarization = find_diarization_data(output)
        
        if diarization is None:
            # If we still can't find it, print everything to debug
            print(f"❌ CRITICAL: Could not find diarization data.")
            print(f"   Output Type: {type(output)}")
            print(f"   Attributes: {dir(output)}")
            return
        # ------------------------

        # Load audio for cutting
        audio = AudioSegment.from_wav(filepath)
        metadata_rows = []
        
        # 2. Iterate Speakers
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = turn.start * 1000
            end_ms = turn.end * 1000
            
            if start_ms < 0: start_ms = 0
            if end_ms > len(audio): end_ms = len(audio)
            
            speaker_audio = audio[start_ms:end_ms]
            
            # 3. Smart Cut
            small_chunks = smart_split_audio(speaker_audio)
            
            for i, chunk in enumerate(small_chunks):
                if len(chunk) < 1000: continue
                
                chunk_name = f"{file_id}_{speaker}_{int(start_ms)}_{i}.wav"
                save_path = f"{WAV_OUT}/{chunk_name}"
                chunk.export(save_path, format="wav")
                
                # 4. Transcribe
                result = whisper_model.transcribe(save_path, language="ar")
                text = result["text"].strip()
                
                if len(text) > 2:
                    text = text.replace("|", "")
                    entry = f"{chunk_name}|{text}|{speaker}"
                    metadata_rows.append(entry)
                    print(f"   Saved: {chunk_name}")

        # Write Metadata
        # 'utf-8-sig' adds the magic character that tells Excel "This is Arabic!"
        with open(f"{DATASET_DIR}/metadata.csv", "a", encoding="utf-8-sig") as f:
            f.write("\n".join(metadata_rows) + "\n")
            
        print(f"--> Finished {file_id}")

        # --- MOVE TO PROCESSED ---
        shutil.move(filepath, os.path.join(PROCESSED_DIR, os.path.basename(filepath)))
        
        script_path = filepath.replace("_full.wav", "_script.txt")
        if os.path.exists(script_path):
            shutil.move(script_path, os.path.join(PROCESSED_DIR, os.path.basename(script_path)))
            
        print(f"    Moved to processed folder")
        
    except Exception as e:
        print(f"❌ Error processing {file_id}: {e}")
        import traceback
        traceback.print_exc()

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith("_full.wav"):
            print(f"⏳ Detected {os.path.basename(event.src_path)}, waiting for write to finish...")
            file_path = event.src_path
            historical_size = -1
            while True:
                try:
                    current_size = os.path.getsize(file_path)
                    if current_size == historical_size and current_size > 0:
                        break
                    historical_size = current_size
                    time.sleep(2) 
                except FileNotFoundError:
                    return
            print(f"✅ File ready. Starting processing.")
            process_new_file(file_path)

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(Handler(), STAGING_DIR, recursive=False)
    observer.start()
    print(f"👀 Watching '{STAGING_DIR}' for new audio files...")
    
    # Check for existing files
    for filename in os.listdir(STAGING_DIR):
        if filename.endswith("_full.wav"):
            print(f"Found existing file: {filename}")
            process_new_file(os.path.join(STAGING_DIR, filename))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()