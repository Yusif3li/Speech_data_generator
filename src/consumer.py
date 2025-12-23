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
# Use 'token' for newer library versions
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)

# Send model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

print(f"✅ PyAnnote loaded on {device}")
whisper_model = whisper.load_model("small") 

def smart_split_audio(audio_segment, min_len=5000, max_len=8000):
    """Splits audio on silence into chunks between 5s and 8s."""
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
            if len(current_chunk) > 1000: # Only save if > 1s
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
        # --- FIX 2: MANUAL LOADING ---
        waveform, sample_rate = torchaudio.load(filepath)
        
        # --- FIX 3: UNWRAP OUTPUT ---
        inputs = {"waveform": waveform, "sample_rate": sample_rate, "uri": file_id}
        
        # Run pipeline
        output = pipeline(inputs)
        
        # Logic to handle different return types based on your debug logs
        if hasattr(output, "speaker_diarization"):
            diarization = output.speaker_diarization
        elif hasattr(output, "annotation"):
            diarization = output.annotation
        elif isinstance(output, tuple):
            diarization = output[0]
        else:
            diarization = output

        # Verify we have the right object before looping
        if not hasattr(diarization, "itertracks"):
            print(f"⚠️ Debug: Extracted object type is {type(diarization)}")
            raise ValueError("Could not find 'itertracks' even after unwrapping.")

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
        with open(f"{DATASET_DIR}/metadata.csv", "a", encoding="utf-8") as f:
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