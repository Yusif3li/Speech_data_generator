import os
import time
import shutil
import torch
import csv
import queue
import gc 
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import torchaudio
from faster_whisper import WhisperModel

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

#  CONFIG 
STAGING_DIR = "staging"
PROCESSED_DIR = "staging/processed"
DATASET_DIR = "final_dataset"
WAV_OUT = f"{DATASET_DIR}/wavs"

# OPTIMIZATION SETTINGS
# 'medium' is good, 'small' is 2x faster and good enough for clear TTS audio.
WHISPER_SIZE = "medium" 
device_str = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device_str == "cuda" else "int8"

os.makedirs(WAV_OUT, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

processing_queue = queue.Queue()

# LOAD MODELS 
print(f"🚀 Loading Models on {device_str}...")

# PyAnnote
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
    pipeline.to(torch.device(device_str))
    print(f"✅ PyAnnote loaded on {device_str}")
except Exception as e:
    print(f"❌ Error loading PyAnnote: {e}")

# 2. Faster Whisper
print(f"⏳ Loading Faster-Whisper ({WHISPER_SIZE})...")
whisper_model = WhisperModel(WHISPER_SIZE, device=device_str, compute_type=compute_type)
print(f"✅ Faster-Whisper loaded.")

# HELPER FUNCTIONS 

def find_diarization_data(output_object):
    if hasattr(output_object, "itertracks"): 
        return output_object
    
    if isinstance(output_object, dict):
        if "annotation" in output_object: 
            return output_object["annotation"]
        if "speaker_diarization" in output_object:
            return output_object["speaker_diarization"]
        
    if hasattr(output_object, "__dict__"):
        for attr_name in dir(output_object):
            if attr_name.startswith("__"):
                continue
            try:
                attr_value = getattr(output_object, attr_name)
                if hasattr(attr_value, "itertracks"):
                    return attr_value
            except: continue
    if hasattr(output_object, "speaker_diarization"):
        return output_object.speaker_diarization
    return None

def smart_split_audio(audio_segment, min_len=2000, max_len=10000):
    # لو الملف أصلاً قصير (أقل من 10 ثواني)، رجعه زي ما هو
    if len(audio_segment) <= max_len:
        return [audio_segment]
    
    # (Aggressive Parameters) 
    # 1. min_silence_len=150: أي سكتة ربع ثانية نعتبرها فاصل (كانت 300)
    # 2. silence_thresh=-30: خلينا الحساسية أعلى شوية عشان يلقط النفس
    # 3. keep_silence=100: سيب 100 مللي ثانية بس عشان الكلام مايتقطعش جامد
    chunks = split_on_silence(
        audio_segment, 
        min_silence_len=150, 
        silence_thresh=-35, 
        keep_silence=100
    )

    final_chunks = []
    current_chunk = AudioSegment.empty()

    for chunk in chunks:
        # لو الشنك الواحد أكبر من المسموح (نادرة الحدوث بس ممكنة)، خده لوحده
        if len(chunk) > max_len:
            if len(current_chunk) > 0:
                final_chunks.append(current_chunk)
                current_chunk = AudioSegment.empty()

            final_chunks.append(chunk)
            continue

        # منطق التجميع (Merger)
        if len(current_chunk) + len(chunk) < max_len:
            current_chunk += chunk
        else:
            if len(current_chunk) > min_len: 
                final_chunks.append(current_chunk)

            current_chunk = chunk
            
    # آخر قطعة
    if len(current_chunk) > min_len:
        final_chunks.append(current_chunk)
        
    return final_chunks

def get_next_id():
    csv_path = f"{DATASET_DIR}/metadata.csv"

    if not os.path.exists(csv_path):
        return 1
    
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return 1
            
            max_id = 0
            for row in reader:
                if row and row[0].isdigit():
                    max_id = max(max_id, int(row[0]))
            return max_id + 1
    except: 
        return 1

def write_to_csv(rows):
    csv_path = f"{DATASET_DIR}/metadata.csv"
    file_exists = os.path.isfile(csv_path)
    current_id = get_next_id()
    rows_with_ids = []
    for row in rows:
        rows_with_ids.append([current_id] + row)
        current_id += 1
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "segment_name", "transcript", "speaker_id"])
        writer.writerows(rows_with_ids)

def process_new_file(filepath):
    if not filepath.endswith("_full.wav"):
        return
    
    file_base = os.path.basename(filepath).replace("_full.wav", "")
    print(f"\n--> 🚀 Processing Podcast: {file_base}")
    
    try:
        # Load Audio
        waveform, sample_rate = torchaudio.load(filepath)
        inputs = {"waveform": waveform, "sample_rate": sample_rate, "uri": file_base}
        
        # Diarization
        output = pipeline(inputs)
        diarization = find_diarization_data(output)
        
        if diarization is None:
            print(f"❌ CRITICAL: Could not find diarization data.")
            return

        # Cutting & Transcribing
        audio = AudioSegment.from_wav(filepath)
        metadata_buffer = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = turn.start * 1000
            end_ms = turn.end * 1000
            
            if start_ms < 0:
                start_ms = 0
            if end_ms > len(audio):
                end_ms = len(audio)
            
            speaker_audio = audio[start_ms:end_ms]
            small_chunks = smart_split_audio(speaker_audio)
            
            for i, chunk in enumerate(small_chunks):

                if len(chunk) < 1000: 
                    continue
                
                chunk_name = f"{file_base}_{speaker}_{int(start_ms)}_{i}.wav"
                save_path = f"{WAV_OUT}/{chunk_name}"
                chunk.export(save_path, format="wav")
                
                # FASTER WHISPER INFERENCE
                segments, info = whisper_model.transcribe(save_path, language="ar", beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()
                
                if len(text) > 2:
                    metadata_buffer.append([chunk_name, text, speaker])
                    print(f"   Saved: {chunk_name}")

        # Save Metadata
        if metadata_buffer:
            write_to_csv(metadata_buffer)
            
        print(f"--> ✅ Finished {file_base}")

        # Move to Processed
        shutil.move(filepath, os.path.join(PROCESSED_DIR, os.path.basename(filepath)))
        script_path = filepath.replace("_full.wav", "_script.txt")
        if os.path.exists(script_path):
            shutil.move(script_path, os.path.join(PROCESSED_DIR, os.path.basename(script_path)))
            
        print(f"    Moved to processed folder")

        # Cleanup Memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error processing {file_base}: {e}")
        import traceback
        traceback.print_exc()

# WATCHDOG 
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith("_full.wav"):
            print(f"⏳ New file detected: {os.path.basename(event.src_path)}. Added to Queue.")
            processing_queue.put(event.src_path)

# MAIN EXECUTION
if __name__ == "__main__":
    
    # PRIORITY: Scan for Backlog (Old Files)
    print(f"🔍 Scanning '{STAGING_DIR}' for unprocessed backlog...")
    
    backlog_files = []
    for filename in os.listdir(STAGING_DIR):
        if filename.endswith("_full.wav"):
            full_path = os.path.join(STAGING_DIR, filename)
            backlog_files.append(full_path)
    
    # Sort by creation time (Oldest First)
    backlog_files.sort(key=os.path.getmtime)
    
    for f in backlog_files:
        print(f"   📜 Found backlog file: {os.path.basename(f)} -> Added to Queue")
        processing_queue.put(f)

    # Start Watchdog (For Future Files)
    observer = Observer()
    observer.schedule(Handler(), STAGING_DIR, recursive=False)
    observer.start()
    print(f"👀 Watchdog started. Waiting for new files...")

    # Main Consumer Loop
    try:
        while True:
            try:
                # Wait for file in queue (timeout allows loop to check for interrupts)
                filepath = processing_queue.get(timeout=1)
                
                # Check if file exists (it might have been moved if we double-queued it)
                if not os.path.exists(filepath):
                    processing_queue.task_done()
                    continue

                # Wait for file write to complete (Size Check)
                print(f"⏳ Preparing to process: {os.path.basename(filepath)}")
                historical_size = -1
                while True:
                    try:
                        if not os.path.exists(filepath):
                            break
                        current_size = os.path.getsize(filepath)
                        if current_size == historical_size and current_size > 0:
                            break
                        historical_size = current_size
                        time.sleep(2)
                    except: 
                        break
        
                if os.path.exists(filepath):
                    process_new_file(filepath)
                
                processing_queue.task_done()
                
            except queue.Empty:
                pass 
            except Exception as e:
                print(f"Error in main loop: {e}")

    except KeyboardInterrupt:
        observer.stop()
    observer.join()