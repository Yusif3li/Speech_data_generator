import os
import csv
import shutil
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

# CONFIGURATION
DATASET_DIR = "final_dataset"
CSV_PATH = f"{DATASET_DIR}/metadata.csv"
WAVS_DIR = f"{DATASET_DIR}/wavs"
REF_AHMED = "refs/ref_ahmed.wav" 
REF_SARAH = "refs/ref_sarah.wav"

# LOAD MODEL 
print("⏳ Loading Speaker Recognition Model...")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
print("✅ Model Loaded.")

def get_embedding(wav_path):
    """Extracts the audio fingerprint (embedding)."""
    signal, fs = torchaudio.load(wav_path)
    
    # SpeechBrain models require 16000Hz audio
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
        
    return verification_model.encode_batch(signal)

def main():
    # Check if reference files exist
    if not os.path.exists(REF_AHMED) or not os.path.exists(REF_SARAH):
        print("❌ Error: Reference files not found! You must create a 'refs' folder and add 'ref_ahmed.wav' and 'ref_sarah.wav'.")
        return

    print("🧠 Computing Reference Embeddings (Ahmed & Sarah)...")
    emb_ahmed = get_embedding(REF_AHMED)
    emb_sarah = get_embedding(REF_SARAH)
    print("✅ References Ready.")

    # 2. Read the old CSV
    if not os.path.exists(CSV_PATH):
        print("❌ CSV not found.")
        return

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    print(f"🔍 Scanning {len(rows)} segments...")
    
    updated_rows = []
    fixed_count = 0
    ahmed_count = 0
    sarah_count = 0

    for row in rows:
        # row structure: [id, filename, transcript, speaker_id]
        file_id, filename, text, current_speaker = row
        
        # Only fix rows that still have SPEAKER_XX labels.
        # If you already fixed them (they say ahmed/sarah), skip logic.
        if "SPEAKER_" in current_speaker:
            wav_path = os.path.join(WAVS_DIR, filename)
            
            if os.path.exists(wav_path):
                try:
                    # Get embedding for the current segment
                    emb_target = get_embedding(wav_path)
                    
                    # Compare similarity against Ahmed and Sarah
                    score_ahmed = verification_model.similarity(emb_target, emb_ahmed).item()
                    score_sarah = verification_model.similarity(emb_target, emb_sarah).item()
                    
                    # Which score is higher?
                    if score_ahmed > score_sarah:
                        new_speaker = "ahmed"
                        ahmed_count += 1
                    else:
                        new_speaker = "sarah"
                        sarah_count += 1
                    
                    # Update speaker name in the data
                    row[3] = new_speaker
                    fixed_count += 1
                    
                    if fixed_count % 100 == 0:
                        print(f"   -> Processed {fixed_count} files...")
                        
                except Exception as e:
                    print(f"⚠️ Error processing {filename}: {e}")
        
        updated_rows.append(row)

    # Save the new file
    print(f"💾 Saving updated metadata...")
    print(f"📊 Stats: Ahmed={ahmed_count}, Sarah={sarah_count}")
    
    # Create a backup of the original CSV
    shutil.copy(CSV_PATH, CSV_PATH + ".bak")
    
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_rows)

    print("✅ Done! All speakers are now fixed.")

if __name__ == "__main__":
    main()