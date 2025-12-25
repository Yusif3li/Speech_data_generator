import os
import csv
import shutil
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from tqdm import tqdm  # مكتبة شريط التحميل

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
    """Extracts the audio fingerprint."""
    signal, fs = torchaudio.load(wav_path)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
    return verification_model.encode_batch(signal)

def main():
    # 1. Validation
    if not os.path.exists(REF_AHMED) or not os.path.exists(REF_SARAH):
        print("❌ Error: 'refs' folder or reference audio files missing!")
        return

    print("🧠 Computing Reference Embeddings...")
    emb_ahmed = get_embedding(REF_AHMED)
    emb_sarah = get_embedding(REF_SARAH)
    
    # 2. Read CSV
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
    ahmed_count = 0
    sarah_count = 0
    fixed_count = 0

    # 3. Processing Loop with Progress Bar
    for row in tqdm(rows, desc="Fixing Speakers", unit="file"):
        
        file_id, filename, text, current_speaker = row
        
        # Check if needs fixing (SPEAKER_XX)
        if "SPEAKER_" in current_speaker:
            wav_path = os.path.join(WAVS_DIR, filename)
            
            if os.path.exists(wav_path):
                try:
                    emb_target = get_embedding(wav_path)
                    
                    score_ahmed = verification_model.similarity(emb_target, emb_ahmed).item()
                    score_sarah = verification_model.similarity(emb_target, emb_sarah).item()
                    
                    if score_ahmed > score_sarah:
                        new_speaker = "ahmed"
                        ahmed_count += 1
                    else:
                        new_speaker = "sarah"
                        sarah_count += 1
                    
                    # Update row
                    row[3] = new_speaker
                    fixed_count += 1
                        
                except Exception:
                    pass # Skip errors to keep progress bar smooth
        
        updated_rows.append(row)

    # 4. Save
    print(f"\n💾 Saving updated metadata...")
    print(f"📊 Fixed: {fixed_count} (Ahmed: {ahmed_count}, Sarah: {sarah_count})")
    
    shutil.copy(CSV_PATH, CSV_PATH + ".bak")
    
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_rows)

    print("✅ Done! Dataset is clean.")

if __name__ == "__main__":
    main()