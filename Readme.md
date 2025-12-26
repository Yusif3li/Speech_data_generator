# 🇪🇬 Egyptian Arabic TTS Data Pipeline (High-Performance)

A fully automated, optimized pipeline designed to generate, process, and clean synthetic datasets for training Text-to-Speech (TTS) models in **Egyptian Arabic (Cairene Slang)**.

The pipeline uses a "Producer-Consumer-Cleaner" architecture to generate conversational audio using **Google Gemini**, process it at 10x speed using **Faster-Whisper**, and fix speaker identities using **Biometric Verification (SpeechBrain)**.

## 🚀 Architecture

1.  **Producer (`src/producer.py`)**:
    * **Infinite Generation:** Runs continuously to create natural Egyptian podcasts on CS topics.
    * **Smart Key Rotation:** Cycles through multiple Google API keys to maximize quota.
    * **Context:** Uses `gemini-2.5-flash` with a "Street Smart" prompt to ensure authentic slang.
    * **Audio:** Uses `gemini-2.5-flash-preview-tts` for high-quality multi-speaker audio.

2.  **Consumer (`src/consumer.py`) - *Optimized for Speed***:
    * **Watcher:** Monitors `staging` for new files.
    * **Diarization:** Uses `pyannote/speaker-diarization-3.1` (GPU accelerated).
    * **Aggressive Splitting:** Cuts audio on `150ms` silence intervals to prevent long segments.
    * **Fast Transcription:** Uses **Faster-Whisper (Large-v3/Medium)** for 5x inference speed compared to standard Whisper.
    * **In-Memory Processing:** Minimizes disk I/O for maximum throughput.

3.  **Cleaner (`src/fix_speakers.py`)**:
    * **Identity Fix:** Scans the dataset and replaces generic tags (`SPEAKER_00`) with correct names (`ahmed`, `sarah`).
    * **Biometric Matching:** Uses **SpeechBrain** to compare every segment against reference audio fingerprints.

## 🛠️ Prerequisites

* **Python 3.10+**
* **FFmpeg** installed and added to system PATH.
* **NVIDIA GPU** (Highly Recommended for speed).
* **Hugging Face Token** (Read permissions) - You must accept the user agreement for `pyannote/speaker-diarization-3.1`.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Yusif3li/Speech_data_generator.git
    cd tts-pipeline
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory:
    ```env
    GENAI_API_KEYS=key1,key2,key3
    HF_TOKEN=your_hugging_face_token
    ```

## 📂 Setup (Plug & Play)

**Reference Audio Included!** ✅
We have pre-packaged reference audio files for speaker verification in the `refs/` directory (`ref_ahmed.wav`, `ref_sarah.wav`).
**No manual recording or setup is required.** Just run the scripts.

## ▶️ Usage

**Terminal 1 (The Consumer):**
Starts the listener. It waits for files, splits them, and transcribes them (Output: `SPEAKER_00`, `SPEAKER_01`).
```bash
python src/consumer.py
```

**Terminal 2 (The Producer):**
Starts generating episodes continuously.
It generates a Script -> Audio -> Saves to staging.
To Stop: Press Ctrl+C in the terminal.
```bash
python src/producer.py
```

**Terminal 3 (The Cleaner - Periodic):** 
Run this occasionally (e.g., every 500 files) to fix speaker IDs in the CSV.
```bash
python src/fix_speakers.py
```

## 📂 Output Structure
The pipeline creates a final_dataset folder with a structured CSV ready for training:
```bash
final_dataset/
├── wavs/
│   ├── G-Ai_Studio_Ep001_Topic_Speaker1_0_1.wav
│   ├── G-Ai_Studio_Ep001_Topic_Speaker2_1500_2.wav
│   └── ...
└── metadata.csv
```
**Metadata Format (Standard CSV):**
| id | segment_name | transcript | speaker_id |
| :--- | :--- | :--- | :--- |
| 1 | Ep001_...0.wav | أهلاً بيكم في حلقة جديدة | sarah |
| 2 | Ep001_...1.wav | النهاردة هنتكلم عن الـ Algorithms | ahmed |