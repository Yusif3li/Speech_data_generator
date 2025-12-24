# 🇪🇬 Egyptian Arabic TTS Data Pipeline

A fully automated pipeline designed to generate, process, and annotate synthetic datasets for training Text-to-Speech (TTS) models in **Egyptian Arabic** and **Standard Arabic**.

The pipeline uses a "Producer-Consumer" architecture to generate conversational audio using **Google Gemini** and post-process it using **PyAnnote** (Diarization) and **OpenAI Whisper** (Transcription).

## 🚀 Architecture

1.  **Producer (`src/producer.py`)**:
    * **Continuous Generation:** Runs in an infinite loop to generate episodes one after another.
    * **Smart Key Rotation:** Automatically cycles through multiple Google API keys to maximize daily quotas (10 requests per key).
    * **Context:** Uses `gemini-2.5-flash` to write natural scripts on CS topics (OS, AI, Algos, etc.).
    * **Audio:** Uses `gemini-2.5-flash-preview-tts` (Native Multi-Speaker) to generate high-quality audio with distinct voices (Zephyr & Puck).

2.  **Consumer (`src/consumer.py`)**:
    * **Watcher:** Monitors the `staging` folder for new audio files.
    * **Speaker Diarization:** Uses `pyannote/speaker-diarization-3.1` to identify who is speaking and when.
    * **Segmentation:** Cuts the audio into training-ready chunks (3-8 seconds) based on silence.
    * **Transcription:** Uses `openai-whisper` (Medium model) to generate accurate Arabic text.
    * **Structured Dataset:** Saves the final wavs and updates a clean `metadata.csv` with headers.

## 🛠️ Prerequisites

* **Python 3.10+**
* **FFmpeg** installed and added to system PATH (Required for audio processing).
* **Google AI Studio API Keys** (Multiple recommended for higher volume).
* **Hugging Face Access Token** (Read permissions).
    * *Note:* You must accept the user agreement for `pyannote/speaker-diarization-3.1` on Hugging Face.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/tts-pipeline.git](https://github.com/yourusername/tts-pipeline.git)
    cd tts-pipeline
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory. You can add multiple Google keys separated by commas (no spaces):
    ```env
    # Add as many keys as you have to maximize throughput
    GENAI_API_KEYS=key1_here,key2_here,key3_here
    
    HF_TOKEN=your_hugging_face_token_here
    ```

## ▶️ Usage

Open two separate terminal windows:

**Terminal 1 (The Consumer):**
Starts the listener. It will wait for files and process them automatically.
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
| 1 | Ep001_...0.wav | أهلاً بيكم في حلقة جديدة | Speaker 1 |
| 2 | Ep001_...1.wav | النهاردة هنتكلم عن الـ Algorithms | Speaker 1 |