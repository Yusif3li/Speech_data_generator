# 🇪🇬 Egyptian Arabic TTS Data Pipeline

A fully automated pipeline designed to generate, process, and annotate synthetic datasets for training Text-to-Speech (TTS) models in **Egyptian Arabic** and **Standard Arabic**.

The pipeline uses a "Producer-Consumer" architecture to generate conversational audio using **Google Gemini** and post-process it using **PyAnnote** (Diarization) and **OpenAI Whisper** (Transcription).

## 🚀 Architecture

1.  **Producer (`1_producer.py`)**:
    * Uses `gemini-2.5-flash` to write a natural podcast script between two speakers (Ahmed & Sara).
    * Uses `gemini-2.5-flash-preview-tts` (Native Multi-Speaker) to generate high-quality audio with distinct voices (Zephyr & Puck).
    * Saves raw audio and scripts to a `staging` directory.

2.  **Consumer (`2_consumer.py`)**:
    * Watches the `staging` folder for new files.
    * **Speaker Diarization:** Uses `pyannote/speaker-diarization-3.1` to identify who is speaking and when.
    * **Segmentation:** Cuts the audio into training-ready chunks (3-8 seconds) based on silence and speaker turns.
    * **Transcription:** Uses `openai-whisper` to generate accurate Arabic text for each chunk.
    * **Dataset Assembly:** Saves the final wavs and updates a `metadata.csv` file.

## 🛠️ Prerequisites

* **Python 3.10+**
* **FFmpeg** installed and added to system PATH (Required for audio processing).
* **Google AI Studio API Key** (for Gemini).
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
    Create a `.env` file in the root directory and add your keys:
    ```env
    GENAI_API_KEY=your_google_api_key_here
    HF_TOKEN=your_hugging_face_token_here
    ```

## ▶️ Usage

Open two separate terminal windows:

**Terminal 1 (The Consumer):**
Starts the listener. Wait until you see "Watching staging...".
```bash
python src/consumer.py
```

**Terminal 2 (The producer):**
it generates the script and the audio file.
```bash
python src/producer.py
```