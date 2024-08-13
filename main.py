import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from io import BytesIO
import numpy as np
import soundfile as sf
import sounddevice as sd

# Load model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def transcribe_audio(audio_file):
    audio = audio_file.read()
    audio_np, samplerate = sf.read(BytesIO(audio))
    result = pipe({"audio": audio_np, "sampling_rate": samplerate})
    return result["text"]

def record_audio(duration=5, samplerate=16000):
    st.write("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording finished.")
    return audio.flatten(), samplerate

st.title("Whisper ASR Streamlit App")

st.header("Audio Input")
option = st.selectbox(
    "Choose Input Method",
    ["Upload Audio File", "Record Audio"]
)

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        transcription = transcribe_audio(uploaded_file)
        st.write("Transcription:")
        st.write(transcription)

if option == "Record Audio":
    duration = st.slider("Duration (seconds)", 1, 60, 5)
    if st.button("Start Recording"):
        audio, samplerate = record_audio(duration=duration)
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio, samplerate, format='wav')
        audio_buffer.seek(0)
        st.audio(audio_buffer, format='audio/wav', sample_rate=samplerate)
        transcription = transcribe_audio(audio_buffer)
        st.write("Transcription:")
        st.write(transcription)
