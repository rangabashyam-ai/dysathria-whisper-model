import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from io import BytesIO
import numpy as np
import soundfile as sf
#import sounddevice as sd
from PyPDF2 import PdfFileReader
from docx import Document
from odf.opendocument import load
from odf.text import P

# Load model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    processor = AutoProcessor.from_pretrained(model_id)
except Exception as e:
    st.error(f"Error loading processor: {e}")

try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
except Exception as e:
    st.error(f"Error creating ASR pipeline: {e}")

def transcribe_audio(audio_bytes):
    try:
        audio_np, samplerate = sf.read(BytesIO(audio_bytes))
        result = pipe({"raw": audio_np, "sampling_rate": samplerate})
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# def record_audio(duration=5, samplerate=16000):
#     try:
#         st.write("Recording...")
#         audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
#         sd.wait()
#         st.write("Recording finished.")
#         return audio.flatten(), samplerate
#     except Exception as e:
#         st.error(f"Error recording audio: {e}")
#         return None, None

def process_pdf(file):
    try:
        pdf = PdfFileReader(BytesIO(file.read()))
        text = ""
        for page_num in range(pdf.numPages):
            text += pdf.getPage(page_num).extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def process_docx(file):
    try:
        doc = Document(BytesIO(file.read()))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return None

def process_odt(file):
    try:
        doc = load(BytesIO(file.read()))
        paragraphs = doc.getElementsByType(P)
        text = "\n".join([para.firstChild.nodeValue for para in paragraphs])
        return text
    except Exception as e:
        st.error(f"Error processing ODT: {e}")
        return None

st.title("Dysarthria-Whisper App")

st.header("Audio Input")
option = st.selectbox(
    "Choose Input Method",
    ["Upload Audio File", "Record Audio", "Type Text", "Upload Document"]
)

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.write("Uploaded file name:", uploaded_file.name)
        st.audio(uploaded_file, format='audio/wav')
        audio_bytes = uploaded_file.read()
        transcription = transcribe_audio(audio_bytes)
        if transcription:
            st.write("Transcription:")
            st.write(transcription)

# elif option == "Record Audio":
#     duration = st.slider("Duration (seconds)", 1, 60, 5)
#     if st.button("Start Recording"):
#         audio, samplerate = record_audio(duration=duration)
#         if audio is not None:
#             audio_buffer = BytesIO()
#             sf.write(audio_buffer, audio, samplerate, format='wav')
#             audio_buffer.seek(0)
#             st.audio(audio_buffer, format='audio/wav', sample_rate=samplerate)
#             transcription = transcribe_audio(audio_buffer.getvalue())
#             if transcription:
#                 st.write("Transcription:")
#                 st.write(transcription)

elif option == "Type Text":
    input_text = st.text_area("Enter text here")
    if input_text:
        st.write("Text Entered:")
        st.write(input_text)

elif option == "Upload Document":
    uploaded_doc = st.file_uploader("Choose a document", type=["pdf", "docx", "odt"])
    if uploaded_doc is not None:
        st.write("Document Uploaded:")
        st.write(uploaded_doc.name)
        if uploaded_doc.type == "application/pdf":
            text = process_pdf(uploaded_doc)
        elif uploaded_doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = process_docx(uploaded_doc)
        elif uploaded_doc.type == "application/vnd.oasis.opendocument.text":
            text = process_odt(uploaded_doc)
        else:
            text = "Unsupported document type."
        
        if text:
            st.write("Document Content:")
            st.write(text)
