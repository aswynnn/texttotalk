import streamlit as st
from docx.document import Document
from transformers import pipeline
import fitz  # PyMuPDF
import os
import tempfile
from google.cloud import texttospeech

# Initialize summarization pipeline and cache it to speed up app re-runs
@st.cache_resource
def get_summarizer():
    """Loads and caches the summarization model."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = get_summarizer()

# Streamlit page configuration
st.set_page_config(page_title="Document to Podcast", layout="centered")
st.title("📄 ➡️ 🎧 Document to Podcast")
st.write("Upload a document, and this app will summarize its content and generate a podcast-style audio version of the summary.")

# --- Helper Functions ---

def extract_text(file):
    """Extracts text content from an uploaded file (.txt, .pdf, .docx)."""
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == ".txt":
        return file.read().decode("utf-8")
    
    elif file_extension == ".pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    
    elif file_extension in [".docx", ".doc"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
            temp_docx.write(file.getvalue())
            temp_docx_path = temp_docx.name
        
        try:
            doc = Document(temp_docx_path)
            return "\n".join([para.text for para in doc.paragraphs])
        finally:
            os.remove(temp_docx_path) # Clean up the temp docx file
    
    else:
        st.error("Unsupported file format.")
        return None

def summarize_large_text(text, chunk_size=1024):
    """Summarizes text in chunks to handle large documents."""
    # Simple chunking by character length
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    summarized_chunks = []
    # Process each chunk with the summarizer
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summarized_chunks.append(result[0]['summary_text'])
        
    return " ".join(summarized_chunks)

def synthesize_speech(text, output_path):
    """Synthesizes speech from text using Google Cloud TTS and saves it."""
    
    # Get credentials from Streamlit's secrets manager
    creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]

    # Create a temporary file to hold the credentials for the API client
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_creds_file:
        temp_creds_file.write(creds_json_str)
        creds_file_path = temp_creds_file.name

    try:
        # Explicitly initialize the client with the service account file
        client = texttospeech.TextToSpeechClient.from_service_account_file(creds_file_path)

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Save the audio content to the specified output file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
    
    finally:
        # Securely clean up the temporary credentials file
        os.remove(creds_file_path)

# --- Streamlit UI ---

uploaded_file = st.file_uploader(
    "Upload a .txt, .pdf, or .docx file", 
    type=["txt", "pdf", "docx", "doc"]
)

if uploaded_file:
    raw_text = extract_text(uploaded_file)

    if raw_text:
        st.subheader("📃 Document Preview")
        st.text_area("Extracted Text", raw_text[:1500] + "...", height=250, disabled=True)

        if st.button("🔍 Summarize and Generate Podcast", use_container_width=True):
            summary = ""
            with st.spinner("✍️ Summarizing the document... This may take a moment."):
                summary = summarize_large_text(raw_text)
                st.subheader("📝 Summary")
                st.write(summary)

            with st.spinner("🎤 Generating audio..."):
                # Use a temporary file for the audio output
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
                    synthesize_speech(summary, tmp_audio_file.name)
                    audio_path = tmp_audio_file.name
                
                # Read the generated audio file for playback and download
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                st.subheader("🎧 Your Podcast is Ready!")
                st.audio(audio_bytes, format='audio/mp3')
                st.download_button(
                    label="⬇️ Download Podcast (MP3)",
                    data=audio_bytes,
                    file_name="summary_podcast.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )
                
                # Clean up the temporary audio file
                os.remove(audio_path)
