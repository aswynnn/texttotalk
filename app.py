import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
from docx import Document
from dotenv import load_dotenv
import os
import tempfile
from google.cloud import texttospeech

# --- Configuration and Initialization ---

# Load environment variables from a .env file if it exists
load_dotenv()

# Set Google Cloud credentials from environment variables
# Note: For Streamlit Cloud, set this as a secret.
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
    creds_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_creds:
        temp_creds.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds.name
elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    pass # Already set
else:
    st.error("Google Cloud credentials are not set. Please configure them in your environment or Streamlit secrets.")
    st.stop()


# Initialize Hugging Face pipelines
# Using a specific BART model for summarization
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Using a smaller model for text generation to keep it faster
    text_generator = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    st.error(f"Failed to load Hugging Face models. Error: {e}")
    st.stop()

# Initialize Google TTS Client
try:
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    st.error(f"Failed to initialize Google TTS client. Error: {e}")
    st.stop()


# --- Helper Functions ---

def extract_text_from_file(file):
    """Extracts text content from uploaded file (.txt, .pdf, .docx)."""
    file_extension = os.path.splitext(file.name)[1]
    
    if file_extension == ".txt":
        return file.read().decode("utf-8")
    
    elif file_extension == ".pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    
    elif file_extension == ".docx":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text])
        
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

def process_text_in_chunks(text, processor, chunk_size=1024, **kwargs):
    """
    Splits text into chunks and applies a processing function (e.g., summarizer).
    Handles chunking to stay within model token limits.
    """
    if not text:
        return ""
    # Split by paragraphs to respect sentence boundaries
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 < chunk_size:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph + "\n"
    chunks.append(current_chunk)
    
    processed_chunks = []
    with st.spinner(f'Processing {len(chunks)} chunks...'):
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                st.info(f"Processing chunk {i+1}/{len(chunks)}")
                try:
                    result = processor(chunk, **kwargs)
                    processed_chunks.append(result[0]['summary_text' if 'summary_text' in result[0] else 'generated_text'])
                except Exception as e:
                    st.warning(f"Could not process chunk {i+1}. Skipping. Error: {e}")
    return " ".join(processed_chunks)

def generate_script_from_topic(topic):
    """Generates a podcast script from a given topic."""
    prompt = f"Create a short, engaging, and conversational podcast script about '{topic}'. Start by introducing the topic clearly."
    try:
        generated_text = text_generator(prompt, max_length=500, num_return_sequences=1, pad_token_id=text_generator.tokenizer.eos_token_id)
        return generated_text[0]['generated_text']
    except Exception as e:
        st.error(f"Failed to generate script: {e}")
        return None

def get_available_voices():
    """Fetches available en-US voices from Google TTS."""
    try:
        response = tts_client.list_voices(language_code="en-US")
        voices = sorted([voice.name for voice in response.voices], key=lambda name: (name.split('-')[-2], name))
        return voices
    except Exception as e:
        st.warning(f"Could not fetch voice list: {e}")
        return ["en-US-Wavenet-F"] # Fallback voice

def synthesize_speech(text, voice_name):
    """Synthesizes speech from text using a selected voice and returns audio bytes."""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        st.error(f"Audio synthesis failed: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="Text-to-Podcast Generator", layout="wide", initial_sidebar_state="auto")

st.title("🎙️ Text-to-Podcast Generator")
st.markdown("Turn any text, document, or topic into a professional-sounding podcast with customizable AI voices.")

# Sidebar for voice selection
st.sidebar.header("⚙️ Podcast Settings")
available_voices = get_available_voices()
selected_voice = st.sidebar.selectbox(
    "Choose a Voice",
    options=available_voices,
    help="Select from a list of available high-quality voices from Google."
)

# --- Main Content Area with Tabs ---

tab1, tab2, tab3 = st.tabs(["📄 Upload a Document", "✍️ Enter Raw Text", "💡 Generate from Topic"])

# --- Tab 1: Upload a Document ---
with tab1:
    st.header("Summarize a Document into a Podcast Script")
    uploaded_file = st.file_uploader(
        "Upload a .txt, .pdf, or .docx file",
        type=["txt", "pdf", "docx"]
    )
    
    if uploaded_file:
        raw_text = extract_text_from_file(uploaded_file)
        if raw_text:
            st.success("File successfully uploaded and text extracted.")
            if st.button("Analyze and Summarize Document", key="summarize_doc"):
                summary = process_text_in_chunks(
                    raw_text, 
                    summarizer, 
                    min_length=30, 
                    max_length=150
                )
                st.session_state.editable_text = summary
        
# --- Tab 2: Enter Raw Text ---
with tab2:
    st.header("Convert Your Text into a Podcast Script")
    raw_text_input = st.text_area("Paste your text here:", height=250, placeholder="Enter any text you want to convert...")
    
    if st.button("Process and Summarize Text", key="summarize_text"):
        if raw_text_input.strip():
            summary = process_text_in_chunks(
                raw_text_input,
                summarizer,
                min_length=30,
                max_length=150
            )
            st.session_state.editable_text = summary
        else:
            st.warning("Please enter some text to process.")
            
# --- Tab 3: Generate from Topic ---
with tab3:
    st.header("Generate a Podcast Script from a Topic")
    topic_input = st.text_input("Enter a topic or keyword:", placeholder="e.g., 'The history of artificial intelligence'")
    
    if st.button("Generate Script", key="generate_topic"):
        if topic_input.strip():
            generated_script = generate_script_from_topic(topic_input)
            st.session_state.editable_text = generated_script
        else:
            st.warning("Please enter a topic.")

# --- Shared UI for Editing and Generation ---

if 'editable_text' in st.session_state and st.session_state.editable_text:
    st.divider()
    st.subheader("✍️ Review and Edit Your Podcast Script")
    st.markdown("Your generated script is ready. You can make any changes here before converting it to audio.")
    
    edited_text = st.text_area(
        "Editable Script:",
        value=st.session_state.editable_text,
        height=300,
        key="editor"
    )

    if st.button("🎤 Generate Podcast Audio", type="primary"):
        if edited_text.strip():
            with st.spinner(f"Generating audio with voice '{selected_voice}'... This may take a moment."):
                audio_bytes = synthesize_speech(edited_text, selected_voice)
                if audio_bytes:
                    st.success("Podcast generated successfully!")
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        label="⬇️ Download Podcast (MP3)",
                        data=audio_bytes,
                        file_name="generated_podcast.mp3",
                        mime="audio/mp3"
                    )
        else:
            st.error("The script is empty. Please generate or enter some text.")
