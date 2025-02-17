import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    BertModel,
    PegasusForConditionalGeneration
)
import logging
import json
import pandas as pd
import nltk
import os
import base64
from gtts import gTTS
import tempfile
import requests
import string
from langdetect import detect
import warnings
import shutil
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import librosa
import moviepy.editor as mpy
from scipy.signal import savgol_filter
import numba
from datetime import datetime
import scipy.ndimage as ndimage
from pydub import AudioSegment
import threading
from queue import Queue

# Constants
CURRENT_USER = "Satwik-Uppada298"
CURRENT_TIME = "2025-02-16 06:37:02"
MAX_TEXT_LENGTH = 50000
CACHE_DIR = "./cache"
MODELS_DIR = "./models/version_2025-02-15_11-39-55"
FIXED_IMAGE_PATH = "female.jpg"

# Supported Languages
LANGUAGES = {
    "English": {"code": "en", "name": "English"},
    "Telugu": {"code": "te", "name": "తెలుగు"},
    "Hindi": {"code": "hi", "name": "हिंदी"},
    "Tamil": {"code": "ta", "name": "தமிழ்"},
    "Malayalam": {"code": "ml", "name": "മലയാളം"},
    "Kannada": {"code": "kn", "name": "ಕನ್ನಡ"}
}

# Model configuration with correct model classes
MODEL_CONFIG = {
    "T5": {
        "name": "t5-base",
        "local_dir": os.path.join(MODELS_DIR, "t5-base"),
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": AutoTokenizer
    },
    "PEGASUS": {
        "name": "pegasus-large",
        "local_dir": os.path.join(MODELS_DIR, "pegasus-large"),
        "model_class": PegasusForConditionalGeneration,
        "tokenizer_class": AutoTokenizer
    },
    "BERT": {
        "name": "bert-base-uncased",
        "local_dir": os.path.join(MODELS_DIR, "bert-base-uncased"),
        "model_class": BertModel,
        "tokenizer_class": AutoTokenizer
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Setup NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


class SummaryEvaluator:
    @staticmethod
    def evaluate_summary(original_text, summary):
        try:
            original_words = set(nltk.word_tokenize(original_text.lower()))
            summary_words = set(nltk.word_tokenize(summary.lower()))

            coverage = len(summary_words.intersection(original_words)) / len(original_words)
            density = len(summary_words) / len(summary.split())
            length_ratio = len(summary.split()) / len(original_text.split())

            score = (0.4 * coverage + 0.3 * density + 0.3 * (1 - abs(0.3 - length_ratio)))
            return score
        except Exception as e:
            logger.error(f"Error evaluating summary: {str(e)}")
            return 0.0


class TextProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.models = {}
        self.tokenizers = {}
        self.evaluator = SummaryEvaluator()

    def load_model(self, model_name):
        if model_name not in self.models:
            config = MODEL_CONFIG[model_name]
            try:
                model_path = config["local_dir"]
                if not os.path.exists(model_path):
                    logger.error(f"Model path not found: {model_path}")
                    return False

                tokenizer = config["tokenizer_class"].from_pretrained(model_path)
                model = config["model_class"].from_pretrained(model_path).to(self.device)

                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                logger.info(f"Successfully loaded {model_name} from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
                return False
        return True

    def generate_summary(self, text, model_name, max_length=150, min_length=50):
        try:
            if not self.load_model(model_name):
                return None

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            # Special handling for different models
            if model_name == "T5":
                text = "summarize: " + text
            elif model_name == "BERT":
                # BERT requires special handling as it's not a seq2seq model
                encoded = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = model(**encoded)
                return self._bert_extractive_summary(text, outputs, max_length)

            inputs = tokenizer(
                text,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return summary
        except Exception as e:
            logger.error(f"Error generating summary with {model_name}: {str(e)}")
            return None

    def _bert_extractive_summary(self, text, outputs, max_length):
        # Simple extractive summarization for BERT
        sentences = nltk.sent_tokenize(text)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Use cosine similarity to find most representative sentences
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings, embeddings)

        # Select top sentences
        scores = similarities.sum(axis=1)
        ranked_sentences = [x for _, x in sorted(zip(scores, sentences), reverse=True)]

        # Join top sentences
        summary = " ".join(ranked_sentences[:3])
        return summary

    def get_best_summary(self, text, max_length=150, min_length=50):
        summaries = {}
        scores = {}

        for model_name in ["T5", "PEGASUS"]:  # Removed BERT from ensemble as it's extractive
            summary = self.generate_summary(text, model_name, max_length, min_length)
            if summary:
                summaries[model_name] = summary
                scores[model_name] = self.evaluator.evaluate_summary(text, summary)

        if scores:
            best_model = max(scores.items(), key=lambda x: x[1])[0]
            return best_model, summaries[best_model]
        return None, None


class EnhancedLipAnimation:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True
        )
        self.frame_queue = Queue(maxsize=100)
        self.processed_frames = []
        self.num_threads = max(1, os.cpu_count() - 1)

        self.lip_landmarks = {
            'upper_outer': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409],
            'lower_outer': [146, 91, 181, 84, 17, 314, 405, 321, 375],
            'upper_inner': [191, 80, 81, 82, 13, 312, 311, 310, 415],
            'lower_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318],
            'left_corner': [78, 191, 80],
            'right_corner': [308, 311, 310]
        }

    def text_to_speech(self, text, lang='en'):
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_audio.name)
            return temp_audio.name
        except Exception as e:
            logger.error(f"Audio generation error: {str(e)}")
            return None

    def translate_text(self, text, target_lang, source_lang='en'):
        if not text or target_lang == 'en':
            return text

        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": source_lang,
                "tl": target_lang,
                "dt": "t",
                "q": text
            }
            headers = {"User-Agent": "Mozilla/5.0"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                translated_text = ''.join([item[0] for item in result[0] if item[0]])
                return translated_text
            return text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

    def process_image(self, image):
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                return None

            return results.multi_face_landmarks[0]
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None

    def _get_lip_points(self, landmarks, shape):
        h, w = shape
        points = {}
        for region, indices in self.lip_landmarks.items():
            points[region] = np.array([[int(landmarks.landmark[idx].x * w),
                                        int(landmarks.landmark[idx].y * h)]
                                       for idx in indices])
        return points

    def _apply_enhanced_movement(self, image, points, amplitude, frequency):
        try:
            h, w = image.shape[:2]
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)

            center = np.mean(np.vstack([points['upper_outer'], points['lower_outer']]), axis=0)
            top = np.min(points['upper_outer'][:, 1])
            bottom = np.max(points['lower_outer'][:, 1])
            left = np.min(np.vstack([points['left_corner']])[:, 0])
            right = np.max(np.vstack([points['right_corner']])[:, 0])

            mouth_mask = ((map_y >= top) & (map_y <= bottom) &
                          (map_x >= left) & (map_x <= right))

            dx = map_x - center[0]
            dy = map_y - center[1]
            dist = np.sqrt(dx * dx + dy * dy)

            movement_mask = mouth_mask & (dist < (bottom - top))
            upper_mask = movement_mask & (map_y < center[1])
            lower_mask = movement_mask & (map_y >= center[1])

            if np.any(upper_mask):
                map_y[upper_mask] -= amplitude * 4.0 * (1.2 + np.sin(frequency * dist[upper_mask]))
            if np.any(lower_mask):
                map_y[lower_mask] += amplitude * 4.0 * (1.2 + np.sin(frequency * dist[lower_mask]))

            left_mask = movement_mask & (map_x < center[0])
            right_mask = movement_mask & (map_x >= center[0])

            if np.any(left_mask):
                map_x[left_mask] -= amplitude * 0.8 * 1.2
            if np.any(right_mask):
                map_x[right_mask] += amplitude * 0.8 * 1.2

            map_x = ndimage.gaussian_filter(map_x, sigma=1)
            map_y = ndimage.gaussian_filter(map_y, sigma=1)

            return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        except Exception as e:
            logger.error(f"Error in movement application: {str(e)}")
            return image

    def _process_frame_worker(self):
        while True:
            try:
                frame_data = self.frame_queue.get()
                if frame_data is None:
                    break

                frame, lip_points, amplitude, frequency = frame_data
                processed_frame = self._apply_enhanced_movement(frame, lip_points, amplitude, frequency)
                self.processed_frames.append(processed_frame)
                self.frame_queue.task_done()
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")

    def create_animation(self, image, landmarks, audio_file, expression_level='natural'):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            fps = 30
            samples_per_frame = sr // fps

            n_fft = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
            spectral = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

            n_frames = int(len(y) / samples_per_frame)
            rms = np.interp(np.linspace(0, 1, n_frames),
                            np.linspace(0, 1, len(rms)), rms)
            spectral = np.interp(np.linspace(0, 1, n_frames),
                                 np.linspace(0, 1, len(spectral)), spectral)

            combined = (0.7 * rms / np.max(rms) + 0.3 * spectral / np.max(spectral))
            smooth_movement = savgol_filter(combined, 5, 2)

            intensity_map = {
                'subtle': 2.5,
                'natural': 4.0,
                'expressive': 5.5
            }
            movement = smooth_movement * intensity_map[expression_level.lower()]

            lip_points = self._get_lip_points(landmarks, image.shape[:2])

            self.processed_frames = []
            workers = []
            for _ in range(self.num_threads):
                worker = threading.Thread(target=self._process_frame_worker)
                worker.start()
                workers.append(worker)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(len(movement)):
                frame = image.copy()
                amplitude = movement[i]
                frequency = 2 * np.pi * (i % 10) / 10
                self.frame_queue.put((frame, lip_points, amplitude, frequency))

                progress = (i + 1) / len(movement)
                progress_bar.progress(progress)
                status_text.text(f"Creating animation... {progress * 100:.1f}%")

            for _ in workers:
                self.frame_queue.put(None)
            for worker in workers:
                worker.join()

            progress_bar.empty()
            status_text.empty()

            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            clip = mpy.ImageSequenceClip(self.processed_frames, fps=fps)
            audio_clip = mpy.AudioFileClip(audio_file)

            duration = min(clip.duration, audio_clip.duration)
            clip = clip.subclip(0, duration)
            audio_clip = audio_clip.subclip(0, duration)
            final_clip = clip.set_audio(audio_clip)

            final_clip.write_videofile(
                temp_video.name,
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                bitrate="8000k",
                threads=self.num_threads,
                preset='ultrafast',
                ffmpeg_params=['-tune', 'zerolatency']
            )

            clip.close()
            audio_clip.close()

            return temp_video.name

        except Exception as e:
            logger.error(f"Error creating animation: {str(e)}")
            return None
@st.cache_resource
def get_text_processor():
    return TextProcessor()

@st.cache_resource
def get_animation_instance():
    return EnhancedLipAnimation()

def main():
    st.set_page_config(
        page_title="Advanced Text-to-Animation Suite",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎭 Advanced Text-to-Animation Suite")
    st.markdown(f"""
    **Current Time (UTC):** {CURRENT_TIME}  
    **Current User:** {CURRENT_USER}
    """)

    text_processor = get_text_processor()
    animation = get_animation_instance()

    # Load the fixed female image
    try:
        image = Image.open(FIXED_IMAGE_PATH)
        image_array = np.array(image)
        landmarks = animation.process_image(image_array)

        if not landmarks:
            st.error("❌ Failed to detect face in the default image!")
            return

        st.session_state['image'] = image_array
        st.session_state['landmarks'] = landmarks

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Avatar Image", use_column_width=True)
            st.success("✅ Avatar ready!")

        with col2:
            st.subheader("📝 Enter Text & Configure Settings")

            text_input = st.text_area(
                "Enter your text:",
                height=150,
                placeholder="Enter the text you want to summarize and animate..."
            )

            if text_input:
                st.info(f"📊 Character count: {len(text_input)}")

                col_a, col_b = st.columns(2)
                with col_a:
                    max_length = st.slider("Maximum summary length", 50, 300, 150)
                with col_b:
                    min_length = st.slider("Minimum summary length", 30, 200, 50)

                st.subheader("🌐 Select Output Language")
                output_lang = st.selectbox(
                    "Select Output Language",
                    options=list(LANGUAGES.keys()),
                    format_func=lambda x: f"{x} ({LANGUAGES[x]['name']})"
                )

                expression_level = st.select_slider(
                    "Expression Level",
                    options=["Subtle", "Natural", "Expressive"],
                    value="Natural"
                )

                if st.button("🚀 Generate Animation"):
                    if len(text_input) > MAX_TEXT_LENGTH:
                        st.error(f"❌ Text is too long! Maximum {MAX_TEXT_LENGTH} characters allowed.")
                        return

                    with st.spinner("🎯 Processing text and generating summary..."):
                        best_model, summary = text_processor.get_best_summary(
                            text_input,
                            max_length=max_length,
                            min_length=min_length
                        )

                        if summary:
                            st.success(f"✨ Best summary generated using {best_model} model")

                            st.subheader("📄 English Summary")
                            st.write(summary)

                            target_lang_code = LANGUAGES[output_lang]["code"]
                            if target_lang_code != "en":
                                with st.spinner(f"🌐 Translating to {output_lang}..."):
                                    translated_summary = animation.translate_text(
                                        summary,
                                        target_lang_code
                                    )

                                    st.subheader(f"📄 {output_lang} Summary")
                                    st.write(translated_summary)
                                    summary = translated_summary

                            with st.spinner("🎬 Creating speaking animation..."):
                                audio_file = animation.text_to_speech(
                                    summary,
                                    lang=target_lang_code
                                )

                                if audio_file:
                                    video_file = animation.create_animation(
                                        st.session_state['image'],
                                        st.session_state['landmarks'],
                                        audio_file,
                                        expression_level.lower()
                                    )

                                    if video_file:
                                        st.subheader("🎥 Generated Animation")
                                        st.video(video_file)
                                        st.success("✅ Animation created successfully!")

                                        # Clean up temporary files
                                        os.unlink(audio_file)
                                        os.unlink(video_file)
                                    else:
                                        st.error("❌ Failed to create animation!")
                                else:
                                    st.error("❌ Failed to generate audio!")
                        else:
                            st.error("❌ Failed to generate summary!")

    except Exception as e:
        st.error(f"❌ Error loading default image: {str(e)}")
        return

if __name__ == "__main__":
    try:
        # Initialize cache and models directories
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Run the application
        main()
    finally:
        # Cleanup cache directory
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR)