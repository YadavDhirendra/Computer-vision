import streamlit as st
import cv2
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time
import random
import tempfile
import os
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="🎵 Music Recommendation via Emotion Detection",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
        text-align: center;
    }
    .playlist-container {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 10px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None
if 'playlist_url' not in st.session_state:
    st.session_state.playlist_url = None
if 'mood_mapped' not in st.session_state:
    st.session_state.mood_mapped = None

# Load models and data
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model('model.h5')
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return model, face_classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def load_music_data():
    try:
        df = pd.read_csv('data_moods.csv')
        return df
    except Exception as e:
        st.error(f"Error loading music data: {str(e)}")
        return None

def initialize_spotify():
    """Initialize Spotify client"""
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id="a108245395774fd8b4a5d30b1f228465",
            client_secret="f32dcdd4f1034baebb96843b19d8f2b1",
            redirect_uri="https://localhost:8000",
            scope="user-read-playback-state streaming ugc-image-upload playlist-modify-public"
        ))
        return sp
    except Exception as e:
        st.error(f"Spotify authentication error: {str(e)}")
        return None

def detect_emotion_from_image(image, model, face_classifier):
    """Detect emotion from uploaded image"""
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, "No face detected in the image"
    
    emotion_counts = {}
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            prediction = model.predict(roi)[0]
            emotion = emotion_labels[prediction.argmax()]
            
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
    
    if emotion_counts:
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        return dominant_emotion, None
    else:
        return None, "Could not detect emotion"

def map_emotion_to_mood(emotion):
    """Map detected emotion to music mood category"""
    if emotion in ['Angry', 'Surprise']:
        return 'Energetic'
    elif emotion in ['Fear', 'Neutral']:
        return 'Calm'
    elif emotion == 'Happy':
        return 'Happy'
    elif emotion == 'Sad':
        return 'Sad'
    elif emotion == 'Disgust':
        return 'Calm'
    else:
        return 'Happy'  # Default

def create_spotify_playlist(sp, mood, music_df):
    """Create Spotify playlist based on mood"""
    try:
        # Filter songs by mood
        mood_songs = music_df[music_df['mood'] == mood]
        
        if len(mood_songs) == 0:
            return None, "No songs found for this mood"
        
        # Prepare track URIs
        track_uris = []
        for _, row in mood_songs.iterrows():
            track_uris.append(f"spotify:track:{row['id']}")
        
        # Limit to 15 songs if more available
        if len(track_uris) > 15:
            track_uris = random.sample(track_uris, 15)
        
        # Create playlist
        user_id = sp.me()['id']
        playlist_name = f"{mood} Songs - {time.strftime('%Y%m%d_%H%M%S')}"
        playlist_description = f"Generated playlist for {mood} mood based on emotion detection"
        
        playlist = sp.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=True,
            description=playlist_description
        )
        
        # Add tracks to playlist
        sp.user_playlist_add_tracks(
            user=user_id,
            playlist_id=playlist['id'],
            tracks=track_uris
        )
        
        return playlist['id'], None
        
    except Exception as e:
        return None, f"Error creating playlist: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">🎵 Music Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Using Facial Emotion Recognition to curate your perfect playlist</p>', unsafe_allow_html=True)
    
    # Load models and data
    model, face_classifier = load_emotion_model()
    music_df = load_music_data()
    
    if model is None or face_classifier is None:
        st.error("⚠️ Could not load emotion detection model. Please ensure 'model.h5' and 'haarcascade_frontalface_default.xml' are in the project directory.")
        return
    
    if music_df is None:
        st.error("⚠️ Could not load music database. Please ensure 'data_moods.csv' is in the project directory.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    st.sidebar.markdown("---")
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose Input Method:",
        ["📷 Upload Image", "📸 Camera Capture", "🎭 Demo Mode"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Emotion Detection")
        
        if input_method == "📷 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image with your face visible"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Analyze Emotion", type="primary"):
                    with st.spinner("Detecting emotion..."):
                        emotion, error = detect_emotion_from_image(image, model, face_classifier)
                        
                        if emotion:
                            st.session_state.detected_emotion = emotion
                            st.session_state.mood_mapped = map_emotion_to_mood(emotion)
                            st.success(f"✅ Emotion detected: **{emotion}**")
                        else:
                            st.error(f"❌ {error}")
        
        elif input_method == "📸 Camera Capture":
            st.info("📱 Camera functionality works best on the deployed version with HTTPS")
            
            camera_input = st.camera_input("Take a picture")
            
            if camera_input is not None:
                image = Image.open(camera_input)
                
                with st.spinner("Detecting emotion..."):
                    emotion, error = detect_emotion_from_image(image, model, face_classifier)
                    
                    if emotion:
                        st.session_state.detected_emotion = emotion
                        st.session_state.mood_mapped = map_emotion_to_mood(emotion)
                        st.success(f"✅ Emotion detected: **{emotion}**")
                    else:
                        st.error(f"❌ {error}")
        
        else:  # Demo Mode
            st.info("🎭 Demo Mode - Select an emotion to see music recommendations")
            demo_emotion = st.selectbox(
                "Choose an emotion:",
                ["Happy", "Sad", "Angry", "Surprise", "Fear", "Neutral", "Disgust"]
            )
            
            if st.button("🎵 Get Recommendations", type="primary"):
                st.session_state.detected_emotion = demo_emotion
                st.session_state.mood_mapped = map_emotion_to_mood(demo_emotion)
                st.success(f"✅ Selected emotion: **{demo_emotion}**")
    
    with col2:
        st.subheader("🎵 Music Recommendations")
        
        if st.session_state.detected_emotion and st.session_state.mood_mapped:
            # Display emotion and mood mapping
            st.markdown(f"""
            <div class="emotion-box">
                <h3>Detected Emotion: {st.session_state.detected_emotion}</h3>
                <h4>Mapped Mood: {st.session_state.mood_mapped}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show available songs for this mood
            mood_songs = music_df[music_df['mood'] == st.session_state.mood_mapped]
            st.write(f"📊 Found **{len(mood_songs)}** songs for {st.session_state.mood_mapped} mood")
            
            # Spotify playlist creation
            st.markdown("---")
            st.subheader("🎧 Create Spotify Playlist")
            
            if st.button("🎶 Create Playlist on Spotify", type="secondary"):
                with st.spinner("Creating Spotify playlist..."):
                    sp = initialize_spotify()
                    if sp:
                        playlist_id, error = create_spotify_playlist(sp, st.session_state.mood_mapped, music_df)
                        
                        if playlist_id:
                            st.session_state.playlist_url = f"https://open.spotify.com/embed/playlist/{playlist_id}"
                            st.success("✅ Playlist created successfully!")
                        else:
                            st.error(f"❌ {error}")
                    else:
                        st.error("❌ Could not authenticate with Spotify")
            
            # Display playlist if created
            if st.session_state.playlist_url:
                st.markdown("### 🎵 Your Personalized Playlist")
                st.components.v1.iframe(
                    st.session_state.playlist_url,
                    width=None,
                    height=400,
                    scrolling=False
                )
        
        else:
            st.info("👆 Please detect an emotion first to get music recommendations")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🎵 Music Recommendation System using Facial Emotion Recognition</p>
        <p>Built with Streamlit • Powered by TensorFlow & Spotify API</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()