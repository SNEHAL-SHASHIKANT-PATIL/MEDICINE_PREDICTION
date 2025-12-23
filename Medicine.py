import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

# ✅ Add background color using HTML
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f8ff;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Streamlit Page Setup ------------------
st.set_page_config(page_title="Medicine Recommendation with Voice Input")
st.title("🗣💊 AI Medicine Recommender (with Voice)")

# ------------------ Load & Preprocess Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Clean_Medicine_Prediction.csv")
    return df

df = load_data()

# Label Encoding
le_gender = LabelEncoder()
le_disease = LabelEncoder()
le_medicine = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Disease'] = le_disease.fit_transform(df['Disease'])
df['Medicine Given'] = le_medicine.fit_transform(df['Medicine Given'])

# Clean Duration (extract numeric values)
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)

# Process Symptoms
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip().lower() for s in str(x).split(',')])
mlb = MultiLabelBinarizer()
symptom_features = mlb.fit_transform(df['Symptoms'])
symptom_df = pd.DataFrame(symptom_features, columns=mlb.classes_, index=df.index)

# Combine features
df_final = pd.concat([df[['Age', 'Gender', 'Duration']], symptom_df], axis=1)
X = df_final
y = df['Medicine Given']

# Model Training
model = RandomForestClassifier()
model.fit(X, y)

# ------------------ Voice Input Setup ------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_data = frame.to_ndarray()
        try:
            audio_bytes = audio_data.tobytes()
            audio_source = sr.AudioData(audio_bytes, frame.sample_rate, 2)
            text = self.recognizer.recognize_google(audio_source)
            st.session_state['voice_symptoms'] = text.lower()
        except sr.UnknownValueError:
            st.session_state['voice_symptoms'] = ""
        except sr.RequestError:
            st.session_state['voice_symptoms'] = ""
        return frame

# Voice Input Component
st.subheader("🎙 Voice Symptom Input (optional)")
try:
    from streamlit_webrtc import ClientSettings
    webrtc_streamer(
        key="speech",
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}),
    )
except ImportError:
    webrtc_streamer(
        key="speech",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

voice_input = st.session_state.get("voice_symptoms", "")

# ------------------ User Inputs ------------------
st.subheader("🧾 Patient Details")
age = st.slider("Age", 0, 100, 30)
gender = st.selectbox("Gender", le_gender.classes_)
duration = st.slider("Duration of Symptoms (in days)", 1, 30, 3)

# ------------------ Symptoms Input ------------------
st.subheader("💬 Select or Speak Symptoms")
manual_symptoms = st.multiselect("Select Symptoms", mlb.classes_)

if voice_input:
    st.write(f"🎤 You said: *{voice_input}*")
    spoken_symptoms = [s.strip().lower() for s in voice_input.split()]
else:
    spoken_symptoms = []

# Merge spoken + selected symptoms
selected_symptoms = list(set(manual_symptoms + spoken_symptoms))

# ------------------ Prediction ------------------
if st.button("🔍 Recommend Medicine"):
    if not selected_symptoms:
        st.warning("Please select or speak at least one symptom.")
    else:
        input_data = {
            'Age': age,
            'Gender': le_gender.transform([gender])[0],
            'Duration': duration
        }

        # Create symptom input vector
        symptom_input = np.zeros(len(mlb.classes_))
        for symptom in selected_symptoms:
            if symptom in mlb.classes_:
                idx = list(mlb.classes_).index(symptom)
                symptom_input[idx] = 1

        input_vector = np.concatenate([[input_data['Age'], input_data['Gender'], input_data['Duration']], symptom_input])
        input_vector = input_vector.reshape(1, -1)

        # Predict
        proba = model.predict_proba(input_vector)[0]
        top_3_idx = np.argsort(proba)[::-1][:3]
        top_3_meds = le_medicine.inverse_transform(top_3_idx)

        st.success("✅ Recommended Medicines:")
        for i, med in enumerate(top_3_meds, 1):
            st.write(f"{i}. *{med}*")

        st.caption("🤖 Prediction based on trained machine learning model.")
