import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

model_path = 'best.pt'

try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e

st.title("Tennis Tracking App") 
st.write("Upload a tennis video to detect and track players.")

uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st.write(" Processing video... Please wait.") 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                results = model(frame)
        else:
            results = model(frame)

        frame = np.squeeze(results.render())
        out.write(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels='RGB', use_container_width=True)
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        time.sleep(1 / fps)

    cap.release()
    out.release()

    st.success(" Video processing complete!") 

    st.write(" Download the processed video:") 
    with open(output_video_path, 'rb') as f:
        st.download_button(
            label=" Download Processed Video", 
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    os.remove(temp_video_path)
    os.remove(output_video_path)