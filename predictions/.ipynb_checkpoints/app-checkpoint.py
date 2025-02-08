import streamlit as st
import cv2
import numpy as np
from datetime import timedelta
from helper import YOLO_Pred

# Initialize YOLO_Pred with your model and YAML configuration
yolo = YOLO_Pred('hell/weights/best.onnx', 'data.yaml')

# Streamlit app title and description
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 YOLO Object Detection")
st.write("Upload a video to detect objects in real-time.")

# File uploader for videos
uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file is not None:
    # Create a temporary file to save the uploaded video
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded! Processing...")

    # Video capture
    video_file = cv2.VideoCapture(temp_video_path)

    # Create a placeholder for video display
    video_placeholder = st.empty()

    # Set the desired width and height for the display
    display_width = 640
    display_height = 360

    # Initialize frame counter for timestamp
    frame_count = 0
    fps = video_file.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video

    # Process video frame by frame
    while True:
        ret, frame = video_file.read()
        if not ret:
            st.write("End of video.")
            break

        # Get predictions
        img_pred, predicted_texts, boxes = yolo.predictions(frame)

        # Convert BGR image to RGB for display
        img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes and labels on the frame
        for i, text in enumerate(predicted_texts):
            box = boxes[i]  # Assuming boxes is a list of [left, top, width, height]
            x, y, w, h = box

            # Draw rectangle and label
            cv2.rectangle(img_pred_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
            cv2.putText(img_pred_rgb, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add timestamp to the video frame
        timestamp = str(timedelta(seconds=int(frame_count / fps)))
        cv2.putText(img_pred_rgb, f'Time: {timestamp}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Resize the image for display
        img_resized = cv2.resize(img_pred_rgb, (display_width, display_height))

        # Show predicted image in the placeholder
        video_placeholder.image(img_resized, caption='Predicted Video Frame', use_column_width=True)

        # Update predictions in the sidebar
        st.sidebar.write("### Predictions")
        st.sidebar.write("Detected Objects:")
        if predicted_texts:
            for text in predicted_texts:
                st.sidebar.write(text)
        else:
            st.sidebar.write("No objects detected.")

        # Show the timestamp in the sidebar
        st.sidebar.write(f"Current Time: {timestamp}")

        # Update frame counter
        frame_count += 1

    # Release video capture
    video_file.release()

# Add footer information
st.sidebar.write("### About")
st.sidebar.info("This app uses the YOLO model for real-time object detection. Upload a video to see predictions in action.")
