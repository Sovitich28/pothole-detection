import contextlib
import io
import os
import queue
import time
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Initial page config and session state setup
st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session states
if "page" not in st.session_state:
    st.session_state.page = "home"
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.5
if "is_detecting" not in st.session_state:
    st.session_state.is_detecting = False
if "is_webcam_active" not in st.session_state:
    st.session_state.is_webcam_active = False
if "previous_page" not in st.session_state:
    st.session_state.previous_page = "home"

# Create output directories
os.makedirs("output/images", exist_ok=True)
os.makedirs("output/videos", exist_ok=True)


# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")


# Initialize model
model = load_model()

# Pothole class (assuming single class for pothole detection, adjust if model has multiple classes)
pothole_class = ["pothole"]

# Add back button style
st.markdown(
    """
    <style>
        .back-button {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
            padding: 10px 20px;
            background-color: #4f46e5;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .back-button:hover {
            background-color: #4338ca;
        }
        .feature-card {
            text-align: center;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 1rem;
            cursor: pointer;
            background-color: #f9f9f9;
        }
        .feature-card:hover {
            background-color: #f0f0f0;
        }
        .feature-icon {
            font-size: 2rem;
            color: #4f46e5;
            margin-bottom: 0.5rem;
        }
        .section-title {
            text-align: center;
            font-weight: 700;
            color: #1e293b;
        }
        .section-subtitle {
            text-align: center;
            color: #64748b;
            max-width: 80%;
            margin: 0 auto 1rem;
        }
        .upload-area {
            text-align: center;
            padding: 1rem;
            border: 2px dashed #ddd;
            border-radius: 5px;
            background-color: #f8fafc;
            margin-bottom: 1rem;
        }
    </style>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)


def detect_potholes_image(image, model, confidence, selected_classes):
    img_array = np.array(image)
    results = model(img_array, conf=confidence)
    detections = results[0]

    boxes = detections.boxes.xyxy.cpu().numpy() if len(detections) > 0 else []
    confs = detections.boxes.conf.cpu().numpy() if len(detections) > 0 else []
    class_ids = (
        detections.boxes.cls.cpu().numpy().astype(int) if len(detections) > 0 else []
    )

    if selected_classes:
        filtered = [
            (box, conf, class_id)
            for box, conf, class_id in zip(boxes, confs, class_ids)
            if pothole_class[class_id] in selected_classes
        ]
        if filtered:
            boxes, confs, class_ids = zip(*filtered)
        else:
            boxes, confs, class_ids = [], [], []

    image_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{pothole_class[class_ids[i]]}: {confs[i]:.2f}"
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(
            image_cv,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    save_path = f"output/images/detection_result_{len(os.listdir('output/images'))}.jpg"
    cv2.imwrite(save_path, image_cv)
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)), save_path


def detect_potholes_video(video_path, model, confidence, max_frames, selected_classes):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_width = 640
    target_height = int(target_width * frame_height / frame_width)
    output_size = (target_width, target_height)

    os.makedirs("output/videos", exist_ok=True)
    original_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"output/videos/{original_filename}_predict.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    if not out.isOpened():
        st.error("Failed to create video writer")
        return None

    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_count = 0

    try:
        with contextlib.redirect_stdout(StringIO()):
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, output_size)
                results = model(frame, verbose=False, conf=confidence)
                detections = results[0]

                boxes = (
                    detections.boxes.xyxy.cpu().numpy() if len(detections) > 0 else []
                )
                confs = (
                    detections.boxes.conf.cpu().numpy() if len(detections) > 0 else []
                )
                class_ids = (
                    detections.boxes.cls.cpu().numpy().astype(int)
                    if len(detections) > 0
                    else []
                )

                if selected_classes:
                    filtered = [
                        (box, conf, class_id)
                        for box, conf, class_id in zip(boxes, confs, class_ids)
                        if pothole_class[class_id] in selected_classes
                    ]
                    if filtered:
                        boxes, confs, class_ids = zip(*filtered)
                    else:
                        boxes, confs, class_ids = [], [], []

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{pothole_class[class_ids[i]]}: {confs[i]:.2f}"
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                out.write(frame)
                frame_count += 1

                progress = min(frame_count / max_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{max_frames}")

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None
    finally:
        cap.release()
        out.release()
        status_text.empty()
        progress_bar.empty()

    if frame_count > 0 and os.path.exists(output_path):
        return str(Path(output_path).resolve())
    return None


def live_streaming(model, confidence, selected_classes):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check camera permissions.")
        return

    try:
        while st.session_state.get("is_detecting", False) and st.session_state.get(
            "is_webcam_active", False
        ):
            ret, frame = cap.read()
            if not ret:
                st.warning("Warning: Failed to read frame from the webcam. Retrying...")
                continue

            try:
                results = model.predict(source=frame, conf=confidence)
                detections = results[0]

                boxes = (
                    detections.boxes.xyxy.cpu().numpy() if len(detections) > 0 else []
                )
                confs = (
                    detections.boxes.conf.cpu().numpy() if len(detections) > 0 else []
                )
                class_ids = (
                    detections.boxes.cls.cpu().numpy().astype(int)
                    if len(detections) > 0
                    else []
                )

                if selected_classes:
                    filtered = [
                        (box, conf, class_id)
                        for box, conf, class_id in zip(boxes, confs, class_ids)
                        if pothole_class[class_id] in selected_classes
                    ]
                    if filtered:
                        boxes, confs, class_ids = zip(*filtered)
                    else:
                        boxes, confs, class_ids = [], [], []

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{pothole_class[class_ids[i]]}: {confs[i]:.2f}"
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                stframe.image(frame, channels="BGR")

            except Exception as e:
                st.error(f"Error during model prediction: {str(e)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# Navigation function for back button
def go_back():
    if st.session_state.previous_page != st.session_state.page:
        st.session_state.page = st.session_state.previous_page
        st.session_state.is_webcam_active = False
        st.session_state.is_detecting = False


# Main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Display back button on all pages except home
if st.session_state.page != "home":
    if st.button("← Back", key="back_button"):
        st.session_state.page = st.session_state.previous_page
        st.session_state.is_webcam_active = False
        st.session_state.is_detecting = False
        st.rerun()

if st.session_state.page == "home":
    st.markdown('<h1 class="section-title">Pothole Vision</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Advanced AI-powered pothole detection for road safety and maintenance. Select a detection method from the sidebar.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    # Define card data with their corresponding pages
    cards = [
        (
            col1,
            "Image Detection",
            "Upload images to detect potholes instantly.",
            "fa-image",
            "image",
        ),
        (
            col2,
            "Video Detection",
            "Process videos to identify potholes in footage.",
            "fa-video",
            "video",
        ),
        (
            col3,
            "Real-time Detection",
            "Use your camera for live pothole detection.",
            "fa-camera",
            "realtime",
        ),
    ]

    for col, title, desc, icon, page in cards:
        with col:
            card = st.container()
            with card:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <i class="fas {icon} feature-icon"></i>
                        <h3 style="color: #1e293b; font-weight: 600; font-size: clamp(1.1rem, 3vw, 1.3rem);">{title}</h3>
                        <p style="color: #64748b; font-size: clamp(0.8rem, 2.5vw, 0.9rem);">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Open " + title, key=f"btn_{page}"):
                    st.session_state.previous_page = st.session_state.page
                    st.session_state.page = page
                    st.rerun()

elif st.session_state.page == "image":
    st.markdown(
        '<h2 class="section-title">Image Detection</h2>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="section-subtitle">Upload an image to detect potholes with AI precision.</p>',
        unsafe_allow_html=True,
    )

    # Upload area
    st.markdown(
        """
        <div class="upload-area">
            <i class="fas fa-cloud-upload-alt"></i>
            <h4>Upload Image</h4>
            <p>PNG, JPG, or JPEG (max. 10MB)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "", type=["png", "jpg", "jpeg"], key="image_upload"
    )

    # Settings
    st.subheader("Settings")
    st.session_state.confidence = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence,
        step=0.05,
        key="image_confidence",
    )
    selected_classes = st.multiselect(
        "Select classes for detection",
        pothole_class,
        default=pothole_class,
        key="image_classes",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        if st.button("Detect Potholes", key="detect_img"):
            with st.spinner("Analyzing image..."):
                detected_image, save_path = detect_potholes_image(
                    image, model, st.session_state.confidence, selected_classes
                )
                st.session_state.detected_image = detected_image
                st.session_state.save_path = save_path

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            if hasattr(st.session_state, "detected_image"):
                st.image(
                    st.session_state.detected_image,
                    caption="Detection Result",
                    use_column_width=True,
                )
                st.success(
                    f"✅ Analysis complete! Saved to {st.session_state.save_path}"
                )

elif st.session_state.page == "video":
    st.title("Video Detection")
    st.write("Upload a video to analyze and detect potholes.")

    # Settings
    st.subheader("Settings")
    st.session_state.confidence = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence,
        step=0.05,
        key="video_confidence",
    )
    max_frames = st.number_input(
        "Max Frames to Process",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
        key="video_max_frames",
    )
    selected_classes = st.multiselect(
        "Select classes for detection",
        pothole_class,
        default=pothole_class,
        key="video_classes",
    )

    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"], key="video_upload")

    if uploaded_file:
        temp_path = f"temp_video_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("Original Video")
        st.video(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Process Video", key="process_video"):
                with st.spinner("Processing video..."):
                    output_path = detect_potholes_video(
                        temp_path,
                        model,
                        st.session_state.confidence,
                        max_frames,
                        selected_classes,
                    )
                    if output_path and os.path.exists(output_path):
                        st.success("✅ Processing complete!")
                        st.subheader("Processed Video with Detections")
                        with open(output_path, "rb") as vid_file:
                            st.video(vid_file.read())
                            st.success(f"✅ Saved as: {os.path.basename(output_path)}")
                    else:
                        st.error("Failed to process video. Please try again.")

        if os.path.exists(temp_path):
            os.remove(temp_path)

elif st.session_state.page == "realtime":
    st.title("Real-time Detection")
    st.write("Use your camera for live pothole detection.")

    # Settings
    st.subheader("Settings")
    st.session_state.confidence = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence,
        step=0.05,
        key="realtime_confidence",
    )
    selected_classes = st.multiselect(
        "Select classes for detection",
        pothole_class,
        default=pothole_class,
        key="realtime_classes",
    )
    if st.button(
        "Use Webcam 📷" if not st.session_state.is_webcam_active else "Stop Webcam 🛑",
        key="webcam_toggle",
    ):
        st.session_state.is_webcam_active = not st.session_state.is_webcam_active
        if st.session_state.is_webcam_active:
            st.session_state.is_detecting = True
        else:
            st.session_state.is_detecting = False

    feed_placeholder = st.empty()

    if st.session_state.is_detecting and st.session_state.is_webcam_active:
        st.info("Detecting potholes using webcam...")
        live_streaming(model, st.session_state.confidence, selected_classes)
    else:
        feed_placeholder.write(
            "Camera Off. Click 'Use Webcam' to begin real-time detection"
        )

st.markdown("</div>", unsafe_allow_html=True)
