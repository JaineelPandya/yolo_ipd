"""
Streamlit web interface for Object Memory Assistant
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

import config
from detection.detector import create_detector
from tracking.tracker import create_tracker
from utils.deduplicator import create_deduplicator
from memory.storage import create_memory
from gemini_api.descriptor import create_scene_descriptor
from query.engine import create_query_engine
from utils.helpers import FrameProcessor, PerformanceMonitor, setup_logging, RaspberryPiOptimizer

# Setup
setup_logging("INFO")
logger = logging.getLogger(__name__)
RaspberryPiOptimizer.optimize_for_rpi()

# Page config
st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    layout=config.STREAMLIT_LAYOUT,
    initial_sidebar_state=config.STREAMLIT_INITIAL_SIDEBAR_STATE
)

# CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .detection-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .small-object {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# =================== SIDEBAR ===================
st.sidebar.title("⚙️ Configuration")

# Model selection
model_option = st.sidebar.radio(
    "Model Type",
    ("PyTorch (Full Quality)", "TensorFlow Lite (Fast - RPi)"),
    help="Choose based on your hardware"
)
use_tflite = model_option == "TensorFlow Lite (Fast - RPi)"

# Confidence threshold - DEFAULT TO 0.25 for better detection
conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,  # Changed from 0.5 to 0.25 for better detection
    step=0.05,
    help="Lower = More detections (even weak ones). Default 0.25 for better coverage. Adjust if too many false positives."
)

# Enable Gemini API
enable_gemini = st.sidebar.checkbox(
    "Enable Scene Description (Requires API Key)",
    value=False,
    help="Generate scene descriptions using Gemini Vision API"
)

# Device info
with st.sidebar.expander("Device Info"):
    device_info = RaspberryPiOptimizer.get_device_info()
    st.json(device_info)

# Troubleshooting guide
with st.sidebar.expander("🆘 Troubleshooting"):
    st.markdown("""
    **⚠️ Not detecting objects?**
    
    1. **Lower confidence threshold** (try 0.20-0.30)
    2. **Check webcam:**
       ```bash
       python test_detection.py
       ```
    3. **Good lighting is important** 📝
    4. **Hold object closer to camera**
    5. **Try different angles**
    
    **Model not loading?**
    - Model file missing in `models/` folder
    - Run: `python setup_system.py`
    
    **Still not working?**
    - Check logs: `logs/system.log`
    - Or run: `python test_detection.py`
    """)

# =================== MAIN APP ===================

# Session state initialization
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'deduplicator' not in st.session_state:
    st.session_state.deduplicator = None
if 'scene_descriptor' not in st.session_state:
    st.session_state.scene_descriptor = None
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'perf_monitor' not in st.session_state:
    st.session_state.perf_monitor = None

# Load components
@st.cache_resource
def load_components(_use_tflite, _enable_gemini):
    """Load all required components for the system"""
    with st.spinner("Loading components..."):
        try:
            detector = create_detector(use_tflite=_use_tflite)
            tracker = create_tracker("bytetrack")
            memory = create_memory()
            deduplicator = create_deduplicator()
            scene_descriptor = create_scene_descriptor() if _enable_gemini else None
            query_engine = create_query_engine(memory)
            perf_monitor = PerformanceMonitor()
            
            logger.info(f"✓ Components loaded successfully (TFLite: {_use_tflite}, Gemini: {_enable_gemini})")
            return detector, tracker, memory, deduplicator, scene_descriptor, query_engine, perf_monitor
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            st.error(f"❌ Error loading components: {e}")
            return None, None, None, None, None, None, None

# Initialize components - recreate cache if settings change
if st.session_state.detector is None:
    (st.session_state.detector, 
     st.session_state.tracker,
     st.session_state.memory,
     st.session_state.deduplicator,
     st.session_state.scene_descriptor,
     st.session_state.query_engine,
     st.session_state.perf_monitor) = load_components(use_tflite, enable_gemini)
elif st.session_state.scene_descriptor is None and enable_gemini:
    # User enabled Gemini after app started
    st.session_state.scene_descriptor = create_scene_descriptor()
    logger.info("✓ Gemini API enabled dynamically")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎥 Live Detection",
    "🔍 Query Objects",
    "📊 Statistics",
    "📚 History"
])

# ======================== TAB 1: LIVE DETECTION ========================
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🎥 Real-Time Object Detection (All Objects with Bounding Boxes)")
        
        # Start/Stop buttons
        start_cam = st.button("📹 Start Camera", key="start_cam")
        stop_cam = st.button("⏹️ Stop Camera", key="stop_cam")
        
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        detection_info = st.empty()
        
        if start_cam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot open webcam. Check device connection.")
            else:
                st.success("✓ Camera started - Showing ALL detections with bounding boxes")
                
                frame_count = 0
                detection_count_total = 0
                small_object_count = 0
                large_object_count = 0
                
                while not stop_cam and st.session_state.detector:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame")
                        break
                    
                    # Skip frames for efficiency
                    frame_count += 1
                    if frame_count % config.INFERENCE_SKIP_FRAMES != 0:
                        continue
                    
                    # Inference
                    import time
                    start_time = time.time()
                    
                    # Get detections with current confidence threshold
                    det_result = st.session_state.detector.detect(frame, conf_threshold)
                    detections = det_result.get('detections', [])
                    detection_count_total = len(detections)
                    
                    # Count small vs large objects
                    small_object_count = sum(1 for d in detections if d['class_name'].lower() in config.SMALL_OBJECTS_TO_TRACK)
                    large_object_count = detection_count_total - small_object_count
                    
                    # Tracking
                    tracked_objects = st.session_state.tracker.update(detections)
                    
                    # Frame deduplication
                    dedup_result = st.session_state.deduplicator.evaluate(
                        frame, detections, tracked_objects
                    )
                    
                    # Draw detections - NOW SHOWS ALL DETECTIONS
                    annotated_frame = FrameProcessor.draw_detections(
                        frame, detections, tracked_objects, 
                        show_track_id=True, 
                        highlight_small_objects=True
                    )
                    
                    # Convert BGR to RGB for display
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Performance monitoring
                    elapsed_time = (time.time() - start_time) * 1000
                    st.session_state.perf_monitor.add_frame_time(elapsed_time)
                    
                    # Display frame directly (no file cache issues)
                    frame_placeholder.image(display_frame, use_container_width=True)
                    
                    # Display detection info
                    with detection_info.container():
                        det_col1, det_col2, det_col3 = st.columns(3)
                        with det_col1:
                            st.metric("Total Detections", detection_count_total, 
                                     delta=f"Conf: {conf_threshold:.2f}")
                        with det_col2:
                            st.metric("🟢 Small Objects", small_object_count)
                        with det_col3:
                            st.metric("🟠 Other Objects", large_object_count)
                    
                    # Stats
                    stats = st.session_state.perf_monitor.get_stats()
                    with stats_placeholder.container():
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("FPS", f"{stats['fps']:.1f}")
                        with col_stats2:
                            st.metric("Frame Time", f"{stats['avg_frame_time_ms']:.1f}ms")
                        with col_stats3:
                            st.metric("Inference", f"{stats['avg_inference_time_ms']:.1f}ms")
                    
                    # Store important frames
                    if dedup_result.should_store and dedup_result.importance_score > 0.2:
                        frame_path = FrameProcessor.save_frame(
                            frame, prefix="important"
                        )
                        
                        # Generate scene description if enabled
                        scene_desc = None
                        if st.session_state.scene_descriptor and enable_gemini and len(detections) > 0:
                            try:
                                scene_desc = st.session_state.scene_descriptor.describe_scene(
                                    frame, detections
                                )
                            except Exception as e:
                                logger.warning(f"Scene description error: {e}")
                        
                        # Store ALL detections in memory (not just small objects)
                        for det in detections:
                            try:
                                obj_id = st.session_state.memory.store_object(
                                    object_name=det['class_name'],
                                    bbox=det['bbox'],
                                    confidence=det['confidence'],
                                    scene_description=scene_desc,
                                    image_path=frame_path,
                                    class_id=det['class_id']
                                )
                            except Exception as e:
                                logger.error(f"Error storing object: {e}")
                    
                    # Check stop flag
                    if stop_cam:
                        break
                
                cap.release()
                st.success("✓ Camera stopped")
    
    with col2:
        st.subheader("Detection Settings")
        st.metric("Skip Frames", config.INFERENCE_SKIP_FRAMES)
        st.metric("Max Tracks", config.MAX_TRACKS)
        st.metric("Tracking Objects", len(config.SMALL_OBJECTS_TO_TRACK))

# ======================== TAB 2: QUERY OBJECTS ========================
with tab2:
    st.subheader("🔍 Find Your Objects")
    
    # Query method selection
    query_method = st.radio(
        "How do you want to search?",
        ("By Object Name", "By Voice", "By Location")
    )
    
    if query_method == "By Object Name":
        object_name = st.text_input("Enter object name (e.g., phone, keys):")
        
        if st.button("Search"):
            if object_name:
                response = st.session_state.query_engine.get_last_seen(object_name)
                st.info(response)
            else:
                st.warning("Please enter an object name")
    
    elif query_method == "By Voice":
        st.info("🎤 Voice input feature (requires speech-to-text integration)")
        voice_query = st.text_area("Or type your query:")
        
        if st.button("Ask"):
            if voice_query:
                response = st.session_state.query_engine.process_voice_query(voice_query)
                st.info(response)
    
    elif query_method == "By Location":
        location = st.text_input("Enter location (e.g., cupboard, table, shelf):")
        
        if st.button("Search"):
            if location:
                results = st.session_state.query_engine.find_by_location(location)
                for result in results:
                    st.success(result)
            else:
                st.warning("Please enter a location")
    
    # Quick reference
    st.markdown("---")
    st.subheader("Quick Reference - Trackable Objects")
    objects_list = ", ".join(config.SMALL_OBJECTS_TO_TRACK[:15])
    st.caption(f"📦 {objects_list}...")

# ======================== TAB 3: STATISTICS ========================
with tab3:
    st.subheader("📊 System Statistics")
    
    if st.session_state.memory:
        # Get statistics
        stats = st.session_state.memory.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Objects Stored", stats.get('total_objects', 0) or 0)
        with col2:
            st.metric("Unique Objects", stats.get('unique_objects', 0) or 0)
        with col3:
            st.metric("Total Frames", stats.get('total_frames', 0) or 0)
        with col4:
            avg_conf = stats.get('avg_confidence', 0.0) or 0.0
            st.metric("Avg Confidence", f"{float(avg_conf):.2f}")
        
        # Today's summary
        st.markdown("### Today's Detections")
        today_objects = st.session_state.memory.get_all_objects_today()
        
        if today_objects:
            for obj in today_objects:
                st.write(f"- **{obj['object_name']}**: {obj['count']} sightings (last: {obj['last_seen']})")
        else:
            st.info("No objects tracked yet today")
    
    # Cleanup button
    if st.button("🧹 Clean old data (>30 days)"):
        st.session_state.memory.cleanup_old_data(30)
        st.success("Cleanup complete")

# ======================== TAB 4: HISTORY ========================
with tab4:
    st.subheader("📚 Object History")
    
    # Select object
    if st.session_state.memory:
        avg_objects = st.session_state.memory.get_all_objects_today()
        if avg_objects:
            available_objects = [obj['object_name'] for obj in avg_objects]
        else:
            available_objects = config.SMALL_OBJECTS_TO_TRACK
        
        selected_object = st.selectbox("Select object to view history:", available_objects)
        
        if selected_object and st.session_state.query_engine:
            history_limit = st.slider("Show last N sightings:", 1, 20, 5)
            
            history_text = st.session_state.query_engine.get_object_history(
                selected_object, history_limit
            )
            st.info(history_text)
    else:
        st.warning("⚠️ System not initialized. Please start camera first.")

# ==================== FOOTER ====================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.memory:
        st.caption(f"Database: {config.DATABASE_PATH}")

with col2:
    st.caption(f"v1.0 - AI Object Memory Assistant")

with col3:
    if config.IS_RASPBERRY_PI:
        st.caption("🍓 Running on Raspberry Pi")
