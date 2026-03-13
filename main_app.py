"""
Main application - Command-line interface for Object Memory Assistant
Usage: python main_app.py --help
"""

import cv2
import argparse
import logging
import time
from typing import Optional
import config
from detection import create_detector
from tracking import create_tracker
from utils import create_deduplicator, FrameProcessor, setup_logging, RaspberryPiOptimizer
from memory import create_memory
from gemini_api import create_scene_descriptor
from query import create_query_engine

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

class ObjectMemoryApp:
    """Main application class"""
    
    def __init__(self, args):
        """Initialize application"""
        self.args = args
        
        # Optimize for RPi if needed
        RaspberryPiOptimizer.optimize_for_rpi()
        
        # Initialize components
        logger.info("Initializing components...")
        
        self.detector = create_detector(
            use_tflite=args.use_tflite or config.USE_TFLITE,
            device=args.device or config.INFERENCE_DEVICE
        )
        
        self.tracker = create_tracker("bytetrack")
        self.deduplicator = create_deduplicator()
        self.memory = create_memory()
        
        self.scene_descriptor = None
        if args.enable_gemini:
            self.scene_descriptor = create_scene_descriptor()
        
        self.query_engine = create_query_engine(self.memory)
        
        self.frame_count = 0
        self.paused = False
    
    def run_webcam(self):
        """Run webcam detection"""
        logger.info("Starting webcam detection mode")
        logger.info("Controls: 'q'=quit, 's'=save, 'p'=pause, 'h'=help")
        
        cap = cv2.VideoCapture(self.args.camera)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.args.camera}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"✓ Camera opened: "
                   f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
                   f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
                   f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}FPS")
        
        frame_times = []
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if self.paused:
                    self.display_frame(frame, [])
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.paused = False
                    continue
                
                # Skip frames if configured
                self.frame_count += 1
                if self.frame_count % config.INFERENCE_SKIP_FRAMES != 0:
                    continue
                
                # Inference
                start_time = time.time()
                
                det_result = self.detector.detect(frame, self.args.conf)
                detections = det_result.get('detections', [])
                
                # Tracking
                tracked_objects = self.tracker.update(detections)
                
                # Frame evaluation
                dedup_result = self.deduplicator.evaluate(
                    frame, detections, tracked_objects
                )
                
                # Store important frames
                if dedup_result.should_store and dedup_result.importance_score > 0.4:
                    frame_path = FrameProcessor.save_frame(
                        frame, prefix="important"
                    )
                    compressed_path = FrameProcessor.save_compressed_frame(frame)
                    
                    # Generate scene description
                    scene_desc = None
                    if self.scene_descriptor and detections:
                        try:
                            scene_desc = self.scene_descriptor.describe_scene(
                                frame, detections
                            )
                        except Exception as e:
                            logger.warning(f"Scene description error: {e}")
                    
                    # Store small objects
                    for det in detections:
                        if det['class_name'].lower() in config.SMALL_OBJECTS_TO_TRACK:
                            if det['confidence'] > self.args.conf:
                                obj_id = self.memory.store_object(
                                    object_name=det['class_name'],
                                    bbox=det['bbox'],
                                    confidence=det['confidence'],
                                    scene_description=scene_desc,
                                    image_path=frame_path,
                                    class_id=det['class_id']
                                )
                                
                                if obj_id:
                                    logger.info(f"✓ Stored: {det['class_name']} "
                                              f"(conf: {det['confidence']:.2f})")
                
                # Display
                elapsed = (time.time() - start_time) * 1000
                frame_times.append(elapsed)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                avg_time = sum(frame_times) / len(frame_times)
                fps = 1000.0 / avg_time if avg_time > 0 else 0
                
                self.display_frame(frame, detections, fps, elapsed, dedup_result.reason)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_frame(frame)
                elif key == ord('p'):
                    self.paused = True
                    logger.info("⏸️  Paused (press 'p' to resume)")
                elif key == ord('h'):
                    self.show_help()
            
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam detection stopped")
    
    def display_frame(self, frame, detections, fps=0, inference_time=0, reason=""):
        """Display frame with detections"""
        display = FrameProcessor.draw_detections(
            frame, detections, highlight_small_objects=True
        )
        
        # Add FPS and other info
        info_text = f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms"
        cv2.putText(display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if reason:
            cv2.putText(display, reason, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Display
        cv2.imshow("Object Memory Assistant", display)
    
    def save_frame(self, frame):
        """Save current frame"""
        path = FrameProcessor.save_frame(frame, prefix="user_saved")
        if path:
            logger.info(f"✓ Frame saved: {path}")
        else:
            logger.error("Failed to save frame")
    
    def show_help(self):
        """Show keyboard shortcuts"""
        help_text = """
        Keyboard Controls:
        q - Quit
        s - Save current frame
        p - Pause/Resume
        h - Show this help
        
        Green boxes = Small/Important objects (being tracked)
        Orange boxes = Regular objects
        """
        logger.info(help_text)
    
    def query_object(self, object_name: str):
        """Query object location"""
        response = self.query_engine.get_last_seen(object_name)
        logger.info(f"\nQuery Result:\n{response}\n")
    
    def show_statistics(self):
        """Show database statistics"""
        stats = self.memory.get_statistics()
        logger.info(f"\n=== Database Statistics ===")
        logger.info(f"Total objects stored: {stats.get('total_objects', 0)}")
        logger.info(f"Unique object types: {stats.get('unique_objects', 0)}")
        logger.info(f"Total frames stored: {stats.get('total_frames', 0)}")
        logger.info(f"Average confidence: {stats.get('avg_confidence', 0):.2f}")
        logger.info(f"========================\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Object Memory Assistant - Find misplaced items",
        epilog="""
Examples:
  # Run webcam detection
  python main_app.py

  # Run on Raspberry Pi with TFLite
  python main_app.py --use-tflite --camera 0

  # Query for object
  python main_app.py --query phone

  # Show statistics
  python main_app.py --stats
        """
    )
    
    # Webcam options
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device: 'cpu' or GPU ID (default: auto)")
    parser.add_argument("--use-tflite", action="store_true",
                       help="Use TensorFlow Lite (for Raspberry Pi)")
    
    # Feature options
    parser.add_argument("--enable-gemini", action="store_true",
                       help="Enable Gemini Vision API for scene descriptions")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save all frames (not just important ones)")
    
    # Query/Statistics options
    parser.add_argument("--query", type=str,
                       help="Query for object location (e.g., --query phone)")
    parser.add_argument("--stats", action="store_true",
                       help="Show database statistics")
    parser.add_argument("--history", type=str,
                       help="Show history for object (e.g., --history keys)")
    
    # System options
    parser.add_argument("--headless", action="store_true",
                       help="Run without display (for Raspberry Pi headless mode)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up data older than 30 days")
    
    args = parser.parse_args()
    
    try:
        app = ObjectMemoryApp(args)
        
        # Handle different modes
        if args.query:
            app.query_object(args.query)
        elif args.history:
            response = app.query_engine.get_object_history(args.history)
            logger.info(f"\nHistory for '{args.history}':\n{response}\n")
        elif args.stats:
            app.show_statistics()
        elif args.cleanup:
            app.memory.cleanup_old_data(30)
            logger.info("✓ Cleanup complete")
        else:
            # Default: run webcam detection
            app.run_webcam()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        if hasattr(app, 'memory'):
            app.memory.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
