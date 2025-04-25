import cv2
import numpy as np
import tensorflow as tf

# Define connections between keypoints (joint pairs to draw lines)
KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]

class PoseDetector:
    def __init__(self, model_url=r"C:\TII\Automations\AI-pose-detection\app\model\movenet"):
        self.model = tf.saved_model.load(model_url)
        self.movenet = self.model.signatures['serving_default']
    
    def detect(self, frame):
        """Process a single frame and return annotated image with keypoints"""
        # Resize and convert
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
        img = tf.cast(img, dtype=tf.int32)
        
        # Inference
        outputs = self.movenet(img)
        keypoints = outputs['output_0'].numpy()[0][0]
        
        # Visualization
        annotated_frame = frame.copy()
        self._draw_keypoints(annotated_frame, keypoints)
        self._draw_connections(annotated_frame, keypoints)
        
        return annotated_frame, keypoints
    
    def _draw_keypoints(self, frame, keypoints):
        for i, (y, x, conf) in enumerate(keypoints):
            if conf > 0.3:
                x = int(x * frame.shape[1])
                y = int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    def _draw_connections(self, frame, keypoints):
        for a, b in KEYPOINT_EDGES:
            ya, xa, ca = keypoints[a]
            yb, xb, cb = keypoints[b]
            if ca > 0.3 and cb > 0.3:
                xa = int(xa * frame.shape[1])
                ya = int(ya * frame.shape[0])
                xb = int(xb * frame.shape[1])
                yb = int(yb * frame.shape[0])
                cv2.line(frame, (xa, ya), (xb, yb), (0, 0, 255), 2)

def run_live_detection():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        result, _ = detector.detect(frame)
        cv2.imshow('Pose Detection', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()