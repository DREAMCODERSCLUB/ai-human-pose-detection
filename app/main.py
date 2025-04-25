from model.pose_model import PoseDetector, run_live_detection
import cv2

def main():
    print("Starting pose detection...")
    run_live_detection()


if __name__ == "__main__":
    main()