import cv2
import os

VIDEO_PATH = "video/latlongvideo1.mp4"      # ← change this to your video file name
OUTPUT_FOLDER = "frames_latlongvideo1"       # folder to save extracted images

def extract_frames():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open video file")
        return

    frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # video finished
        # if frame_index % 2 != 0:
        #     frame_index += 1
            # continue
        filename = f"{frame_index}.jpeg"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        cv2.imwrite(filepath, frame)  # save frame
        print(f"[SAVED] {filepath}")

        frame_index += 1

    cap.release()
    print("\n✔ Extraction Complete!")


if __name__ == "__main__":
    extract_frames()
