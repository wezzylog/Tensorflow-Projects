import argparse
import os

import cv2
import mediapipe as mp


def blur_faces(img, face_detection):
    """Detect faces in a BGR image and blur them in place."""
    h_img, w_img, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if not results.detections:
        return img

    for detection in results.detections:
        box = detection.location_data.relative_bounding_box

        # Convert to pixels and clamp to image bounds
        x1 = max(int(box.xmin * w_img), 0)
        y1 = max(int(box.ymin * h_img), 0)
        x2 = min(int((box.xmin + box.width) * w_img), w_img)
        y2 = min(int((box.ymin + box.height) * h_img), h_img)

        if x2 <= x1 or y2 <= y1:
            continue

        face = img[y1:y2, x1:x2]

        # Kernel size scales with face size; must be odd
        k = max(3, (min(face.shape[:2]) // 3) | 1)
        img[y1:y2, x1:x2] = cv2.GaussianBlur(face, (k, k), 0)

    return img


def process_image(path, output_dir, face_detection):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = blur_faces(img, face_detection)
    out_path = os.path.join(output_dir, "output.png")
    cv2.imwrite(out_path, img)
    print(f"Saved {out_path}")


def process_video(path, output_dir, face_detection):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    if not ret:
        raise FileNotFoundError(f"Could not read video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    h, w = frame.shape[:2]
    out_path = os.path.join(output_dir, "output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    try:
        while ret:
            writer.write(blur_faces(frame, face_detection))
            ret, frame = cap.read()
    finally:
        cap.release()
        writer.release()
    print(f"Saved {out_path}")


def process_webcam(face_detection):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Face blur (press q to quit)", blur_faces(frame, face_detection))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Blur faces in images, videos or webcam.")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], default="webcam")
    parser.add_argument("--filePath", help="Input file (required for image/video modes)")
    args = parser.parse_args()

    if args.mode in ("image", "video") and not args.filePath:
        parser.error(f"--filePath is required in {args.mode} mode")

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:
        if args.mode == "image":
            process_image(args.filePath, output_dir, face_detection)
        elif args.mode == "video":
            process_video(args.filePath, output_dir, face_detection)
        else:
            process_webcam(face_detection)


if __name__ == "__main__":
    main()
