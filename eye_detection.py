"""
Assignment 3 - Eye Detection Using Python & OpenCV
----------------------------------------------------
Detects human eyes in an image using a pre-trained Haar Cascade
classifier, and draws bounding boxes around the detected regions.

Usage:
    python eye_detection.py --image path/to/image.jpg

If no --image argument is given, the script will look for an
image named "eye.jpg" in the current directory.
"""

import argparse
import cv2
import matplotlib.pyplot as plt


def detect_eyes(image_path: str, output_path: str = "detected_eyes.jpg"):
    """
    Loads an image, detects eyes using a Haar Cascade classifier,
    draws bounding boxes around them, and displays/saves the result.
    """

    # ---------- Load the image ----------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    # ---------- Convert to grayscale (improves detection accuracy) ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- Load the pre-trained Haar Cascade for eye detection ----------
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    # ---------- Detect eyes ----------
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    print(f"Detected {len(eyes)} eye region(s).")

    # ---------- Draw rectangles around detected eyes ----------
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ---------- Save the result ----------
    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")

    # ---------- Display the result ----------
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Eyes")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye Detection using OpenCV Haar Cascade")
    parser.add_argument(
        "--image", type=str, default="eye.jpg",
        help="Path to the input image (default: eye.jpg)"
    )
    parser.add_argument(
        "--output", type=str, default="detected_eyes.jpg",
        help="Path to save the output image (default: detected_eyes.jpg)"
    )
    args = parser.parse_args()

    detect_eyes(args.image, args.output)
