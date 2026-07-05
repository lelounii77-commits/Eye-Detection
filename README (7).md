# Eye Detection Using Python & OpenCV

Assignment 3

## Objective

Develop a computer vision application using Python and OpenCV to automatically detect human eyes in an uploaded image using a pre-trained Haar Cascade classifier.

The program performs image loading, grayscale conversion, eye region detection, and visualization of the detected areas to illustrate the performance of the detection process.

## How It Works

1. **Load the image** — the input image is read from disk with OpenCV
2. **Convert to grayscale** — improves detection accuracy and speed
3. **Load the Haar Cascade classifier** — uses OpenCV's built-in pre-trained `haarcascade_eye.xml` model
4. **Detect eyes** — `detectMultiScale()` scans the grayscale image and returns bounding boxes for detected eye regions
5. **Draw bounding boxes** — green rectangles are drawn around each detected eye
6. **Display and save the result** — the final image is shown with matplotlib and saved to disk

## Requirements

- Python 3.8+
- opencv-python
- numpy
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python eye_detection.py --image path/to/your_image.jpg
```

If no `--image` argument is provided, the script looks for a file named `eye.jpg` in the current directory.

The detected result is saved as `detected_eyes.jpg` by default (customizable with `--output`).

## Results

The model successfully detects human eyes in well-lit, frontal images, highlighting each detected eye region with a green bounding box.

## Conclusion

This experiment demonstrates that Haar Cascade is capable of detecting eye regions effectively in well-lit frontal images. However, accuracy may degrade in cases involving head rotations, occlusions, or dim lighting. For improved performance, advanced deep learning detectors such as DNN face/eye detectors or MTCNN are recommended.

## Author

Layan Alfares
