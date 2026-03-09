from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import joblib
import os
import tensorflow as tf

try:
    import serial
except ImportError:
    serial = None

app = Flask(__name__)

# =========================
#  MODEL LOADING
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Hybrid CNN + tabular gender model
    MODEL_PATH = os.path.join(BASE_DIR, "cnn_tabular_egg_gender_model.keras")
    model = tf.keras.models.load_model(MODEL_PATH)

    scaler_path = os.path.join(BASE_DIR, "numerical_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    le_path = os.path.join(BASE_DIR, "label_encoder.pkl")
    label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None

    print("✅ CNN hybrid model & preprocessors loaded successfully.")

    # Threshold tuning file (for sigmoid output)
    threshold_path = os.path.join(BASE_DIR, "best_threshold.txt")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            BEST_THRESHOLD = float(f.read().strip())
        print("✅ Loaded tuned threshold:", BEST_THRESHOLD)
    else:
        BEST_THRESHOLD = 0.5
        print("⚠️ best_threshold.txt not found. Using default threshold = 0.5")

except Exception as e:
    print(f"❌ Error loading gender model or preprocessors: {e}")
    model = None
    scaler = None
    label_encoder = None
    BEST_THRESHOLD = 0.5

# Image size for hybrid gender model
IMAGE_SIZE = (224, 224)  # height, width


ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 9600

arduino = None
if serial is not None:
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
        print("✅ Connected to Arduino at", ARDUINO_PORT)
    except Exception as e:
        print("⚠️ Could not open Arduino serial:", e)
else:
    print("⚠️ pyserial not installed; Arduino integration disabled.")


# =========================
#  IMAGE HELPERS
# =========================

def remove_background(image):
    """Otsu + largest contour mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    final_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, cv2.FILLED)

    masked = cv2.bitwise_and(image, image, mask=final_mask)
    return masked, largest_contour


def extract_features_from_contour(image, contour):
    """
    Returns:
        features_8, ellipse_dict, vis_image
        OR (None, None, None) if contour is not egg-like.
    """
    h_img, w_img = image.shape[:2]
    img_area = float(h_img * w_img)

    area = cv2.contourArea(contour)
    if area < 0.005 * img_area or area > 0.7 * img_area:
        return None, None, None

    contour = cv2.convexHull(contour)
    if len(contour) < 5:
        return None, None, None

    ellipse = cv2.fitEllipse(contour)

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), rect_angle = rect

    if w == 0 or h == 0:
        return None, None, None

    short_axis = min(w, h)
    long_axis = max(w, h)
    shape_index = short_axis / long_axis

    a = long_axis / 2.0
    b = short_axis / 2.0
    if a > 0:
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (b * b) / (a * a))))
    else:
        eccentricity = 0.0

    bbox_area = w * h
    extent = float(area) / bbox_area if bbox_area > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0

    ok_shape_index = 0.45 <= shape_index <= 0.9
    ok_ecc = 0.2 <= eccentricity <= 0.99
    ok_extent = 0.5 <= extent <= 0.95 
    ok_solidity = solidity >= 0.9

    if not (ok_shape_index and ok_ecc and ok_extent and ok_solidity):
        return None, None, None

    m = cv2.moments(contour)
    hu = cv2.HuMoments(m).flatten()
    Hu1, Hu2, Hu3 = float(hu[0]), float(hu[1]), float(hu[2])

    width = short_axis
    height = long_axis

    ellipse_dict = {
        "cx": float(ellipse[0][0]),
        "cy": float(ellipse[0][1]),
        "major": float(long_axis),
        "minor": float(short_axis),
        "angle": float(ellipse[2]),
    }

    vis_image = image.copy()
    cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(vis_image, [box], -1, (255, 0, 0), 2)
    cv2.ellipse(vis_image, ellipse, (0, 255, 255), 2)

    features = np.array(
        [width, height, shape_index, eccentricity, extent, Hu1, Hu2, Hu3],
        dtype=np.float32,
    )

    return features, ellipse_dict, vis_image


def prepare_cnn_input(original_frame):
    """Resize with aspect ratio preserved, pad to 224x224, normalize."""
    target_h, target_w = IMAGE_SIZE

    img = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target_h, target_w, 3), dtype=resized.dtype)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized

    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=0)
    return padded


# =========================
#  FLASK ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data received."}), 400

        # Decode base64 image
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Background removal & contour
        bg_removed, largest_contour = remove_background(frame)
        if bg_removed is None or largest_contour is None:
            return jsonify({"error": "No egg detected"}), 400

        # Feature extraction
        features_8, ellipse_dict, processed_image = extract_features_from_contour(
            bg_removed, largest_contour
        )
        if features_8 is None:
            return jsonify({"error": "No egg detected"}), 400

        # Scale numerical features
        if scaler is not None:
            feats_scaled = scaler.transform(features_8.reshape(1, -1))
        else:
            feats_scaled = features_8.reshape(1, -1)

        # CNN image input
        cnn_input = prepare_cnn_input(frame)

        preds = model.predict(
            {
                "image_input": cnn_input,
                "feat_input": feats_scaled,
            },
            verbose=0,
        )
        prob = float(preds[0][0])
        pred_class = int(prob >= BEST_THRESHOLD)

        if label_encoder is not None:
            gender_label = label_encoder.inverse_transform([pred_class])[0]
        else:
            gender_label = "Male" if pred_class == 1 else "Female"

        confidence = round(prob * 100, 2) if pred_class == 1 else round((1 - prob) * 100, 2)
        shape_index = float(features_8[2])

        # Encode processed image for frontend
        _, buffer = cv2.imencode(".jpg", processed_image)
        processed_image_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "gender": gender_label,
                "confidence": confidence,
                "shape_index": round(shape_index, 3),
                "ellipse": ellipse_dict,
                "processed_image": processed_image_b64,
            }
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route("/arduino-status")
def arduino_status():
    """
    Browser calls this to check if Arduino just sent 'CAPTURE'.
    Returns: {"capture": true/false}
    """
    capture = False

    if arduino is not None:
        try:
            line = arduino.readline().decode(errors="ignore").strip()
            if line:
                print("Arduino:", line)
            if line == "CAPTURE":
                capture = True
        except Exception as e:
            print("Arduino read error:", e)

    return jsonify({"capture": capture})


if __name__ == "__main__":
    app.run(debug=False)
