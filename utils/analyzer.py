import cv2
import numpy as np
from deepface import DeepFace

# Import utils
try:
    from .color_utils import hsv_stats, blur_laplacian
except ImportError:
    from color_utils import hsv_stats, blur_laplacian


# --------------------------------------------------------
# 1. REALISTIC EYE-BASED JAUNDICE DETECTION
# --------------------------------------------------------
def detect_jaundice_eye(frame):
    """Detect jaundice using sclera (white part of eye) instead of skin color."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    jaundice_score = 0
    detected = False

    for (x, y, w, h) in eyes:
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_mean = np.mean(hsv[:, :, 0])  # hue
        s_mean = np.mean(hsv[:, :, 1])  # saturation
        v_mean = np.mean(hsv[:, :, 2])  # brightness

        # TRUE MEDICAL JAUNDICE CONDITIONS (for sclera only)
        if (18 < h_mean < 45) and s_mean > 70 and v_mean > 120:
            jaundice_score += 1
            detected = True

    return detected, jaundice_score


# --------------------------------------------------------
# 2. SKIN REGION EXTRACTION
# --------------------------------------------------------
def analyze_skin_regions(frame):
    """Extract facial regions for Ayurvedic skin diagnostics."""
    height, width = frame.shape[:2]

    regions = {
        'forehead': frame[int(height*0.1):int(height*0.25), int(width*0.3):int(width*0.7)],
        'cheeks': frame[int(height*0.30):int(height*0.60), int(width*0.15):int(width*0.85)],
        'chin': frame[int(height*0.65):int(height*0.90), int(width*0.35):int(width*0.65)]
    }

    region_data = {}
    for name, roi in regions.items():
        if roi.size:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            region_data[name] = {
                'mean_h': np.mean(hsv[:, :, 0]),
                'mean_s': np.mean(hsv[:, :, 1]),
                'mean_v': np.mean(hsv[:, :, 2]),
                'std_v': np.std(hsv[:, :, 2])
            }

    return region_data


# --------------------------------------------------------
# 3. EYE FEATURES – DARK CIRCLES, REDNESS, PUFFINESS
# --------------------------------------------------------
def detect_eye_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 5)

    data = {'dark_circles': False, 'redness': False, 'puffiness': False}

    for (x, y, w, h) in eyes:
        under_eye = frame[y+h:y+h+20, x:x+w]
        if under_eye.size == 0:
            continue

        u_gray = cv2.cvtColor(under_eye, cv2.COLOR_BGR2GRAY)
        avg_b = np.mean(u_gray)

        if avg_b < 80:
            data['dark_circles'] = True

        b, g, r = cv2.split(under_eye)
        if np.mean(r) > np.mean(g) + 15:
            data['redness'] = True

        if np.std(u_gray) < 18 and avg_b > 110:
            data['puffiness'] = True

    return data


# --------------------------------------------------------
# 4. SKIN TEXTURE (Ayurvedic Mukha Pariksha)
# --------------------------------------------------------
def analyze_skin_texture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    texture_var = np.var(gray)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness_std = np.std(hsv[:, :, 2])

    return {
        "texture_variance": texture_var,
        "edge_density": edge_density,
        "brightness_std": brightness_std
    }


# --------------------------------------------------------
# 5. FACIAL SYMMETRY – Neurological Imbalance
# --------------------------------------------------------
def detect_facial_symmetry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    L = gray[:, :w//2]
    R = cv2.flip(gray[:, w//2:], 1)

    min_w = min(L.shape[1], R.shape[1])
    diff = cv2.absdiff(L[:, :min_w], R[:, :min_w])

    return np.mean(diff)  # asymmetry score


# --------------------------------------------------------
# MAIN ANALYZER
# --------------------------------------------------------
def analyze_frame(frame):
    data = {
        "emotion": None,
        "race": None,
        "age": None,
        "diseases": [],
        "health_score": 100,
        "ayurvedic_summary": "Tri-Dosha Balanced",
        "recommendations": []
    }

    # DeepFace (Optional if fails)
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion', 'age', 'race'],
            enforce_detection=False,
            detector_backend='opencv'
        )[0]

        data["emotion"] = results.get("dominant_emotion", "Unknown")
        data["race"] = results.get("dominant_race", "Unknown")
        data["age"] = results.get("age", "Unknown")
    except:
        pass

    # Feature extraction
    hsv_mean = hsv_stats(frame)
    blur_val = blur_laplacian(frame)
    region_data = analyze_skin_regions(frame)
    eye_data = detect_eye_features(frame)
    texture = analyze_skin_texture(frame)
    asymm = detect_facial_symmetry(frame)

    avg_h, avg_s, avg_v = hsv_mean["h"], hsv_mean["s"], hsv_mean["v"]

    diseases = []
    health_score = 100

    # --------------------------------------------------------
    # ★ 1. TRUE JAUNDICE (EYE-BASED)
    # --------------------------------------------------------
    jaundice_detected, j_score = detect_jaundice_eye(frame)

    if jaundice_detected:
        diseases.append({
            "name": "Jaundice (Liver Issue)",
            "ayurvedic_term": "Kamala (Pitta Aggravation)",
            "confidence": "High" if j_score >= 2 else "Medium",
            "description": "Yellow sclera suggests liver imbalance.",
            "remedy": "Avoid spicy/oily food. Take sugarcane juice & Kutki herbs."
        })
        health_score -= 30

    # --------------------------------------------------------
    # ★ 2. VITILIGO IMPROVED DETECTION
    # --------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 200, 255)
    white_ratio = np.sum(bright == 255) / bright.size

    if white_ratio > 0.20 and avg_s < 55 and texture['brightness_std'] > 35:
        diseases.append({
            "name": "Vitiligo / Hypopigmentation",
            "ayurvedic_term": "Shwitra",
            "confidence": "Medium",
            "description": "Patchy depigmentation detected.",
            "remedy": "Apply Bakuchi oil; morning sunlight recommended."
        })
        health_score -= 15

    # --------------------------------------------------------
    # ★ 3. ANEMIA
    # --------------------------------------------------------
    if avg_s < 40 and avg_v > 140:
        diseases.append({
            "name": "Anemia",
            "ayurvedic_term": "Pandu Roga",
            "confidence": "Medium",
            "description": "Pale complexion detected.",
            "remedy": "Eat beetroot, pomegranate, leafy greens."
        })
        health_score -= 15

    # --------------------------------------------------------
    # ★ 4. ACNE
    # --------------------------------------------------------
    if texture['edge_density'] > 0.12 and texture['texture_variance'] > 600:
        diseases.append({
            "name": "Acne / Blemishes",
            "ayurvedic_term": "Mukha Dushika",
            "confidence": "Medium",
            "description": "Blemish & pore density elevated.",
            "remedy": "Neem water facewash; avoid oily foods."
        })
        health_score -= 12

    # --------------------------------------------------------
    # ★ 5. ECZEMA / DRYNESS
    # --------------------------------------------------------
    if texture['texture_variance'] > 800 and avg_s < 55:
        diseases.append({
            "name": "Dry Skin / Eczema",
            "ayurvedic_term": "Vicharchika",
            "confidence": "Low",
            "description": "Skin roughness detected.",
            "remedy": "Sesame oil massage; hydrate well."
        })
        health_score -= 10

    # --------------------------------------------------------
    # ★ 6. DEHYDRATION
    # --------------------------------------------------------
    if avg_v < 90 and blur_val < 60:
        diseases.append({
            "name": "Dehydration",
            "ayurvedic_term": "Udakavaha Srotas Dushti",
            "confidence": "Medium",
            "description": "Low moisture & skin dullness detected.",
            "remedy": "Drink more water; eat juicy fruits."
        })
        health_score -= 10

    # --------------------------------------------------------
    # ★ 7. MENTAL STRESS
    # --------------------------------------------------------
    if data["emotion"] in ["sad", "fear", "angry"]:
        diseases.append({
            "name": "Stress / Anxiety",
            "ayurvedic_term": "Chittodvega",
            "confidence": "Medium",
            "description": "Emotional imbalance visible.",
            "remedy": "Meditation, Shirodhara, Brahmi tea."
        })
        health_score -= 10

    # --------------------------------------------------------
    # ★ HEALTHY
    # --------------------------------------------------------
    if len(diseases) == 0:
        diseases.append({
            "name": "Healthy",
            "ayurvedic_term": "Swastha",
            "confidence": "High",
            "description": "No major imbalances detected.",
            "remedy": "Continue healthy routine."
        })

    # Collect recommendations
    recommendations = [f"{d['ayurvedic_term']}: {d['remedy']}" for d in diseases]

    # Dosha summary
    if eye_data['dark_circles'] or texture["texture_variance"] > 800:
        data["ayurvedic_summary"] = "Vata Dominant"
    elif eye_data['redness'] or avg_h < 20:
        data["ayurvedic_summary"] = "Pitta Dominant"
    elif eye_data['puffiness'] or (avg_v > 160 and avg_s < 40):
        data["ayurvedic_summary"] = "Kapha Dominant"

    data["diseases"] = diseases
    data["health_score"] = max(0, health_score)
    data["recommendations"] = recommendations

    return data
