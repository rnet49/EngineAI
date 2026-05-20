import io
import os
import numpy as np
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from scipy.signal import iirnotch, lfilter


def preprocess_audio(y, sr):
    y = y - np.mean(y)
    b, a = iirnotch(50, 30, sr)
    y = lfilter(b, a, y)
    threshold = np.percentile(np.abs(y), 99)
    y = np.clip(y, -threshold, threshold)
    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)
    median_mag = np.median(magnitude, axis=1, keepdims=True)
    mask = np.minimum(1.0, median_mag / (magnitude + 1e-8))
    D_filtered = (magnitude * mask) * np.exp(1j * phase)
    y = librosa.istft(D_filtered, length=len(y))
    return y


import sys

# рядом с exe если собрано, рядом с py если запускать напрямую
if hasattr(sys, '_MEIPASS'):
    MODELS_DIR = os.path.join(os.path.dirname(sys.executable), "models")
else:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "pca_model.pkl")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "threshold.npy")


def load_dataset(folder="dataset"):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        images.append(np.array(img).flatten() / 255.0)
    return np.array(images)


def train(folder="dataset", n_components=150):
    X = load_dataset(folder)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    errors = np.mean((X - X_reconstructed) ** 2, axis=1)
    threshold = errors.mean() + 2 * errors.std()
    joblib.dump(pca, MODEL_FILE)
    np.save(THRESHOLD_FILE, threshold)
    print(f"Обучено на {len(X)} картинках, порог: {threshold:.6f}")
    return pca, threshold


def is_anomaly(audio_path, model=None, threshold=None):
    if model is None:
        model = joblib.load(MODEL_FILE)
    if threshold is None:
        threshold = float(np.load(THRESHOLD_FILE))

    y, sr = librosa.load(audio_path, sr=None)

    # берём самые информативные 5 секунд — окно с максимальной энергией
    window = 5 * sr
    if len(y) > window:
        step = sr // 4  # шаг 0.25 сек
        best_start = 0
        best_energy = -1
        for i in range(0, len(y) - window, step):
            energy = np.mean(y[i:i + window] ** 2)
            if energy > best_energy:
                best_energy = energy
                best_start = i
        y = y[best_start:best_start + window]

    y = preprocess_audio(y, sr)
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    librosa.display.specshow(spectrogram, sr=sr, ax=ax, cmap="inferno")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_orig = Image.open(buf).convert("RGB")

    x = np.array(img_orig).flatten() / 255.0
    x_rec = model.inverse_transform(model.transform(x.reshape(1, -1)))
    img_rec = Image.fromarray((x_rec.reshape(np.array(img_orig).shape) * 255).astype(np.uint8))
    error = float(np.mean((x - x_rec) ** 2))

    print(f"{os.path.basename(audio_path)}: {'АНОМАЛИЯ' if error > threshold else 'норма'} (ошибка={error:.6f})")
    return {
        "is_anomaly": error > threshold,
        "error": error,
        "threshold": threshold,
        "spectrogram": spectrogram,
        "img_orig": img_orig,
        "img_rec": img_rec,
        "sr": sr,
    }
