import os
import tempfile
import httpx
import anthropic
from flask import Flask, request, jsonify, send_from_directory
from autoencoder import is_anomaly

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/icon.png")
def icon():
    return send_from_directory("screenshots", "screenshot_main.PNG")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    f = request.files["file"]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        f.save(tmp.name)
        tmp.close()
        result = is_anomaly(tmp.name)
        ai_text = _get_ai(result["is_anomaly"], result["error"], result["threshold"], f.filename)
        return jsonify({
            "is_anomaly": result["is_anomaly"],
            "recon_error": round(result["error"], 6),
            "threshold": round(result["threshold"], 6),
            "ai": ai_text,
        })
    finally:
        os.unlink(tmp.name)


def _get_ai(anomaly, error, threshold, filename):
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            with open("api_key.txt", encoding="utf-8") as f:
                api_key = f.read().strip()
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.scarlex.ru",
            http_client=httpx.Client(proxy=None, trust_env=False),
        )
        status = "АНОМАЛИЯ" if anomaly else "НОРМА"
        msg = client.messages.create(
            model="claude-opus-4.7",
            max_tokens=300,
            messages=[{"role": "user", "content": (
                f"Ты эксперт по диагностике двигателей.\n"
                f"Файл: {filename}\nРезультат: {status}\n"
                f"Ошибка реконструкции: {error:.6f}\nПорог: {threshold:.6f}\n\n"
                f"Дай краткое профессиональное заключение (2-3 предложения)."
            )}],
        )
        return msg.content[0].text
    except FileNotFoundError:
        return "Создайте файл api_key.txt с вашим ключом"
    except Exception as e:
        return f"AI недоступен: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
