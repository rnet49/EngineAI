import sys
import os
import time
import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import httpx
import anthropic
import joblib
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QCursor

from autoencoder import train, is_anomaly


def resource_path(filename):
    # внутри exe файлы лежат в sys._MEIPASS
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


UI_FILE = resource_path(os.path.join("assets", "main.ui"))
INFO_IMAGE = resource_path(os.path.join("assets", "info_diagram.png"))


class InfoPopup(QtWidgets.QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background-color: #151821; border: 1px solid #2b3242;")
        self.setFixedWidth(720)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Как работает алгоритм?")
        title.setStyleSheet("color: white; font-size: 13pt; font-weight: 900; border: none;")
        layout.addWidget(title)

        desc = QtWidgets.QLabel(
            "Система обучается на нормальных звуках двигателя. "
            "Каждая запись превращается в спектрограмму, используя оконное преобразование Фурье. "
            "PCA-модель запоминает как выглядит «норма» и пытается восстановить любую новую картинку. "
            "Если восстановление сильно отличается от оригинала — это аномалия."
        )
        desc.setStyleSheet("color: #9ca3af; font-size: 9pt; border: none;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        img_label = QtWidgets.QLabel()
        img_label.setStyleSheet("border: none;")
        try:
            pil_img = Image.open(image_path).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            if not pixmap.isNull():
                pixmap = pixmap.scaledToWidth(680, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
                img_label.setFixedSize(pixmap.size())
        except Exception:
            pass
        layout.addWidget(img_label)
        self.adjustSize()


# Загружает аудио в фоне чтобы не зависал интерфейс
class AudioLoader(QThread):
    ready = pyqtSignal(object, int, int)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        y, sr = librosa.load(self.path, sr=None)
        self.ready.emit(y, sr, int(len(y) / sr * 10))


# Запускает анализ в фоне чтобы не зависал интерфейс
class AnalysisWorker(QThread):
    done = pyqtSignal(object, object, dict)

    def __init__(self, audio_path, model, threshold):
        super().__init__()
        self.audio_path = audio_path
        self.model = model
        self.threshold = threshold

    def run(self):
        result = is_anomaly(self.audio_path, self.model, self.threshold)

        fig = Figure(facecolor="#111827")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_facecolor("#111827")
        librosa.display.specshow(result["spectrogram"], sr=result["sr"], ax=ax)
        ax.axis("off")
        canvas.draw()
        img_orig = np.asarray(canvas.buffer_rgba())[:, :, :3]
        img_rec = np.array(result["img_rec"])

        self.done.emit(img_orig, img_rec, result)


# Запрашивает ответ у Claude AI в фоне
class AIWorker(QThread):
    token = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, is_anomaly_flag, error_val, threshold, filename):
        super().__init__()
        self.is_anomaly_flag = is_anomaly_flag
        self.error_val = error_val
        self.threshold = threshold
        self.filename = filename

    def run(self):
        try:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                with open("api_key.txt", "r", encoding="utf-8") as f:
                    api_key = f.read().strip()

            client = anthropic.Anthropic(
                api_key=api_key,
                base_url="https://api.scarlex.ru",
                http_client=httpx.Client(proxy=None, trust_env=False),
            )

            status = "АНОМАЛИЯ" if self.is_anomaly_flag else "НОРМА"
            prompt = (
                f"Ты эксперт по диагностике двигателей.\n"
                f"Файл: {self.filename}\n"
                f"Результат: {status}\n"
                f"Ошибка реконструкции: {self.error_val:.6f}\n"
                f"Порог: {self.threshold:.6f}\n\n"
                f"Дай краткое профессиональное заключение (2-3 предложения)."
            )

            msg = None
            while not msg or not msg.content:
                msg = client.messages.create(
                    model="claude-opus-4.7",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )

            for ch in msg.content[0].text:
                self.token.emit(ch)
                time.sleep(0.012)

        except FileNotFoundError:
            self.error.emit("Создайте файл api_key.txt с вашим ключом")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)

        # модель и путь к файлу
        self.model = None
        self.threshold = None
        self.audio_path = None

        # данные аудио для воспроизведения
        self.audio_data = None
        self.audio_sr = None
        self.audio_pos = 0
        self.audio_stream = None
        self.is_playing = False
        self.play_offset = 0.0
        self.play_finished = False

        # анимация "думает..."
        self.thinking_pos = None
        self.thinking_dots = 0
        self.thinking_timer = QTimer()
        self.thinking_timer.setInterval(400)
        self.thinking_timer.timeout.connect(self.animate_thinking)

        # таймер обновления полосы прогресса
        self.play_timer = QTimer()
        self.play_timer.setInterval(100)
        self.play_timer.timeout.connect(self.update_timeline)

        # график оригинальной спектрограммы
        self.fig_orig, self.ax_orig = plt.subplots(facecolor="#111827")
        self.fig_orig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax_orig.set_facecolor("#111827")
        self.canvas_orig = FigureCanvasQTAgg(self.fig_orig)
        self.canvas_orig.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout1 = QtWidgets.QVBoxLayout(self.originalSpectrogramWidget)
        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.addWidget(self.canvas_orig)

        # график реконструированной спектрограммы
        self.fig_rec, self.ax_rec = plt.subplots(facecolor="#111827")
        self.fig_rec.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax_rec.set_facecolor("#111827")
        self.canvas_rec = FigureCanvasQTAgg(self.fig_rec)
        self.canvas_rec.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout2 = QtWidgets.QVBoxLayout(self.reconstructedSpectrogramWidget)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.addWidget(self.canvas_rec)

        # кнопки
        self.outputTextEdit.setReadOnly(True)
        self.loadAudioButton.clicked.connect(self.load_file)
        self.playPauseButton.clicked.connect(self.toggle_play_pause)
        self.timeSlider.sliderReleased.connect(self.seek)
        self.runButton.clicked.connect(self.run_analysis)

        # всплывающая подсказка на кнопке ?
        self.info_popup = InfoPopup(INFO_IMAGE)
        self.infoButton.enterEvent = self.show_info
        self.infoButton.leaveEvent = self.hide_info

    def show_info(self, _event):
        pos = QCursor.pos()
        self.info_popup.move(pos.x() + 16, pos.y() + 16)
        self.info_popup.show()
        self.info_popup.raise_()

    def hide_info(self, _event):
        self.info_popup.hide()

    # ---- чат ----

    def chat(self, html):
        self.outputTextEdit.append(html)
        self.outputTextEdit.ensureCursorVisible()

    def chat_status(self, text):
        self.chat(f'<p style="color:#4b5563; font-size:9pt; margin:2px 0 4px 2px;">{text}</p>')

    def chat_result(self, anomaly, error, threshold):
        color = "#ef4444" if anomaly else "#22c55e"
        icon = "❌" if anomaly else "✅"
        label = "АНОМАЛИЯ" if anomaly else "НОРМА"
        self.chat(f"""
        <table cellpadding="0" cellspacing="0" width="100%" style="margin:6px 0 10px 0;">
        <tr>
          <td width="3" bgcolor="{color}">&nbsp;</td>
          <td bgcolor="#161b27" style="padding:12px 16px;">
            <font color="{color}"><b>{icon}&nbsp;&nbsp;{label}</b></font><br><br>
            <font color="#6b7280">Ошибка: &nbsp;&nbsp;{error:.6f}</font><br>
            <font color="#6b7280">Порог: &nbsp;&nbsp;&nbsp;&nbsp;{threshold:.6f}</font>
          </td>
        </tr>
        </table>
        """)

    def chat_ai_start(self):
        self.thinking_dots = 0
        self.outputTextEdit.append('<b style="color:#60a5fa; font-size:10pt;">ENGINE AI</b>&nbsp;&nbsp;.')
        cursor = self.outputTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        self.thinking_pos = cursor.position() - 1
        self.thinking_timer.start()

    def animate_thinking(self):
        self.thinking_dots = (self.thinking_dots + 1) % 4
        dots = "." * (self.thinking_dots + 1)
        cursor = self.outputTextEdit.textCursor()
        cursor.setPosition(self.thinking_pos)
        cursor.movePosition(cursor.End, cursor.KeepAnchor)
        cursor.insertText(dots)
        self.outputTextEdit.setTextCursor(cursor)

    def chat_ai_token(self, text):
        if self.thinking_timer.isActive():
            self.thinking_timer.stop()
            cursor = self.outputTextEdit.textCursor()
            cursor.setPosition(self.thinking_pos)
            cursor.movePosition(cursor.End, cursor.KeepAnchor)
            cursor.removeSelectedText()
        cursor = self.outputTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.outputTextEdit.setTextCursor(cursor)
        self.outputTextEdit.ensureCursorVisible()

    def chat_ai_done(self):
        self.thinking_timer.stop()
        r = self.last_result
        self.chat_result(r["is_anomaly"], r["error"], r["threshold"])

    def chat_ai_error(self, msg):
        self.thinking_timer.stop()
        self.chat_status(f"⚠ AI: {msg}")

    # ---- воспроизведение ----

    def audio_callback(self, outdata, frames, _time, _status):
        remaining = len(self.audio_data) - self.audio_pos
        if remaining <= 0:
            outdata.fill(0)
            raise sd.CallbackStop()
        chunk = min(frames, remaining)
        outdata[:chunk, 0] = self.audio_data[self.audio_pos:self.audio_pos + chunk]
        if chunk < frames:
            outdata[chunk:].fill(0)
        self.audio_pos += chunk

    def on_audio_finished(self):
        self.play_finished = True

    def start_stream(self):
        self.play_finished = False
        self.audio_stream = sd.OutputStream(
            samplerate=self.audio_sr, channels=1, dtype="float32", blocksize=2048,
            callback=self.audio_callback, finished_callback=self.on_audio_finished,
        )
        self.audio_stream.start()

    def stop_stream(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

    def toggle_play_pause(self):
        if not self.audio_path:
            self.chat_status("Сначала загрузите WAV файл")
            return
        if self.audio_data is None:
            self.chat_status("Аудио ещё загружается...")
            return

        if self.is_playing:
            self.play_offset = self.audio_pos / self.audio_sr
            self.stop_stream()
            self.is_playing = False
            self.playPauseButton.setText("▶")
            self.play_timer.stop()
        else:
            if self.play_offset >= len(self.audio_data) / self.audio_sr:
                self.play_offset = 0.0
                self.timeSlider.setValue(0)
            self.audio_pos = int(self.play_offset * self.audio_sr)
            self.start_stream()
            self.is_playing = True
            self.playPauseButton.setText("⏸")
            self.play_timer.start()

    def update_timeline(self):
        if self.play_finished:
            self.is_playing = False
            self.playPauseButton.setText("▶")
            self.play_timer.stop()
            self.timeSlider.setValue(0)
            self.play_offset = 0.0
            return
        if self.audio_sr:
            self.timeSlider.setValue(int(self.audio_pos / self.audio_sr * 10))

    def seek(self):
        self.play_offset = self.timeSlider.value() / 10.0
        if self.is_playing:
            self.stop_stream()
            self.audio_pos = int(self.play_offset * self.audio_sr)
            self.start_stream()

    # ---- загрузка файла ----

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать WAV", "", "WAV files (*.wav)")
        if not path:
            return
        self.stop_stream()
        self.is_playing = False
        self.playPauseButton.setText("▶")
        self.play_timer.stop()
        self.audio_path = path
        self.play_offset = 0.0
        self.audio_data = None
        self.timeSlider.setValue(0)
        self.outputTextEdit.clear()
        self.chat_status(f"{os.path.basename(path)}")
        self.loader = AudioLoader(path)
        self.loader.ready.connect(self.on_audio_loaded)
        self.loader.start()

    def on_audio_loaded(self, y, sr, slider_max):
        duration = len(y) / sr
        if duration > 10:
            self.chat_status(f"Ошибка: файл слишком длинный ({duration:.1f} сек). Максимум — 10 секунд.")
            self.runButton.setEnabled(False)
            self.audio_data = None
            return
        if duration < 5:
            self.chat_status(f"Ошибка: файл слишком короткий ({duration:.1f} сек). Минимум — 5 секунд.")
            self.runButton.setEnabled(False)
            self.audio_data = None
            return
        self.audio_data = y.astype(np.float32)
        self.audio_sr = sr
        self.timeSlider.setMaximum(slider_max)
        self.runButton.setEnabled(True)

    # ---- анализ ----

    def run_analysis(self):
        if not self.audio_path:
            self.chat_status("Сначала загрузите WAV файл")
            return
        if hasattr(self, "worker") and self.worker.isRunning():
            return

        self.runButton.setEnabled(False)
        self.progressBar.setValue(20)

        if self.model is None:
            try:
                from autoencoder import MODEL_FILE, THRESHOLD_FILE
                self.model = joblib.load(MODEL_FILE)
                self.threshold = float(np.load(THRESHOLD_FILE))
            except FileNotFoundError:
                self.chat_status("Обучение модели...")
                self.model, self.threshold = train()

        self.progressBar.setValue(50)
        self.worker = AnalysisWorker(self.audio_path, self.model, self.threshold)
        self.worker.done.connect(self.on_analysis_done)
        self.worker.start()

    def on_analysis_done(self, img_orig, img_rec, result):
        self.ax_orig.clear()
        self.ax_orig.set_facecolor("#111827")
        self.ax_orig.imshow(img_orig, aspect="auto")
        self.ax_orig.axis("off")
        self.canvas_orig.draw()

        self.ax_rec.clear()
        self.ax_rec.set_facecolor("#111827")
        self.ax_rec.imshow(img_rec, aspect="auto")
        self.ax_rec.axis("off")
        self.fig_rec.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas_rec.draw()

        self.progressBar.setValue(100)
        self.progressBar.setValue(0)
        self.runButton.setEnabled(True)

        self.last_result = result
        self.chat_ai_start()
        self.ai_worker = AIWorker(
            result["is_anomaly"], result["error"], result["threshold"],
            os.path.basename(self.audio_path)
        )
        self.ai_worker.token.connect(self.chat_ai_token)
        self.ai_worker.finished.connect(self.chat_ai_done)
        self.ai_worker.error.connect(self.chat_ai_error)
        self.ai_worker.start()


app = QtWidgets.QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec_())
