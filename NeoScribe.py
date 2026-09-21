import os
import sys

# HuggingFace uyarılarını sustur (zararsızdır).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import time
import math
import wave
import queue
import ctypes
import inspect
import pkgutil
import importlib
import threading
import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from ctypes import wintypes


if sys.platform != "win32":
    raise SystemExit("Bu program yalnızca Windows üzerinde çalışır.")


# =========================================================
# NVIDIA CUDA / cuDNN DLL yolları (faster_whisper'dan ÖNCE)
# =========================================================

def configure_cuda_dll_paths():
    """
    pip ile kurulan nvidia-cublas-cu12 ve nvidia-cudnn-cu12
    paketlerinin DLL klasörlerini Windows'a tanıtır.
    """
    import site

    roots = []

    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass

    roots.extend(
        path for path in sys.path
        if path.lower().endswith("site-packages")
    )

    added = []

    for root in roots:
        nvidia_dir = os.path.join(root, "nvidia")

        if not os.path.isdir(nvidia_dir):
            continue

        for package in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, package, "bin")

            if not os.path.isdir(bin_dir) or bin_dir in added:
                continue

            try:
                os.add_dll_directory(bin_dir)
            except (AttributeError, OSError):
                pass

            os.environ["PATH"] = (
                bin_dir + os.pathsep + os.environ.get("PATH", "")
            )

            added.append(bin_dir)

    return added


CUDA_DLL_PATHS = configure_cuda_dll_paths()

import numpy as np
import psutil
import proctap

from faster_whisper import WhisperModel


# =========================================================
# Ayarlar
# =========================================================

# Arayüzden seçilebilecek modeller.
MODEL_CHOICES = ["small", "medium", "large-v3-turbo", "large-v3"]
DEFAULT_MODEL = "medium"

# Cümleyi ortadan kesmemek için parça uzunluğu sabit değil:
# 8 sn dolunca en sessiz noktada (konuşma arasında) kesilir.
# Uygun sessizlik bulunamazsa 15 sn'de zorla kesilir.
TARGET_SEGMENT_SECONDS = 8.0
MAX_SEGMENT_SECONDS = 15.0
MIN_SEGMENT_SECONDS = 4.0

TARGET_SAMPLE_RATE = 16000

# 0-1 aralığında (float). ~0.0003 = int16 ölçeğinde yaklaşık 10.
MIN_PEAK_LEVEL = 0.0003

# Ses normalizasyonu: düşük sesi yükseltir (float32 ile, kayıpsız).
NORMALIZE_TARGET = 0.5
MAX_GAIN = 300.0

LANGUAGES = {
    "Türkçe": (
        "tr",
        "Bu konuşma Türkçe bir bilgisayar bilimi, yapay zekâ ve Python "
        "dersidir. Konular arasında makine öğrenmesi, derin öğrenme, "
        "bilgisayarlı görü, doğal dil işleme, transformer mimarisi, "
        "attention, embedding, token, encoder, decoder, model, veri seti, "
        "eğitim, tahmin, sınıflandırma, nesne tespiti ve algoritmalar "
        "bulunur."
    ),
    "English": (
        "en",
        "This is a technical talk about computer science, artificial "
        "intelligence and Python: machine learning, deep learning, "
        "computer vision, natural language processing, transformer "
        "architecture, attention, embeddings, tokens, encoder, decoder."
    ),
}

AUDIO_CLASS_NAMES = (
    "ProcessAudioCapture",
    "ProcTap",
)

user32 = ctypes.windll.user32


# =========================================================
# ProcTap sınıfını bul
# =========================================================

def find_audio_class():
    for class_name in AUDIO_CLASS_NAMES:
        item = getattr(proctap, class_name, None)

        if inspect.isclass(item):
            return item

        if inspect.ismodule(item):
            for inner_name in AUDIO_CLASS_NAMES:
                inner_item = getattr(item, inner_name, None)

                if inspect.isclass(inner_item):
                    return inner_item

    package_path = getattr(proctap, "__path__", [])

    for module_info in pkgutil.walk_packages(package_path, "proctap."):
        module_name = module_info.name

        if any(
            part in module_name
            for part in ("__main__", "cli", "contrib")
        ):
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for class_name in AUDIO_CLASS_NAMES:
            item = getattr(module, class_name, None)

            if inspect.isclass(item):
                return item

    return None


AudioCaptureClass = find_audio_class()

if AudioCaptureClass is None:
    raise SystemExit(
        "proctap içinde ses yakalama sınıfı bulunamadı.\n"
        "ProcessAudioCapture veya ProcTap mevcut değil."
    )


def create_audio_capture(pid, callback):
    attempts = [
        lambda: AudioCaptureClass(pid=pid, on_data=callback),
        lambda: AudioCaptureClass(process_id=pid, on_data=callback),
        lambda: AudioCaptureClass(pid, on_data=callback),
    ]

    errors = []

    for attempt in attempts:
        try:
            return attempt()
        except Exception as error:
            errors.append(repr(error))

    raise RuntimeError(
        "Ses yakalama sınıfı oluşturulamadı:\n" + "\n".join(errors)
    )


# =========================================================
# Windows süreçleri
# =========================================================

def visible_window_pids():
    pids = set()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, lparam):
        if (
            user32.IsWindowVisible(hwnd)
            and user32.GetWindowTextLengthW(hwnd) > 0
        ):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pids.add(pid.value)

        return True

    user32.EnumWindows(callback, 0)

    return pids


def list_visible_processes():
    window_pids = visible_window_pids()
    result = []

    for process in psutil.process_iter(["pid", "name"]):
        try:
            pid = process.info.get("pid")
            name = process.info.get("name")

            if pid and name and pid in window_pids:
                result.append((name, pid))

        except psutil.Error:
            continue

    result.sort(key=lambda item: (item[0].lower(), item[1]))

    return result


# =========================================================
# Ses işleme (tamamı float32 - düşük seviyeli sesi bozmaz)
# =========================================================

def to_mono_float32(raw_data, channels, is_float):
    """Ham PCM baytlarını -1..1 aralığında mono float32'ye çevirir."""

    channels = max(1, int(channels))

    if is_float:
        data = np.frombuffer(raw_data, dtype=np.float32).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        data = data / 32768.0

    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)

    return np.clip(data, -1.0, 1.0).astype(np.float32)


def resample_audio(data, source_rate, target_rate):
    """Alçak geçiren filtre + yeniden örnekleme (aliasing olmadan)."""

    if len(data) == 0 or source_rate == target_rate:
        return data

    ratio = source_rate / target_rate

    if ratio > 1.0:
        taps = 127
        cutoff = 0.45 * target_rate / source_rate
        n = np.arange(taps) - (taps - 1) / 2.0
        kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        kernel /= kernel.sum()
        data = np.convolve(data, kernel, mode="same")

    new_length = int(len(data) / ratio)

    if new_length <= 0:
        return np.empty(0, dtype=np.float32)

    positions = np.arange(new_length) * ratio

    return np.interp(
        positions,
        np.arange(len(data)),
        data
    ).astype(np.float32)


def normalize_audio(data):
    """DC kaymasını siler ve sesi güvenli biçimde yükseltir."""

    data = data - np.mean(data)

    reference = float(np.percentile(np.abs(data), 99.5))

    if reference < 1e-6:
        return data.astype(np.float32), 1.0

    gain = NORMALIZE_TARGET / reference
    gain = max(1.0, min(gain, MAX_GAIN))

    return np.clip(data * gain, -1.0, 1.0).astype(np.float32), gain


def to_db(value):
    return 20.0 * math.log10(max(float(value), 1e-9))


def find_cut_point(mono, rate, min_seconds, force):
    """
    Konuşma arasındaki en sessiz noktayı bulur ve örnek indeksini döndürür.
    Uygun sessizlik yoksa (force=False iken) None döner.
    """

    frame = max(1, int(rate * 0.02))
    count = len(mono) // frame
    window = 15  # 15 x 20 ms = 300 ms

    if count < window + 10:
        return None

    frames = mono[: count * frame].reshape(count, frame)
    energy = np.sqrt(np.mean(frames * frames, axis=1))

    smooth = np.convolve(
        energy,
        np.ones(window) / window,
        mode="valid"
    )

    start = int(min_seconds / 0.02)

    if start >= len(smooth):
        start = 0

    region = smooth[start:]
    best = int(np.argmin(region))
    best_level = float(region[best])

    if not force and best_level > 0.35 * float(np.mean(energy)):
        return None

    return (start + best + window // 2) * frame


# =========================================================
# Ana arayüz
# =========================================================

class LiveTranscriptionApp:
    def __init__(self, root):
        self.root = root

        self.root.title("Uygulama Sesi - Çevrim Dışı Canlı Metin (Whisper)")
        self.root.geometry("1000x720")
        self.root.minsize(800, 560)

        self.apps = []
        self.audio_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.transcript_lines = []
        self.running = False
        self.model = None
        self.last_text = ""
        self.session = {}

        self.capture_format = {
            "sample_rate": 48000,
            "channels": 2,
            "is_float": True,
        }

        self.create_style()
        self.create_interface()

        self.refresh_processes()
        self.process_ui_queue()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # -----------------------------------------------------
    # Arayüz
    # -----------------------------------------------------

    def create_style(self):
        style = ttk.Style()

        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))
        style.configure("Start.TButton", font=("Segoe UI", 10, "bold"))

    def create_interface(self):
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="Uygulama Sesi → Çevrim Dışı Canlı Konuşma Metni",
            style="Title.TLabel"
        ).pack(anchor="w", pady=(0, 12))

        settings = ttk.LabelFrame(main, text="Ayarlar", padding=10)
        settings.pack(fill="x", pady=(0, 10))
        settings.columnconfigure(1, weight=1)

        # Uygulama / PID
        ttk.Label(settings, text="Uygulama / PID:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=5
        )

        self.process_combo = ttk.Combobox(settings, state="readonly")
        self.process_combo.grid(row=0, column=1, sticky="ew", pady=5)

        self.refresh_button = ttk.Button(
            settings, text="Yenile", command=self.refresh_processes
        )
        self.refresh_button.grid(row=0, column=2, padx=(8, 0), pady=5)

        # Dil / Model / Cihaz
        options = ttk.Frame(settings)
        options.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)

        ttk.Label(options, text="Dil:").pack(side="left")

        self.language_combo = ttk.Combobox(
            options,
            state="readonly",
            values=list(LANGUAGES.keys()),
            width=10
        )
        self.language_combo.set("Türkçe")
        self.language_combo.pack(side="left", padx=(6, 18))

        ttk.Label(options, text="Model:").pack(side="left")

        self.model_combo = ttk.Combobox(
            options,
            state="readonly",
            values=MODEL_CHOICES,
            width=16
        )
        self.model_combo.set(DEFAULT_MODEL)
        self.model_combo.pack(side="left", padx=(6, 18))

        ttk.Label(options, text="Cihaz:").pack(side="left")

        self.device_combo = ttk.Combobox(
            options,
            state="readonly",
            values=["GPU (CUDA)", "CPU"],
            width=12
        )
        self.device_combo.set("GPU (CUDA)")
        self.device_combo.pack(side="left", padx=(6, 0))

        # Kayıt türü
        ttk.Label(settings, text="Kaydet:").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=5
        )

        save_frame = ttk.Frame(settings)
        save_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=5)

        self.save_text_var = tk.BooleanVar(value=True)
        self.save_audio_var = tk.BooleanVar(value=False)
        self.normalize_var = tk.BooleanVar(value=True)

        self.save_text_check = ttk.Checkbutton(
            save_frame,
            text="Metin (TXT)",
            variable=self.save_text_var
        )
        self.save_text_check.pack(side="left", padx=(0, 18))

        self.save_audio_check = ttk.Checkbutton(
            save_frame,
            text="Ses (WAV)",
            variable=self.save_audio_var
        )
        self.save_audio_check.pack(side="left", padx=(0, 18))

        self.normalize_check = ttk.Checkbutton(
            save_frame,
            text="Ses kaydını yükselt (normalize)",
            variable=self.normalize_var
        )
        self.normalize_check.pack(side="left")

        # TXT dosyası
        ttk.Label(settings, text="TXT dosyası:").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=5
        )

        self.file_path = tk.StringVar(
            value=os.path.abspath("konusma_metni.txt")
        )

        self.file_entry = ttk.Entry(settings, textvariable=self.file_path)
        self.file_entry.grid(row=3, column=1, sticky="ew", pady=5)

        self.file_button = ttk.Button(
            settings, text="Dosya Seç", command=self.select_file
        )
        self.file_button.grid(row=3, column=2, padx=(8, 0), pady=5)

        # WAV dosyası
        ttk.Label(settings, text="WAV dosyası:").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=5
        )

        self.audio_path = tk.StringVar(
            value=os.path.abspath("kayit_sesi.wav")
        )

        self.audio_entry = ttk.Entry(settings, textvariable=self.audio_path)
        self.audio_entry.grid(row=4, column=1, sticky="ew", pady=5)

        self.audio_button = ttk.Button(
            settings, text="Dosya Seç", command=self.select_audio_file
        )
        self.audio_button.grid(row=4, column=2, padx=(8, 0), pady=5)

        # Butonlar
        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=(0, 10))

        self.start_button = ttk.Button(
            buttons, text="Başlat", style="Start.TButton", command=self.start
        )
        self.start_button.pack(side="left")

        self.stop_button = ttk.Button(
            buttons, text="Durdur", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))

        self.save_button = ttk.Button(
            buttons, text="TXT Olarak Kaydet", command=self.save_txt
        )
        self.save_button.pack(side="left", padx=(8, 0))

        self.clear_button = ttk.Button(
            buttons, text="Temizle", command=self.clear_text
        )
        self.clear_button.pack(side="left", padx=(8, 0))

        self.status = tk.StringVar(value="Hazır")

        ttk.Label(
            buttons, textvariable=self.status, style="Status.TLabel"
        ).pack(side="right")

        # Metin kutusu
        frame = ttk.LabelFrame(main, text="Canlı Konuşma Metni", padding=8)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.text_box = tk.Text(
            frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 11),
            padx=10,
            pady=10
        )
        self.text_box.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            frame, orient="vertical", command=self.text_box.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.text_box.configure(yscrollcommand=scrollbar.set)

        self.text_box.tag_configure(
            "system", foreground="#6b6b6b", font=("Segoe UI", 9)
        )

        ttk.Label(
            main,
            text=(
                "Opera/Teams içinde konuşmalı ses aç. Ses gelmezse aynı "
                "uygulamanın farklı PID'sini dene. En iyi sonuç için "
                "uygulamanın ses seviyesini Windows Ses Karıştırıcısı'nda "
                "yüksek tut."
            )
        ).pack(anchor="w", pady=(8, 0))

    # -----------------------------------------------------
    # UI kuyruğu
    # -----------------------------------------------------

    def send_ui(self, message_type, message=""):
        self.ui_queue.put((message_type, message))

    def append_system_message(self, message):
        self.text_box.configure(state="normal")
        self.text_box.insert("end", f"[Sistem] {message}\n", "system")
        self.text_box.see("end")
        self.text_box.configure(state="disabled")

    def append_transcript(self, message):
        self.transcript_lines.append(message)

        self.text_box.configure(state="normal")
        self.text_box.insert("end", message + "\n")
        self.text_box.see("end")
        self.text_box.configure(state="disabled")

        self.append_text_to_file(message)

    def process_ui_queue(self):
        try:
            while True:
                message_type, message = self.ui_queue.get_nowait()

                if message_type == "system":
                    self.append_system_message(message)

                elif message_type == "transcript":
                    self.append_transcript(message)

                elif message_type == "status":
                    self.status.set(message)

                elif message_type == "finished":
                    self.finish_ui()

        except queue.Empty:
            pass

        self.root.after(200, self.process_ui_queue)

    # -----------------------------------------------------
    # Dosya işlemleri
    # -----------------------------------------------------

    def normalized_file_path(self):
        path = self.file_path.get().strip()

        if not path:
            return ""

        if not path.lower().endswith(".txt"):
            path += ".txt"

        self.file_path.set(path)

        return path

    def append_text_to_file(self, text):
        path = self.normalized_file_path()

        if not path:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            with open(path, "a", encoding="utf-8") as file:
                file.write(text + "\n")

        except OSError as error:
            self.append_system_message(f"TXT dosyasına yazılamadı: {error}")

    def select_file(self):
        path = filedialog.asksaveasfilename(
            title="TXT dosyası seç",
            defaultextension=".txt",
            initialfile="konusma_metni.txt",
            filetypes=[
                ("Metin dosyası", "*.txt"),
                ("Tüm dosyalar", "*.*")
            ]
        )

        if not path:
            return

        if not path.lower().endswith(".txt"):
            path += ".txt"

        self.file_path.set(path)

    def select_audio_file(self):
        path = filedialog.asksaveasfilename(
            title="WAV dosyası seç",
            defaultextension=".wav",
            initialfile="kayit_sesi.wav",
            filetypes=[
                ("WAV ses dosyası", "*.wav"),
                ("Tüm dosyalar", "*.*")
            ]
        )

        if not path:
            return

        if not path.lower().endswith(".wav"):
            path += ".wav"

        self.audio_path.set(path)

    def save_txt(self):
        path = self.normalized_file_path()

        if not path:
            self.select_file()
            path = self.normalized_file_path()

        if not path:
            return

        text = "\n".join(self.transcript_lines).strip()

        if not text:
            messagebox.showwarning("Uyarı", "Kaydedilecek konuşma metni yok.")
            return

        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(text + "\n")

            self.status.set("TXT kaydedildi")
            messagebox.showinfo("Başarılı", f"Metin kaydedildi:\n\n{path}")

        except OSError as error:
            messagebox.showerror("Kaydetme hatası", str(error))

    def clear_text(self):
        if self.running:
            messagebox.showwarning(
                "Uyarı", "Dinleme devam ederken temizleme yapılamaz."
            )
            return

        self.transcript_lines.clear()
        self.last_text = ""

        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.configure(state="disabled")

        path = self.normalized_file_path()

        if path:
            try:
                with open(path, "w", encoding="utf-8"):
                    pass
            except OSError:
                pass

        self.status.set("Temizlendi")

    # -----------------------------------------------------
    # Süreç listesi
    # -----------------------------------------------------

    def refresh_processes(self):
        try:
            self.apps = list_visible_processes()

            values = [f"{name} - PID {pid}" for name, pid in self.apps]
            self.process_combo["values"] = values

            preferred = -1

            for index, (name, pid) in enumerate(self.apps):
                if any(
                    item in name.lower()
                    for item in (
                        "opera", "teams", "chrome",
                        "edge", "firefox", "discord"
                    )
                ):
                    preferred = index
                    break

            if preferred >= 0:
                self.process_combo.current(preferred)
            elif values:
                self.process_combo.current(0)

            self.status.set(f"{len(values)} süreç bulundu")

        except Exception as error:
            messagebox.showerror("Süreç listeleme hatası", str(error))

    # -----------------------------------------------------
    # Başlat / Durdur
    # -----------------------------------------------------

    def set_controls(self, running):
        idle = "disabled" if running else "normal"
        readonly = "disabled" if running else "readonly"

        self.start_button.configure(state=idle)
        self.stop_button.configure(state="normal" if running else "disabled")
        self.refresh_button.configure(state=idle)
        self.file_button.configure(state=idle)
        self.clear_button.configure(state=idle)
        self.save_button.configure(state=idle)
        self.language_combo.configure(state=readonly)
        self.model_combo.configure(state=readonly)
        self.device_combo.configure(state=readonly)
        self.save_text_check.configure(state=idle)
        self.save_audio_check.configure(state=idle)
        self.normalize_check.configure(state=idle)
        self.audio_entry.configure(state=idle)
        self.audio_button.configure(state=idle)

    def start(self):
        if self.running:
            return

        selected_index = self.process_combo.current()

        if selected_index < 0:
            messagebox.showwarning("Uyarı", "Önce uygulama/PID seç.")
            return

        want_text = self.save_text_var.get()
        want_audio = self.save_audio_var.get()

        if not want_text and not want_audio:
            messagebox.showwarning(
                "Uyarı", "En az bir kayıt türü seç: Metin veya Ses."
            )
            return

        if want_text and not self.normalized_file_path():
            messagebox.showwarning("Uyarı", "TXT dosyası seç.")
            return

        audio_final = ""

        if want_audio:
            base = self.audio_path.get().strip()

            if not base:
                messagebox.showwarning("Uyarı", "WAV dosyası seç.")
                return

            if not base.lower().endswith(".wav"):
                base += ".wav"

            # Eski kayıtların üzerine yazmamak için tarih/saat eklenir.
            root_name, extension = os.path.splitext(base)
            audio_final = (
                f"{root_name}_{time.strftime('%Y%m%d_%H%M%S')}{extension}"
            )

            try:
                os.makedirs(
                    os.path.dirname(os.path.abspath(audio_final)),
                    exist_ok=True
                )
            except OSError as error:
                messagebox.showerror("Klasör hatası", str(error))
                return

        name, pid = self.apps[selected_index]

        language_code, prompt = LANGUAGES[self.language_combo.get()]

        settings = {
            "model": self.model_combo.get(),
            "device": (
                "cuda" if self.device_combo.get().startswith("GPU")
                else "cpu"
            ),
            "language": language_code,
            "prompt": prompt,
            "transcribe": want_text,
            "audio_final": audio_final,
            "normalize": self.normalize_var.get(),
        }

        self.session = settings

        self.audio_queue = queue.Queue()
        self.stop_event.clear()
        self.running = True

        self.set_controls(True)

        self.status.set(f"Başlatılıyor: {name} - PID {pid}")
        self.append_system_message(f"Dinleme başladı: {name} - PID {pid}")

        if audio_final:
            self.append_system_message(f"Ses kaydı: {audio_final}")

        if not want_text:
            self.append_system_message(
                "Yalnızca ses kaydı modu: metne çevirme yapılmayacak."
            )

        threading.Thread(
            target=self.transcription_worker,
            args=(pid, settings),
            daemon=True
        ).start()

    def stop(self):
        if not self.running:
            return

        self.stop_event.set()

        self.status.set("Durduruluyor...")
        self.stop_button.configure(state="disabled")
        self.append_system_message("Durdurma istendi.")

    def finish_ui(self):
        self.running = False
        self.set_controls(False)
        self.status.set("Hazır")

    # -----------------------------------------------------
    # Model yükleme (GPU -> GPU int8 -> CPU yedek)
    # -----------------------------------------------------

    def load_model(self, settings):
        size = settings["model"]
        device = settings["device"]
        language = settings["language"]

        if device == "cuda":
            attempts = [
                (size, "cuda", "float16"),
                (size, "cuda", "int8_float16"),
                ("small", "cpu", "int8"),
            ]
        else:
            attempts = [(size, "cpu", "int8")]

        for model_size, model_device, compute_type in attempts:
            self.send_ui(
                "status", f"Whisper {model_size} yükleniyor..."
            )
            self.send_ui(
                "system",
                f"Model: {model_size} | Cihaz: {model_device} | "
                f"Hesaplama: {compute_type} (ilk seferde indirilebilir)"
            )

            try:
                model = WhisperModel(
                    model_size,
                    device=model_device,
                    compute_type=compute_type,
                    cpu_threads=4,
                    num_workers=1
                )

                # cuDNN/cuBLAS hataları çoğu zaman yüklemede değil,
                # ilk çalıştırmada çıkar. Bu yüzden ısınma testi yapılır.
                warmup = np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32)

                segments, _ = model.transcribe(
                    warmup,
                    language=language,
                    beam_size=1,
                    vad_filter=False,
                    without_timestamps=True
                )

                list(segments)

                self.model = model

                self.send_ui(
                    "system",
                    f"Whisper hazır ({model_size}, {model_device}, "
                    f"{compute_type})."
                )

                return

            except Exception as error:
                self.send_ui(
                    "system",
                    f"Bu ayar çalışmadı ({model_size}/{model_device}/"
                    f"{compute_type}): {error}"
                )

        raise RuntimeError("Hiçbir model ayarı çalışmadı.")

    # -----------------------------------------------------
    # Ses yakalama (ayrı iş parçacığı)
    # -----------------------------------------------------

    def capture_worker(self, pid):
        tap = None
        raw_file = None
        received = {"bytes": 0}

        def on_data(pcm, frames):
            if not pcm:
                return

            received["bytes"] += len(pcm)

            if received["bytes"] == len(pcm):
                self.send_ui(
                    "system",
                    f"İlk ses verisi geldi: {len(pcm)} byte"
                )

            data = bytes(pcm)

            if raw_file is not None:
                try:
                    raw_file.write(data)
                except Exception:
                    pass

            if self.session.get("transcribe", True):
                self.audio_queue.put(data)

        try:
            if self.session.get("audio_final"):
                raw_file = open(
                    self.session["audio_final"] + ".tmp",
                    "wb",
                    buffering=1024 * 1024
                )

            tap = create_audio_capture(pid, on_data)

            format_info = {}

            if hasattr(tap, "get_format"):
                format_info = tap.get_format()

            self.send_ui("system", f"Ses formatı: {format_info}")

            if isinstance(format_info, dict):
                fmt = self.capture_format

                fmt["sample_rate"] = int(
                    format_info.get(
                        "sample_rate",
                        format_info.get("rate", fmt["sample_rate"])
                    )
                )

                fmt["channels"] = int(
                    format_info.get("channels", fmt["channels"])
                )

                sample_format = str(
                    format_info.get(
                        "sample_format", format_info.get("format", "")
                    )
                ).lower()

                bits = format_info.get(
                    "bits_per_sample", format_info.get("bits")
                )

                if "float" in sample_format:
                    fmt["is_float"] = True
                elif (
                    "int16" in sample_format
                    or "pcm16" in sample_format
                    or sample_format == "s16"
                ):
                    fmt["is_float"] = False
                elif bits == 16:
                    fmt["is_float"] = False
                elif bits == 32:
                    fmt["is_float"] = True

            tap.start()

            self.send_ui("status", "Dinleniyor...")

            last_received = 0
            last_report = time.time()

            while not self.stop_event.wait(0.5):
                now = time.time()

                if now - last_report >= 5:
                    if received["bytes"] == last_received:
                        self.send_ui(
                            "system",
                            "Yeni ses verisi gelmedi. Farklı PID dene "
                            "veya videoyu oynat."
                        )

                    last_received = received["bytes"]
                    last_report = now

        except Exception as error:
            self.send_ui("system", f"HATA - Ses yakalama: {error}")

        finally:
            if tap is not None:
                for method in ("stop", "close"):
                    try:
                        getattr(tap, method)()
                    except Exception:
                        pass

            if raw_file is not None:
                try:
                    raw_file.close()
                except Exception:
                    pass

            self.audio_queue.put(None)

    # -----------------------------------------------------
    # Metne çevirme
    # -----------------------------------------------------

    def transcription_worker(self, pid, settings):
        if settings["transcribe"]:
            try:
                self.load_model(settings)
            except Exception as error:
                self.send_ui(
                    "system", f"Whisper modeli yüklenemedi: {error}"
                )
                self.send_ui("finished")
                return

        if self.stop_event.is_set():
            self.send_ui("system", "Kayıt bitti.")
            self.send_ui("finished")
            return

        capture_thread = threading.Thread(
            target=self.capture_worker,
            args=(pid,),
            daemon=True
        )
        capture_thread.start()

        # Yalnızca ses kaydı: Whisper kullanılmaz.
        if not settings["transcribe"]:
            self.send_ui("status", "Ses kaydediliyor...")

            capture_thread.join()

            self.finalize_audio(settings)

            self.send_ui("system", "Kayıt bitti.")
            self.send_ui("finished")
            return

        parts = []
        total = 0
        leftover = b""
        next_check_at = TARGET_SEGMENT_SECONDS
        finished = False

        while not finished:
            try:
                item = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            fmt = self.capture_format
            rate = fmt["sample_rate"]

            if item is None:
                finished = True
            else:
                frame_bytes = fmt["channels"] * (4 if fmt["is_float"] else 2)

                data = leftover + item
                usable = len(data) - (len(data) % frame_bytes)
                leftover = data[usable:]

                mono = to_mono_float32(
                    data[:usable], fmt["channels"], fmt["is_float"]
                )

                parts.append(mono)
                total += len(mono)

                duration = total / rate

                if duration >= next_check_at:
                    pending = np.concatenate(parts)

                    cut = find_cut_point(
                        pending,
                        rate,
                        MIN_SEGMENT_SECONDS,
                        force=duration >= MAX_SEGMENT_SECONDS
                    )

                    if cut is None:
                        parts = [pending]
                        next_check_at = duration + 0.5
                        continue

                    segment = pending[:cut]
                    rest = pending[cut:]

                    parts = [rest]
                    total = len(rest)
                    next_check_at = TARGET_SEGMENT_SECONDS

                    backlog = self.audio_queue.qsize() * 0.01

                    if backlog > 15:
                        self.send_ui(
                            "system",
                            f"Uyarı: yaklaşık {backlog:.0f} sn geride "
                            "kalındı. Daha küçük model seç."
                        )

                    self.process_segment(segment, rate, settings)
                    continue

            if finished and total / rate >= 0.5:
                self.process_segment(
                    np.concatenate(parts), rate, settings
                )

        if settings.get("audio_final"):
            self.finalize_audio(settings)

        self.send_ui("system", "Kayıt bitti.")
        self.send_ui("finished")

    def finalize_audio(self, settings):
        """
        Kayıt sırasında yazılan ham sesi (.tmp) 16-bit WAV dosyasına çevirir.
        Normalize açıksa tüm kaydın tepe noktasına göre sesi yükseltir.
        Uzun kayıtlarda bellek şişmesin diye parça parça çalışır.
        """

        raw_path = settings["audio_final"] + ".tmp"
        final_path = settings["audio_final"]

        if not os.path.exists(raw_path):
            return

        fmt = dict(self.capture_format)
        rate = int(fmt["sample_rate"])
        channels = max(1, int(fmt["channels"]))
        is_float = bool(fmt["is_float"])

        frame_bytes = channels * (4 if is_float else 2)
        chunk_bytes = rate * 10 * frame_bytes

        total_frames = os.path.getsize(raw_path) // frame_bytes

        if total_frames == 0:
            self.send_ui("system", "Ses kaydı boş, WAV oluşturulmadı.")

            try:
                os.remove(raw_path)
            except OSError:
                pass

            return

        self.send_ui("status", "Ses dosyası kaydediliyor...")

        def read_chunks():
            with open(raw_path, "rb") as file:
                while True:
                    buffer = file.read(chunk_bytes)

                    if not buffer:
                        break

                    usable = len(buffer) - (len(buffer) % frame_bytes)

                    if usable == 0:
                        break

                    if is_float:
                        data = np.frombuffer(
                            buffer[:usable], dtype=np.float32
                        ).astype(np.float32)
                        data = np.nan_to_num(
                            data, nan=0.0, posinf=1.0, neginf=-1.0
                        )
                    else:
                        data = np.frombuffer(
                            buffer[:usable], dtype=np.int16
                        ).astype(np.float32) / 32768.0

                    yield data

        try:
            gain = 1.0
            peak = 0.0

            if settings.get("normalize"):
                for data in read_chunks():
                    peak = max(peak, float(np.max(np.abs(data))))

                if peak > 1e-6:
                    gain = max(1.0, min(0.9 / peak, MAX_GAIN))

            with wave.open(final_path, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(rate)

                for data in read_chunks():
                    data = np.clip(data * gain, -1.0, 1.0)

                    wav_file.writeframes(
                        np.rint(data * 32767.0).astype("<i2").tobytes()
                    )

            os.remove(raw_path)

            self.send_ui(
                "system",
                f"Ses kaydedildi: {final_path} "
                f"({total_frames / rate:.0f} sn, "
                f"kazanç {to_db(gain):+.0f} dB)"
            )

        except Exception as error:
            self.send_ui(
                "system",
                f"HATA - Ses dosyası oluşturulamadı: {error}. "
                f"Ham ses şurada duruyor: {raw_path}"
            )

    def process_segment(self, mono, rate, settings):
        if len(mono) == 0:
            return

        raw_peak = float(np.max(np.abs(mono)))

        if raw_peak < MIN_PEAK_LEVEL:
            self.send_ui(
                "system",
                f"Sessiz parça atlandı (tepe {to_db(raw_peak):.0f} dBFS)."
            )
            return

        audio, gain = normalize_audio(mono)
        audio = resample_audio(audio, rate, TARGET_SAMPLE_RATE)

        if len(audio) < TARGET_SAMPLE_RATE // 2:
            return

        seconds = len(mono) / rate

        self.send_ui(
            "system",
            f"Parça {seconds:.1f} sn | ham tepe {to_db(raw_peak):.0f} dBFS "
            f"| kazanç x{gain:.0f}"
        )

        self.send_ui("status", "Konuşma analiz ediliyor...")

        prompt = settings["prompt"]

        if self.last_text:
            prompt = prompt + " " + self.last_text[-200:]

        try:
            segments, _ = self.model.transcribe(
                audio,
                language=settings["language"],
                task="transcribe",
                beam_size=5,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=prompt,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.35,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 300
                }
            )

            text_parts = []

            for segment in segments:
                text = segment.text.strip()

                if not text:
                    continue

                # Konuşma olmayan yerlerdeki uydurma metinleri ele.
                if (
                    segment.no_speech_prob > 0.7
                    and segment.avg_logprob < -1.0
                ):
                    continue

                text_parts.append(text)

            text = " ".join(text_parts).strip()

        except Exception as error:
            self.send_ui("system", f"HATA - Whisper: {error}")
            return

        if not text:
            self.send_ui("system", "Bu parçada anlaşılır konuşma yok.")
            self.send_ui("status", "Dinleniyor...")
            return

        if text == self.last_text:
            return

        self.last_text = text

        self.send_ui(
            "transcript",
            f"[{time.strftime('%H:%M:%S')}] {text}"
        )

        self.send_ui("status", "Dinleniyor...")

    # -----------------------------------------------------
    # Kapatma
    # -----------------------------------------------------

    def close(self):
        if self.running:
            if self.session.get("audio_final"):
                messagebox.showwarning(
                    "Uyarı",
                    "Ses kaydı sürüyor. Önce Durdur'a bas ve WAV dosyasının "
                    "kaydedilmesini bekle, sonra kapat."
                )
                return

            if not messagebox.askyesno(
                "Programı kapat",
                "Dinleme devam ediyor. Program kapatılsın mı?"
            ):
                return

            self.stop_event.set()

        self.root.destroy()


# =========================================================
# Başlat
# =========================================================

if __name__ == "__main__":
    root = tk.Tk()

    application = LiveTranscriptionApp(root)

    root.mainloop()