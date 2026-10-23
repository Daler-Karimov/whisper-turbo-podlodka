import gc
import os
import subprocess
import sys
import time
import shutil
import tempfile
import traceback
import ctypes
import threading
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GRADIO_CACHE = Path(tempfile.gettempdir()) / "gradio"
FLAGGED_DIR = Path(__file__).resolve().parent / "flagged"

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import gradio as gr
import torch

# ------------------- Локальные папки моделей -------------------
LOCAL_MODEL_OPENAI = MODELS_DIR / "openai--whisper-large-v3-turbo"
LOCAL_MODEL_PODLODKA = MODELS_DIR / "bond005--whisper-podlodka-turbo"

def git_clone_if_needed(repo_url: str, local_dir: Path) -> None:
    """Клонирует репозиторий, если папка отсутствует или пуста."""
    local_dir = Path(local_dir)
    if local_dir.exists() and any(local_dir.iterdir()):
        return
    print(f"[GIT] Клонирование {repo_url} -> {local_dir}")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(local_dir)],
            check=True,
            capture_output=False,
            text=True,
            timeout=1800,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Не удалось клонировать репозиторий {repo_url}: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git не найден. Установите git и добавьте его в PATH.")
    print("[GIT] Клонирование завершено.")

# Клонируем обе модели при старте (если ещё нет)
try:
    git_clone_if_needed("https://huggingface.co/openai/whisper-large-v3-turbo", LOCAL_MODEL_OPENAI)
    git_clone_if_needed("https://huggingface.co/bond005/whisper-podlodka-turbo", LOCAL_MODEL_PODLODKA)
except Exception as e:
    print(f"[!] Критическая ошибка при подготовке моделей: {e}")
    sys.exit(1)

HF_MODEL_OPENAI = str(LOCAL_MODEL_OPENAI)   # локальный путь
HF_MODEL_PODLODKA = str(LOCAL_MODEL_PODLODKA)

MODEL_LABELS = [
    "openai whisper-large-v3-turbo (Transformers)",
    "bond005 whisper-podlodka-turbo (Transformers)",
    "Podlodka-Turbo CTranslate2 (faster-whisper)",
]
PARALLEL_LABEL = "ВСЕ МОДЕЛИ ПАРАЛЛЕЛЬНО (сравнение)"
ALL_MODEL_CHOICES = MODEL_LABELS + [PARALLEL_LABEL]

LANGUAGE_OPTIONS = [
    "auto", "russian", "english", "french", "german", "spanish",
    "italian", "portuguese", "dutch", "polish", "ukrainian",
    "chinese", "japanese", "korean", "arabic", "hindi",
    "turkish", "czech", "swedish", "finnish",
]

LANG_CODE: Dict[str, str] = {
    "russian": "ru", "english": "en", "french": "fr",
    "german": "de", "spanish": "es", "italian": "it",
    "portuguese": "pt", "dutch": "nl", "polish": "pl",
    "ukrainian": "uk", "chinese": "zh", "japanese": "ja",
    "korean": "ko", "arabic": "ar", "hindi": "hi",
    "turkish": "tr", "czech": "cs", "swedish": "sv",
    "finnish": "fi",
}

_cache: Dict[str, Any] = {}
_cublas_check_result: Optional[bool] = None
_model_load_lock = threading.Lock()

_forced_device: str = "GPU" if torch.cuda.is_available() else "CPU"
DEVICE_CHOICES = ["CPU", "GPU"]


def _device() -> str:
    if _forced_device == "GPU" and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _dtype():
    dev = _device()
    return torch.float16 if dev.startswith("cuda") else torch.float32


def _is_using_gpu() -> bool:
    return _device().startswith("cuda")


def _get_vram_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    return total / 1_073_741_824


def _get_gpu_memory_info() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}
    total = _get_vram_total_gb()
    allocated = torch.cuda.memory_allocated(0) / 1_073_741_824
    reserved = torch.cuda.memory_reserved(0) / 1_073_741_824
    free = total - reserved
    return {"total": total, "allocated": allocated, "reserved": reserved, "free": free}


def _get_ram_info() -> Dict[str, float]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1_073_741_824,
            "used": mem.used / 1_073_741_824,
            "available": mem.available / 1_073_741_824,
            "percent": mem.percent,
        }
    except ImportError:
        return {"total": 0, "used": 0, "available": 0, "percent": 0}


def _check_cublas_available() -> bool:
    global _cublas_check_result
    if _cublas_check_result is not None:
        return _cublas_check_result
    if not torch.cuda.is_available():
        _cublas_check_result = False
        return False
    result = _do_check_cublas()
    _cublas_check_result = result
    return result


def _do_check_cublas() -> bool:
    if sys.platform == "win32":
        dll_names = ["cublas64_12.dll", "cublas64_11.dll"]
    else:
        dll_names = ["libcublas.so.12", "libcublas.so.11", "libcublas.so"]

    for name in dll_names:
        try:
            ctypes.CDLL(name)
            print(f"[CUDA] cuBLAS найден: {name}")
            return True
        except OSError:
            continue

    cublas_lib_dirs: List[Path] = []
    try:
        import nvidia.cublas as _cublas_mod
        mod_file = getattr(_cublas_mod, "__file__", None)
        if mod_file is not None:
            cublas_lib_dirs.append(Path(mod_file).parent / "lib")
            cublas_lib_dirs.append(Path(mod_file).parent)
        mod_path = getattr(_cublas_mod, "__path__", None)
        if mod_path:
            for p in mod_path:
                cublas_lib_dirs.append(Path(p) / "lib")
                cublas_lib_dirs.append(Path(p))
    except ImportError:
        pass

    try:
        import nvidia.cublas.lib as _cublas_lib_mod
        mod_file = getattr(_cublas_lib_mod, "__file__", None)
        if mod_file is not None:
            cublas_lib_dirs.append(Path(mod_file).parent)
        mod_path = getattr(_cublas_lib_mod, "__path__", None)
        if mod_path:
            for p in mod_path:
                cublas_lib_dirs.append(Path(p))
    except ImportError:
        pass

    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cublas")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                cublas_lib_dirs.append(Path(loc) / "lib")
                cublas_lib_dirs.append(Path(loc))
                cublas_lib_dirs.append(Path(loc) / "bin")
    except Exception:
        pass

    for d in cublas_lib_dirs:
        if not d.exists():
            continue
        try:
            if sys.platform == "win32":
                os.add_dll_directory(str(d))
            for name in dll_names:
                try:
                    ctypes.CDLL(name)
                    print(f"[CUDA] cuBLAS найден через pip-пакет: {d}")
                    return True
                except OSError:
                    full = d / name
                    if full.exists():
                        try:
                            ctypes.CDLL(str(full))
                            print(f"[CUDA] cuBLAS найден: {full}")
                            return True
                        except OSError:
                            continue
        except OSError:
            continue

    cuda_paths: List[Path] = []
    if sys.platform == "win32":
        cuda_path_env = os.environ.get("CUDA_PATH", "")
        if cuda_path_env:
            cuda_paths.append(Path(cuda_path_env) / "bin")
        for ver in ["13.0", "12.8", "12.6", "12.4", "12.2", "12.1", "12.0", "11.8"]:
            cuda_paths.append(
                Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{ver}/bin")
            )
    else:
        for ver in ["13", "12", "11"]:
            cuda_paths.append(Path(f"/usr/local/cuda-{ver}/lib64"))
        cuda_paths.append(Path("/usr/local/cuda/lib64"))

    for p in cuda_paths:
        if not p.exists():
            continue
        try:
            if sys.platform == "win32":
                os.add_dll_directory(str(p))
            for name in dll_names:
                full = p / name
                if full.exists():
                    try:
                        ctypes.CDLL(str(full))
                        print(f"[CUDA] cuBLAS найден в CUDA Toolkit: {p}")
                        return True
                    except OSError:
                        continue
                try:
                    ctypes.CDLL(name)
                    print(f"[CUDA] cuBLAS найден после добавления {p}")
                    return True
                except OSError:
                    continue
        except OSError:
            continue

    try:
        import ctranslate2
        ct2_dir = Path(ctranslate2.__file__).parent
        for subdir in [ct2_dir, ct2_dir / "lib",
                       ct2_dir.parent / "nvidia" / "cublas" / "lib"]:
            if subdir.exists():
                try:
                    if sys.platform == "win32":
                        os.add_dll_directory(str(subdir))
                    for name in dll_names:
                        try:
                            ctypes.CDLL(name)
                            print(f"[CUDA] cuBLAS найден через ctranslate2: {subdir}")
                            return True
                        except OSError:
                            continue
                except OSError:
                    continue
    except (ImportError, AttributeError):
        pass

    return False


# ------------------- Загрузка моделей -------------------
def _load_hf_pipeline(model_path: str):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = _device()
    dtype = _dtype()

    print(f"\n{'='*60}")
    print(f"[MODEL] Загрузка: {model_path}")
    print(f"[MODEL] Устройство: {device}, dtype: {dtype}")
    print(f"{'='*60}\n")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,          # никаких сетевых запросов
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    print(f"[MODEL] {model_path} загружена!\n")
    return pipe


def _ensure_ct2_files(source_model_dir: Path, output_dir: Path):
    needed = [
        "tokenizer.json", "preprocessor_config.json", "config.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "normalizer.json", "vocab.json", "merges.txt",
    ]
    source = Path(source_model_dir)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for fname in needed:
        src_file = source / fname
        dst_file = dest / fname
        if src_file.exists() and not dst_file.exists():
            shutil.copy2(str(src_file), str(dst_file))


def _convert_whisper_to_ct2(local_model_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    quantization = "float16" if torch.cuda.is_available() else "int8"
    print(f"\n[CONVERT] {local_model_path} -> CTranslate2 ({quantization})...")

    errors: list = []

    try:
        from ctranslate2.converters import TransformersConverter
        converter = TransformersConverter(local_model_path)
        converter.convert(str(output_dir), quantization=quantization, force=True)
        if (output_dir / "model.bin").exists():
            _ensure_ct2_files(Path(local_model_path), output_dir)
            return
        errors.append("TransformersConverter: model.bin не создан")
    except (ImportError, Exception) as e:
        errors.append(f"TransformersConverter: {e}")

    for imp in ("ctranslate2.converters.whisper", "ctranslate2.converters"):
        try:
            mod = __import__(imp, fromlist=["WhisperConverter"])
            WC = getattr(mod, "WhisperConverter")
            converter = WC(local_model_path)
            converter.convert(str(output_dir), quantization=quantization, force=True)
            if (output_dir / "model.bin").exists():
                _ensure_ct2_files(Path(local_model_path), output_dir)
                return
        except (ImportError, AttributeError, Exception) as e:
            errors.append(f"WhisperConverter({imp}): {e}")

    for cmd in [
        [sys.executable, "-m", "ctranslate2.converters.transformers"],
        ["ct2-transformers-converter"],
    ]:
        try:
            result = subprocess.run(
                cmd + ["--model", local_model_path, "--output_dir", str(output_dir),
                       "--quantization", quantization, "--force"],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode == 0 and (output_dir / "model.bin").exists():
                _ensure_ct2_files(Path(local_model_path), output_dir)
                return
            errors.append(f"{cmd[0]}: rc={result.returncode}")
        except Exception as e:
            errors.append(f"{cmd[0]}: {e}")

    raise RuntimeError(
        "Конвертация CT2 не удалась:\n" + "\n".join(f"  - {e}" for e in errors)
    )


def _determine_faster_whisper_device() -> Tuple[str, str]:
    dev = _device()
    if dev == "cpu":
        return "cpu", "int8"
    if _check_cublas_available():
        return "cuda", "float16"
    print("\n[!] cuBLAS не найден — faster-whisper на CPU")
    print("    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n")
    return "cpu", "int8"


def _load_faster_whisper():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        WhisperModel = None

    ct2_dir = MODELS_DIR / "ct2" / "whisper-podlodka-turbo-ct2"
    model_bin = ct2_dir / "model.bin"

    if model_bin.exists():
        dev, compute = _determine_faster_whisper_device()
        print(f"[MODEL] CTranslate2: {ct2_dir} ({dev}/{compute})")
        model = WhisperModel(str(ct2_dir), device=dev, compute_type=compute)
        print("[MODEL] CTranslate2 загружена!\n")
        return model

    conv_err = None
    try:
        _convert_whisper_to_ct2(HF_MODEL_PODLODKA, ct2_dir)
    except RuntimeError as e:
        conv_err = e
        print(f"\n[!] CT2 конвертация не удалась: {e}")

    if model_bin.exists():
        dev, compute = _determine_faster_whisper_device()
        model = WhisperModel(str(ct2_dir), device=dev, compute_type=compute)
        print("[MODEL] CTranslate2 загружена!\n")
        return model

    print("\n[!] faster-whisper недоступен, используем Transformers pipeline")
    print("[!] (результат будет идентичен модели 2)\n")
    return _load_hf_pipeline(HF_MODEL_PODLODKA)


def _is_faster_whisper_model(model) -> bool:
    return type(model).__name__ == "WhisperModel" and hasattr(model, "transcribe")


def _get_model(label: str):
    if label in _cache:
        return _cache[label]

    with _model_load_lock:
        if label in _cache:
            return _cache[label]

        if label == MODEL_LABELS[0]:
            obj = _load_hf_pipeline(HF_MODEL_OPENAI)
        elif label == MODEL_LABELS[1]:
            obj = _load_hf_pipeline(HF_MODEL_PODLODKA)
        elif label == MODEL_LABELS[2]:
            obj = _load_faster_whisper()
        else:
            raise ValueError(f"Неизвестная модель: {label}")

        _cache[label] = obj
        return obj


# ------------------- Транскрибация (ffmpeg внутри pipeline) -------------------
def _transcribe_hf(
    pipe, audio_path: str, language: str, task: str, timestamps: bool
) -> Tuple[str, str]:
    generate_kwargs: Dict[str, Any] = {"task": task}
    if language != "auto":
        generate_kwargs["language"] = language

    result = pipe(
        audio_path, generate_kwargs=generate_kwargs, return_timestamps=True
    )

    if timestamps and "chunks" in result:
        lines = [
            f"[{chunk['timestamp'][0]:.2f}s → {chunk['timestamp'][1]:.2f}s]  {chunk['text'].strip()}"
            for chunk in result["chunks"]
        ]
        return "\n".join(lines), ""

    return result["text"], ""


def _transcribe_faster(
    model, audio_path: str, language: str, task: str, timestamps: bool
) -> Tuple[str, str]:
    lang = LANG_CODE.get(language, language) if language != "auto" else None

    try:
        segments_gen, info = model.transcribe(audio_path, language=lang, task=task)
        segments = list(segments_gen)
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "cublas" in err_msg or "cuda" in err_msg or "cudnn" in err_msg:
            from faster_whisper import WhisperModel
            ct2_dir = MODELS_DIR / "ct2" / "whisper-podlodka-turbo-ct2"
            model_path = (
                str(ct2_dir)
                if (ct2_dir / "model.bin").exists()
                else HF_MODEL_PODLODKA
            )
            new_model = WhisperModel(model_path, device="cpu", compute_type="int8")
            _cache[MODEL_LABELS[2]] = new_model
            segments_gen, info = new_model.transcribe(
                audio_path, language=lang, task=task
            )
            segments = list(segments_gen)
        else:
            raise

    detected = getattr(info, "language", "")

    if timestamps:
        lines = [
            f"[{seg.start:.2f}s → {seg.end:.2f}s]  {seg.text.strip()}"
            for seg in segments
        ]
        return "\n".join(lines), detected

    return " ".join(seg.text.strip() for seg in segments), detected


def _run_single_transcription(
    model_label: str,
    audio_path: str,
    language: str,
    task: str,
    timestamps: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "model": model_label,
        "text": "",
        "detected_lang": "",
        "error": None,
        "load_time": 0.0,
        "transcribe_time": 0.0,
        "total_time": 0.0,
        "gpu_mem_before": 0.0,
        "gpu_mem_after": 0.0,
        "gpu_mem_delta": 0.0,
    }

    t_load_start = time.perf_counter()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["gpu_mem_before"] = torch.cuda.memory_allocated(0) / 1_073_741_824

        model = _get_model(model_label)
        t_load_end = time.perf_counter()
        result["load_time"] = t_load_end - t_load_start

        t_trans_start = time.perf_counter()

        if model_label == MODEL_LABELS[2] and _is_faster_whisper_model(model):
            text, detected = _transcribe_faster(
                model, audio_path, language, task, timestamps
            )
        elif model_label in (MODEL_LABELS[0], MODEL_LABELS[1], MODEL_LABELS[2]):
            text, detected = _transcribe_hf(
                model, audio_path, language, task, timestamps
            )
        else:
            text, detected = "Неизвестный бэкенд", ""

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_trans_end = time.perf_counter()
        result["transcribe_time"] = t_trans_end - t_trans_start

        if torch.cuda.is_available():
            result["gpu_mem_after"] = torch.cuda.memory_allocated(0) / 1_073_741_824
            result["gpu_mem_delta"] = (
                result["gpu_mem_after"] - result["gpu_mem_before"]
            )

        result["text"] = text
        result["detected_lang"] = detected
        result["total_time"] = result["load_time"] + result["transcribe_time"]

    except Exception as exc:
        result["error"] = f"{exc}\n{traceback.format_exc()}"
        result["total_time"] = time.perf_counter() - t_load_start

    return result


def _run_all_parallel(
    audio_path: str,
    language: str,
    task: str,
    timestamps: bool,
) -> Tuple[str, str]:
    ram_before = _get_ram_info()
    gpu_before = _get_gpu_memory_info()
    wall_start = time.perf_counter()

    print(f"\n{'═'*60}")
    print(f"  ПАРАЛЛЕЛЬНЫЙ ЗАПУСК ВСЕХ МОДЕЛЕЙ")
    print(f"{'═'*60}\n")

    print("[PARALLEL] Предзагрузка моделей...")
    for label in MODEL_LABELS:
        try:
            t0 = time.perf_counter()
            _get_model(label)
            dt = time.perf_counter() - t0
            print(f"  ✓ {label}  ({dt:.1f}s)")
        except Exception as e:
            print(f"  ✗ {label}  ОШИБКА: {e}")

    print(f"\n[PARALLEL] Запуск транскрибации...\n")

    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_LABELS)) as executor:
        future_to_label = {}
        for label in MODEL_LABELS:
            future = executor.submit(
                _run_single_transcription,
                label, audio_path, language, task, timestamps,
            )
            future_to_label[future] = label

        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            try:
                res = future.result(timeout=600)
                results.append(res)
                status = "✓" if not res["error"] else "✗"
                print(f"  {status} {label}: {res['total_time']:.2f}s")
            except Exception as e:
                results.append({
                    "model": label,
                    "text": "",
                    "detected_lang": "",
                    "error": str(e),
                    "load_time": 0, "transcribe_time": 0, "total_time": 0,
                    "gpu_mem_before": 0, "gpu_mem_after": 0, "gpu_mem_delta": 0,
                })

    wall_end = time.perf_counter()
    wall_total = wall_end - wall_start

    ram_after = _get_ram_info()
    gpu_after = _get_gpu_memory_info()

    label_order = {label: i for i, label in enumerate(MODEL_LABELS)}
    results.sort(key=lambda r: label_order.get(r["model"], 99))

    text_parts = []
    for i, res in enumerate(results, 1):
        short_name = res["model"].split("(")[0].strip()
        text_parts.append(f"{'━'*60}")
        text_parts.append(f"  МОДЕЛЬ {i}: {short_name}")
        text_parts.append(f"{'━'*60}")
        if res["error"]:
            text_parts.append(f"  ❌ ОШИБКА: {res['error'][:200]}")
        else:
            text_parts.append(res["text"])
        text_parts.append("")

    combined_text = "\n".join(text_parts)

    info_lines = []
    info_lines.append(f"{'═'*50}")
    info_lines.append(f"  СРАВНИТЕЛЬНЫЙ ОТЧЁТ")
    info_lines.append(f"{'═'*50}")
    info_lines.append(f"  Общее время (wall clock): {wall_total:.2f} сек")
    info_lines.append("")
    info_lines.append(f"  {'Модель':<45} {'Загр.':<8} {'Транс.':<8} {'Итого':<8}")
    info_lines.append(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8}")

    for res in results:
        short = res["model"][:42]
        if res["error"]:
            info_lines.append(f"  {short:<45} {'ОШИБКА':<24}")
        else:
            info_lines.append(
                f"  {short:<45} "
                f"{res['load_time']:<8.2f} "
                f"{res['transcribe_time']:<8.2f} "
                f"{res['total_time']:<8.2f}"
            )

    successful = [r for r in results if not r["error"]]
    if successful:
        fastest = sorted(successful, key=lambda r: r["transcribe_time"])
        info_lines.append("")
        info_lines.append("  🏆 Рейтинг по скорости транскрибации:")
        for rank, res in enumerate(fastest, 1):
            short = res["model"].split("(")[0].strip()[:35]
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f" {rank}.")
            info_lines.append(f"    {medal} {short}  —  {res['transcribe_time']:.2f}s")

    info_lines.append("")
    info_lines.append(f"  {'─'*50}")
    info_lines.append(f"  РЕСУРСЫ:")

    if torch.cuda.is_available():
        device_name = f"GPU: {torch.cuda.get_device_name(0)}"
        info_lines.append(f"  {device_name}")
        info_lines.append(
            f"  VRAM до: {gpu_before['allocated']:.2f} ГБ"
            f"  → после: {gpu_after['allocated']:.2f} ГБ"
            f"  (из {gpu_before['total']:.1f} ГБ)"
        )
    else:
        info_lines.append(f"  CPU")

    if ram_before["total"] > 0:
        info_lines.append(
            f"  RAM до: {ram_before['used']:.1f} ГБ ({ram_before['percent']:.0f}%)"
            f"  → после: {ram_after['used']:.1f} ГБ ({ram_after['percent']:.0f}%)"
            f"  (из {ram_before['total']:.0f} ГБ)"
        )

    cublas = _check_cublas_available() if torch.cuda.is_available() else False
    if torch.cuda.is_available() and not cublas:
        info_lines.append("")
        info_lines.append("  ⚠️ cuBLAS не найден — faster-whisper на CPU")
        info_lines.append("  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")

    combined_info = "\n".join(info_lines)
    print(f"\n[PARALLEL] Завершено за {wall_total:.2f}s\n")
    return combined_text, combined_info


def transcribe(
    audio_path: Optional[str],
    model_label: str,
    language: str,
    task: str,
    timestamps: bool,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    if not audio_path:
        return "Пожалуйста, загрузите или запишите аудио.", ""

    if model_label not in ALL_MODEL_CHOICES:
        return f"Неизвестная модель: {model_label}", ""

    if model_label == PARALLEL_LABEL:
        progress(0.0, desc="Загрузка всех моделей...")
        progress(0.1, desc="Параллельная транскрибация...")
        text, info = _run_all_parallel(audio_path, language, task, timestamps)
        progress(1.0, desc="Готово!")
        return text, info

    t0 = time.time()

    try:
        progress(0.0, desc="Загрузка модели...")
        model = _get_model(model_label)

        progress(0.3, desc="Транскрибация...")
        detected_lang = ""

        if model_label == MODEL_LABELS[2] and _is_faster_whisper_model(model):
            text, detected_lang = _transcribe_faster(
                model, audio_path, language, task, timestamps
            )
        elif model_label in (MODEL_LABELS[0], MODEL_LABELS[1], MODEL_LABELS[2]):
            text, detected_lang = _transcribe_hf(
                model, audio_path, language, task, timestamps
            )
        else:
            text = "Неизвестный бэкенд"

        progress(1.0, desc="Готово!")
        elapsed = time.time() - t0

        actual_dev = _device()
        device_name = "CUDA" if actual_dev.startswith("cuda") else "CPU"
        gpu_name = ""
        if actual_dev.startswith("cuda"):
            gpu_name = f" ({torch.cuda.get_device_name(0)})"
        mode_str = f" [режим: {_forced_device}]" if _forced_device != "auto" else ""

        info_parts = [
            f"Модель: {model_label}",
            f"Время: {elapsed:.2f} сек",
            f"Устройство: {device_name}{gpu_name}{mode_str}",
            f"Папка моделей: {MODELS_DIR}",
        ]
        if detected_lang:
            info_parts.append(f"Определённый язык: {detected_lang}")

        if model_label == MODEL_LABELS[2] and torch.cuda.is_available():
            if not _check_cublas_available():
                info_parts.append("")
                info_parts.append("⚠️ faster-whisper на CPU (cuBLAS не найден)")
                info_parts.append("pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")

        return text, "\n".join(info_parts)

    except Exception as exc:
        elapsed = time.time() - t0
        err = f"Ошибка: {exc}\n\n{traceback.format_exc()}"
        return err, f"Время: {elapsed:.2f} сек"


def unload_all_models() -> str:
    count = len(_cache)
    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        f"Выгружено моделей: {count}\n"
        f"Память освобождена.\n"
        f"Файлы моделей остаются в {MODELS_DIR}"
    )


def switch_device(new_device: str) -> str:
    global _forced_device
    if new_device == "GPU" and not torch.cuda.is_available():
        _forced_device = "CPU"
        return "❌ GPU недоступен на этой машине.\n   Работа продолжится на CPU."

    old = _forced_device
    _forced_device = new_device
    count = len(_cache)
    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    actual = _device()
    dtype = _dtype()

    lines = [
        f"🔄 Устройство переключено: {old} → {new_device}",
        f"   Фактическое устройство: {actual}",
        f"   Тип данных: {dtype}",
    ]
    if count > 0:
        lines.append(f"   Выгружено моделей: {count} (будут перезагружены)")

    if actual.startswith("cuda"):
        lines.append(f"   GPU: {torch.cuda.get_device_name(0)}")
        vram = _get_vram_total_gb()
        lines.append(f"   VRAM: {vram:.1f} ГБ")
    else:
        ram = _get_ram_info()
        if ram['total'] > 0:
            lines.append(f"   RAM: {ram['available']:.1f} ГБ свободно из {ram['total']:.0f} ГБ")

    return "\n".join(lines)


def show_models_info() -> str:
    lines = [f"Папка моделей: {MODELS_DIR}"]
    lines.append(f"Режим устройства: {_forced_device} (фактически: {_device()})\n")

    total_size = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(MODELS_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
                file_count += 1

    if total_size > 0:
        if total_size > 1_073_741_824:
            size_str = f"{total_size / 1_073_741_824:.2f} ГБ"
        else:
            size_str = f"{total_size / 1_048_576:.1f} МБ"
        lines.append(f"Общий размер: {size_str}  ({file_count} файлов)")
    else:
        lines.append("Модели ещё не скачаны.")
    lines.append("")

    if torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu = _get_gpu_memory_info()
        lines.append(
            f"VRAM: {gpu['allocated']:.1f} / {gpu['total']:.1f} ГБ "
            f"(зарез. {gpu['reserved']:.1f} ГБ)"
        )
    else:
        lines.append("GPU: не обнаружен (работа на CPU)")

    ram = _get_ram_info()
    if ram["total"] > 0:
        lines.append(
            f"RAM: {ram['used']:.1f} / {ram['total']:.0f} ГБ "
            f"({ram['percent']:.0f}%)"
        )

    cublas_ok = _check_cublas_available() if torch.cuda.is_available() else False
    lines.append(
        f"cuBLAS: {'доступен' if cublas_ok else 'НЕ НАЙДЕН (faster-whisper на CPU)'}"
    )

    ct2_dir = MODELS_DIR / "ct2" / "whisper-podlodka-turbo-ct2"
    ct2_ready = (ct2_dir / "model.bin").exists()
    lines.append(f"\nМодели в памяти:")
    loaded = list(_cache.keys())
    for label in MODEL_LABELS:
        status = " ← загружена" if label in loaded else ""
        lines.append(f"  [{'+' if label in loaded else ' '}] {label}{status}")
    lines.append(f"\nCT2: {'готова' if ct2_ready else 'не конвертирована'}")

    if not cublas_ok and torch.cuda.is_available():
        lines.append("")
        lines.append("── GPU для faster-whisper ──")
        lines.append("pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")

    return "\n".join(lines)


def build_app():
    with gr.Blocks(
        title="Whisper Multi-Model Transcription",
        analytics_enabled=False,
    ) as app:
        gr.Markdown(
            """
            # 🎙️ Multi-Model Whisper Transcription

            Транскрибация аудио тремя вариантами Whisper.
            Режим **«Все модели параллельно»** запускает все 3 модели
            и показывает сравнение скорости и ресурсов.

            | # | Модель | Бэкенд | Размер |
            |---|--------|--------|--------|
            | 1 | openai/whisper-large-v3-turbo | HF Transformers | ~3 ГБ |
            | 2 | bond005/whisper-podlodka-turbo | HF Transformers | ~3 ГБ |
            | 3 | Podlodka CTranslate2 | faster-whisper | из модели 2 |
            | ⚡ | **Все параллельно** | сравнение | все сразу |
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",          # ← ffmpeg работает с файлом
                    label="Аудио (загрузите файл или запишите)",
                )

                model_dropdown = gr.Dropdown(
                    choices=ALL_MODEL_CHOICES,
                    value=MODEL_LABELS[0],
                    label="Модель",
                    allow_custom_value=False,
                )

                device_radio = gr.Radio(
                    choices=DEVICE_CHOICES,
                    value=_forced_device,
                    label="⚡ Устройство",
                )

                language_dropdown = gr.Dropdown(
                    choices=LANGUAGE_OPTIONS,
                    value="auto",
                    label="Язык (auto = автоопределение)",
                    allow_custom_value=False,
                )

                task_radio = gr.Radio(
                    choices=["transcribe", "translate"],
                    value="transcribe",
                    label="Задача (translate = перевод на английский)",
                )

                timestamps_checkbox = gr.Checkbox(
                    label="Показать временные метки",
                    value=False,
                )

                with gr.Row():
                    btn_transcribe = gr.Button(
                        "🎯 Транскрибировать",
                        variant="primary",
                        scale=3,
                    )
                with gr.Row():
                    btn_unload = gr.Button(
                        "🗑️ Выгрузить модели",
                        variant="secondary",
                        scale=1,
                    )
                    btn_info = gr.Button(
                        "📊 Статус",
                        variant="secondary",
                        scale=1,
                    )

            with gr.Column(scale=2, min_width=500):
                output_text = gr.Textbox(
                    label="Результат транскрибации",
                    lines=28,
                    placeholder="Здесь появится распознанный текст...",
                )
                output_info = gr.Textbox(
                    label="Информация / Сравнительный отчёт",
                    lines=18,
                    interactive=False,
                )

        btn_transcribe.click(
            fn=transcribe,
            inputs=[
                audio_input, model_dropdown, language_dropdown,
                task_radio, timestamps_checkbox,
            ],
            outputs=[output_text, output_info],
        )

        device_radio.change(fn=switch_device, inputs=[device_radio], outputs=[output_info])
        btn_unload.click(fn=unload_all_models, inputs=[], outputs=[output_info])
        btn_info.click(fn=show_models_info, inputs=[], outputs=[output_info])

        gr.Markdown(f"---\nПапка моделей: `{MODELS_DIR}`")

    return app


if __name__ == "__main__":
    if GRADIO_CACHE.exists():
        print(f"[INFO] Очистка кэша Gradio: {GRADIO_CACHE}")
        try:
            shutil.rmtree(GRADIO_CACHE, ignore_errors=True)
        except Exception:
            pass

    print("=" * 60)
    print("  Whisper Multi-Model Transcription")
    print("=" * 60)
    print(f"  Gradio       : {gr.__version__}")
    print(f"  PyTorch      : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        vram = _get_vram_total_gb()
        print(f"  VRAM         : {vram:.1f} ГБ")
        cublas = _check_cublas_available()
        print(f"  cuBLAS       : {'Доступен' if cublas else 'НЕ НАЙДЕН'}")
        if not cublas:
            print()
            print(f"  ⚠️  cuBLAS не найден — faster-whisper будет на CPU")
            print(f"  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
    else:
        print(f"  GPU          : НЕ ОБНАРУЖЕН — работа на CPU")
        print(f"  Режим        : CPU (float32)")

    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  RAM          : {ram.total / 1_073_741_824:.0f} ГБ "
              f"({ram.percent}% занято)")
    except ImportError:
        print(f"  RAM          : pip install psutil (для мониторинга)")

    print(f"  Папка моделей: {MODELS_DIR}")
    print("=" * 60)
    print()

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True,
    )