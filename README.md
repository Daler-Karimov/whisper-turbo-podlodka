# Whisper Multi-Model Transcription & Benchmark

[English](#english) | [Русский](#русский)

---

<a name="english"></a>
## English

### Overview
**Whisper Multi-Model Transcription** is an interactive web application powered by **Gradio** and **PyTorch** designed for high-performance audio transcription and translation. It provides a flexible workbench for side-by-side evaluation of OpenAI's official Whisper models and community fine-tuned checkpoints (optimized for Russian speech), offering real-time parallel benchmarking across **Transformers** and **CTranslate2 (faster-whisper)** backends.

---

### Supported Models
| # | Model | Backend | Description |
|---|-------|---------|-------------|
| 1 | `openai/whisper-large-v3-turbo` | HF Transformers | OpenAI's official large-v3-turbo model, optimized for fast and accurate multilingual transcription and translation. |
| 2 | `bond005/whisper-podlodka-turbo` | HF Transformers | Community fine-tuned model optimized specifically for Russian conversational speech, interviews, and podcasts. |
| 3 | `Podlodka-Turbo CTranslate2` | faster-whisper | Quantized CTranslate2 conversion of Model 2 with cuBLAS GPU acceleration and CPU int8 fallback for maximum speed. |
| * | **ALL MODELS PARALLEL** | Benchmark | Runs all 3 models concurrently on the same audio file with a full performance and memory report. |

---

### Key Features
- **Parallel Multi-Model Comparison**: Run all models concurrently and compare transcription accuracy, execution time, and speed ranking (1st / 2nd / 3rd).
- **Flexible Audio Input**: Upload audio files (`.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`) or record directly from your microphone.
- **Transcription & Translation**: Transcribe in the original language or translate directly into English (`translate` task).
- **Timestamped Segments**: Toggle detailed chunk-level timestamps `[0.00s -> 4.50s]` for subtitles and transcripts.
- **Dynamic Device Switching**: Switch between **GPU (CUDA float16)** and **CPU (int8/float32)** on the fly without restarting.
- **Resource & Memory Monitoring**: Live telemetry of GPU VRAM usage, system RAM, cuBLAS status, and active model cache.
- **Memory Management**: Unload models from RAM/VRAM with a single click.
- **Public Share Link & Auto-open**: Generates a public Gradio link (`https://...gradio.live`) and automatically opens your browser on launch.
- **One-Click Automated Setup**: `start.bat` provisions an isolated virtual environment (`venv`) and installs dependencies automatically.

---

### Prerequisites
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 - 3.13
- **Git** & **Git LFS** (required to download model weights)
- **NVIDIA GPU** *(Optional but recommended for CUDA acceleration; RTX/GTX supported)*
- **FFmpeg**: Required for audio decoding

---

### Quick Start

#### Windows (One-Click Launch)
Simply double-click `start.bat` or run in terminal:
```bat
start.bat
```
The script will:
1. Verify Python availability.
2. Automatically create and configure `venv` if not present.
3. Install dependencies from `requirements.txt`.
4. Check GPU / CUDA support.
5. Launch the application, open your browser (`http://localhost:7860`), and generate a public Gradio share link.

#### Manual Installation (Any OS)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Daler-Karimov/whisper-turbo-podlodka.git
   cd whisper-turbo-podlodka
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > **Note for NVIDIA GPU users**: For CUDA acceleration, ensure PyTorch with CUDA support is installed:
   > ```bash
   > pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   > pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   > ```
4. **Run the app**:
   ```bash
   python app.py
   ```

---

### Note on Initial Model Download (Git LFS)
On the first run, model weights are cloned into the local `models/` directory.
When cloning, Git may display:
```text
Filtering content:  80% (4/5)
```
**This is expected**: Git LFS is transferring the large model weights file (`model.safetensors`, ~1.6 - 3.2 GB). The terminal may seem paused while downloading large files. Once finished, files are cached locally and will not need to be downloaded again.

---

<a name="русский"></a>
## Русский

### Описание
**Whisper Multi-Model Transcription** — интерактивное веб-приложение на базе **Gradio** и **PyTorch** для высокоточного распознавания и перевода речи. Проект предоставляет удобную среду для сравнительного тестирования оригинальных моделей OpenAI Whisper и дообученных версий (адаптированных под русскую речь), поддерживая параллельный бенчмарк на движках **Transformers** и **CTranslate2 (faster-whisper)**.

---

### Поддерживаемые модели
| # | Модель | Бэкенд | Описание |
|---|--------|--------|----------|
| 1 | `openai/whisper-large-v3-turbo` | HF Transformers | Официальная модель OpenAI large-v3-turbo для быстрого мультиязычного распознавания. |
| 2 | `bond005/whisper-podlodka-turbo` | HF Transformers | Дообученная модель, специально адаптированная для сложной русской речи, интервью и подкастов. |
| 3 | `Podlodka-Turbo CTranslate2` | faster-whisper | Версия Модели 2, конвертированная в CTranslate2 (float16/int8) для максимальной скорости на GPU и CPU. |
| * | **ВСЕ МОДЕЛИ ПАРАЛЛЕЛЬНО** | Сравнение | Одновременный запуск всех трёх моделей на одном аудиофайле с отчётом по скорости и ресурсам. |

---

### Основные возможности
- **Параллельное сравнение моделей**: Запуск всех моделей одновременно с генерацией детального отчёта, замером времени и рейтингом скорости (1-е / 2-е / 3-е места).
- **Гибкий ввод аудио**: Загрузка аудиофайлов (`.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`) или прямая запись с микрофона.
- **Транскрибация и перевод**: Распознавание на исходном языке либо прямой перевод на английский язык (`translate`).
- **Временные метки (таймкоды)**: Посегментный вывод временных меток вида `[0.00s -> 4.50s]` для удобной навигации и субтитров.
- **Переключение GPU / CPU на лету**: Переключение между видеокартой и процессором прямо в интерфейсе без перезапуска.
- **Мониторинг ресурсов**: Отображение доступности cuBLAS, выделенной памяти VRAM, оперативной памяти RAM и кэша моделей.
- **Управление памятью**: Кнопка быстрой выгрузки моделей из видеопамяти и RAM в один клик.
- **Публичная ссылка и автозапуск браузера**: Генерация публичной ссылки (`https://...gradio.live`) для удалённого доступа и автоматическое открытие вкладки в браузере при старте.
- **Автоматическая установка venv**: Скрипт `start.bat` сам создаёт виртуальное окружение и ставит зависимости.

---

### Системные требования
- **ОС**: Windows 10/11, Linux или macOS
- **Python**: 3.10 - 3.13
- **Git** и **Git LFS** (необходимы для загрузки весов моделей)
- **NVIDIA GPU** *(рекомендуется для быстрого ускорения CUDA; также поддерживается CPU)*
- **FFmpeg**: Для декодирования аудио

---

### Быстрый запуск

#### Windows (запуск в один клик)
Просто запустите файл `start.bat`:
```bat
start.bat
```
Скрипт автоматически:
1. Проверит наличие Python в системе.
2. Создаст изолированное виртуальное окружение `venv` (если ещё не создано).
3. Установит все зависимости из `requirements.txt`.
4. Проверит видеокарту и поддержку CUDA.
5. Запустит сервис, откроет его в браузере (`http://localhost:7860`) и предоставит публичную ссылку Gradio.

#### Ручная установка (любая ОС)
1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/Daler-Karimov/whisper-turbo-podlodka.git
   cd whisper-turbo-podlodka
   ```
2. **Создайте и активируйте виртуальное окружение**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```
3. **Установите зависимости**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > **Для пользователей видеокарт NVIDIA**: Чтобы задействовать ускорение CUDA, установите сборку PyTorch с поддержкой CUDA:
   > ```bash
   > pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   > pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   > ```
4. **Запустите приложение**:
   ```bash
   python app.py
   ```

---

### Примечание о загрузке моделей (Git LFS)
При первом запуске веса моделей клонируются в локальную директорию `models/`.
Если в консоли отображается:
```text
Filtering content:  80% (4/5)
```
**Это нормальный процесс**: Git LFS загружает файл весов модели (`model.safetensors`, от 1.6 до 3.2 ГБ). Во время загрузки больших бинарных файлов Git может не показывать динамический прогресс. Дождитесь завершения скачивания — после этого модели сохраняются локально и больше не скачиваются заново.

---

### License
This project is open-source under the MIT License. Model weights are subject to their respective licenses from OpenAI and bond005.
