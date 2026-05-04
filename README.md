# Guard (SHIELD) Repository

This repository contains the Guard (SHIELD) product including its UI, CLI, and async service.

## Prerequisites

Before installing the project, make sure you have the following installed on your system:
- **Python:** 3.10 or higher
- **ffmpeg / ffprobe:** (Optional) for media metadata and frame extraction
- **tesseract:** (Optional) for OCR support

## Installation and Setup

It is highly recommended to set up the project within a Python Virtual Environment (`venv`) to avoid interfering with system packages. The provided scripts expect the virtual environment to be located in the `shield_product` directory.

To initialize the environment and install dependencies, run the following commands from the root of the repository:

```bash
# 1. Navigate to the product directory
cd shield_product

# 2. Create the Python virtual environment
python -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# 4. Install the required Python packages
pip install -r requirements.txt
```

## How to Run

Once installed, there are several ways to execute the applications within the Guard (SHIELD) repo.

### 1. CLI (Single File Analysis)

You can run the inference CLI tool against a single file. You will need to export necessary API keys (like `GROQ_API_KEY`) before running.

**From the repository root (using the bash script):**
```bash
export GROQ_API_KEY="YOUR_NEW_KEY"
./run_guard.sh --input sample_for_analysis.txt --modality auto
```

**Or directly via Python (from the `shield_product` folder):**
```bash
export GROQ_API_KEY="YOUR_NEW_KEY"
python ../run.py --input ../sample_for_analysis.txt --modality auto
```

### 2. UI (Streamlit Viewer)

To run the interactive UI dashboard with Streamlit:

```bash
cd shield_product
./run_guard.sh ui
```

### 3. Diagnostics (Doctor)

If you are facing issues with your environment or installation, run the doctor script to perform diagnostics:

```bash
cd shield_product
./run_guard.sh doctor
```

### 4. Async Service (API + Worker)

The async service uses Redis as a broker. To run the full service suite locally, you need three separate terminal windows:

**Terminal 1 (Redis):**
```bash
cd shield_product
./run_guard.sh redis
```

**Terminal 2 (API Server):**
```bash
cd shield_product
./run_guard.sh serve
```

**Terminal 3 (Celery Worker):**
```bash
cd shield_product
./run_guard.sh worker
```

## Notes

- **Weights & Models:** Some optional model components download their specific weights upon first execution. This might cause the first run to take longer.
- **Minimal Run Environment Variables:** Need a lighter execution environment? You can disable specific processes by running with these environment variables prior to launch:
  ```bash
  export GUARD_DISABLE_EMBEDDINGS=1
  export GUARD_DISABLE_HATE_CLASSIFIER=1
  ```
