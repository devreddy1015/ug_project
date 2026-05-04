# Guard (SHIELD) Repo

This repo contains the Guard (SHIELD) product UI, CLI, and async service.

## Prerequisites

- Python 3.10+
- (Optional) ffmpeg/ffprobe for media metadata and frame extraction
- (Optional) tesseract for OCR

## Setup (Recommended)

Run from the shield_product directory so the launcher uses the expected venv.

```bash
cd shield_product
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### CLI (single file)

From repo root:

```bashexport GROQ_API_KEY="YOUR_NEW_KEY"
./run_guard.sh --input sample_for_analysis.txt --modality auto
```

Or from shield_product:

```bash
python ../run.py --input ../sample_for_analysis.txt --modality auto
```

### UI (Streamlit)

```bash
cd shield_product
./run_guard.sh ui
```

### Diagnostics

```bash
cd shield_product
./run_guard.sh doctor
```

### Async Service (API + worker)

```bash
cd shield_product
./run_guard.sh redis
./run_guard.sh serve
./run_guard.sh worker
```

## Notes

- Some optional model components download weights on first use.
- If you want a minimal run, you can set:
  - GUARD_DISABLE_EMBEDDINGS=1
  - GUARD_DISABLE_HATE_CLASSIFIER=1
