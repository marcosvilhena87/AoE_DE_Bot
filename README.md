# AoE DE Bot

## Setup

1. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
2. Copy `config.sample.json` to `config.json` if needed.
3. Configure the Tesseract executable path by either:
   - setting the `tesseract_path` value in `config.json`, or
   - setting the `TESSERACT_CMD` environment variable to the executable path.

If neither option is provided, `pytesseract` will attempt to find `tesseract` on the system `PATH`.

