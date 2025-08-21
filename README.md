# AoE DE Bot

## Setup

1. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
2. Copy `config.sample.json` to `config.json` if needed.
3. Configure the Tesseract executable path by either:
   - setting the `tesseract_path` value in `config.json`, or
   - setting the `TESSERACT_CMD` environment variable to the executable path.

If neither option is provided, `pytesseract` will attempt to find `tesseract` on the system `PATH`.

## Configuration notes

The bot locates the minimap on the screen to establish an anchor for HUD
offset calculations. This anchor no longer limits the capture area—frames are
grabbed from the whole screen unless a different region is explicitly
requested.

HUD-related coordinates, such as `areas.pop_box`, are specified as
``[dx, dy, width, height]`` fractions relative to this anchor. The minimap
position serves as the reference origin for these offsets.

