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
``[x, y, width, height]`` fractions of the full screen. These values are
interpreted relative to the screen; the minimap anchor is only used to adjust
offsets when necessary.

## Capturing `hud_resources.png`

Some HUD layouts place the minimap and the resource bar at different offsets.
The bot now searches for either element, so a suitable `hud_resources.png`
template is required for your configuration.

1. Set the game's UI scale to 100%.
2. Take a screenshot of the in-game HUD.
3. Crop a tight image around the resource bar and save it as
   `assets/hud_resources.png`.
4. If your layout differs between profiles, capture and replace this template
   for each one to ensure `wait_hud()` can anchor correctly.

