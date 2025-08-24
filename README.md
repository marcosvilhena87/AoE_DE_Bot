# AoE DE Bot

## Setup

1. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
2. Copy `config.sample.json` to `config.json` if needed.
3. Configure the Tesseract executable path by either:
   - setting the `tesseract_path` value in `config.json`, or
   - setting the `TESSERACT_CMD` environment variable to the executable path.

If neither option is provided, `pytesseract` will attempt to find `tesseract` on the system `PATH`.

## Configuration notes

The bot detects the HUD by matching a single template of the resource bar
(`assets/resources.png`). This image is also used to read the individual
resource icons. Frames are captured from the whole screen unless a different
region is explicitly requested.

HUD-related coordinates, such as `areas.pop_box`, use ``[x, y, width, height]``
fractions of the entire screen. The default values in `config.json` are
placeholders and should be calibrated for your setup.

The `resource_panel` section includes `max_width`, which caps the width of the region used to read each resource value. When extra space is available between icons, the ROI is centered and limited to this width (default `160`).

### OCR tuning

Two fields in `config.json` allow adjusting how resource numbers are read:

* `ocr_kernel_size` – size of the square kernel used for morphological dilation
  before running OCR (default `2`).
* `ocr_psm_list` – list of Tesseract [page segmentation modes](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#page-segmentation-method)
  tried in order when extracting digits (default `[6, 7, 8, 10, 13]`).

## Capturing `assets/resources.png`

`assets/resources.png` is the only template needed for HUD detection and
resource icon recognition. The previous `hud_resources.png` file is no longer
used.

1. Set the game's UI scale to 100%.
2. Take a screenshot of the in-game HUD.
3. Crop a tight image around the resource bar and save it as
   `assets/resources.png`.
4. If your layout differs between profiles, capture and replace this template
   for each one to ensure `wait_hud()` can anchor correctly.

## Resource Icons

`assets/resources.png` contains the resource icons in order: wood stockpile,
food stockpile, gold stockpile, stone stockpile, population limit, and idle
villager.

### Customizing required icons

The bot determines which HUD icons must be read through the `hud_icons`
section in `config.json`:

```json
"hud_icons": {
  "required": ["wood_stockpile", "food_stockpile", "gold_stockpile", "stone_stockpile", "population_limit", "idle_villager"],
  "optional": []
}
```

Entries in `required` cause the bot to raise an error when any of those icons
cannot be detected or read. Icons listed under `optional` are attempted but do
not stop execution if they are missing. Adjust these lists to match the
resources shown in your game profile.

## Calibration helper

To calibrate the `areas.pop_box` fractions interactively, run:

```
python tools/calibrate_pop_box.py
```

The script waits for the HUD to be detected, shows a screenshot, and lets you
draw the population box. The normalized `[x, y, width, height]` values are
printed and you can choose to write them back to `config.json` automatically.

