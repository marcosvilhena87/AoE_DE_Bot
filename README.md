# AoE DE Bot

> **Note:** This project is intended for **Age of Empires I: Definitive Edition**. It is not compatible with **Age of Empires II: Definitive Edition**. For example, it targets campaigns such as *Ascent of Egypt* from Age of Empires I.

## Setup

1. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
2. Copy `config.sample.json` to `config.json` if needed.
3. Configure the Tesseract executable path by either:
   - setting the `tesseract_path` value in `config.json`, or
   - setting the `TESSERACT_CMD` environment variable to the executable path.

If neither option is provided, `pytesseract` will attempt to find `tesseract` on the system `PATH`.

## Configuration notes

The bot locates HUD elements (such as the minimap or resource bar) only to
confirm that the interface is visible. Frames are captured from the whole
screen unless a different region is explicitly requested.

HUD-related coordinates, such as `areas.pop_box`, use ``[x, y, width, height]``
fractions of the entire screen. The default values in `config.json` are
placeholders and should be calibrated for your setup.

### OCR tuning

Two fields in `config.json` allow adjusting how resource numbers are read:

* `ocr_kernel_size` – size of the square kernel used for morphological dilation
  before running OCR (default `2`).
* `ocr_psm_list` – list of Tesseract [page segmentation modes](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#page-segmentation-method)
  tried in order when extracting digits (default `[6, 7, 8, 10, 13]`).

## Capturing `hud_resources.png`

Some HUD layouts place the minimap and the resource bar at different positions.
The bot now searches for either element, so a suitable `hud_resources.png`
template is required for your configuration.

1. Set the game's UI scale to 100%.
2. Take a screenshot of the in-game HUD.
3. Crop a tight image around the resource bar and save it as
   `assets/hud_resources.png`.
4. If your layout differs between profiles, capture and replace this template
   for each one to ensure `wait_hud()` can anchor correctly.

## Resource Icons

`assets/resources.png` contains the resource icons in order: wood stockpile,
food stockpile, gold stockpile, stone stockpile, population limit, and idle
villager.

## Calibration helper

To calibrate the `areas.pop_box` fractions interactively, run:

```
python tools/calibrate_pop_box.py
```

The script waits for the HUD to be detected, shows a screenshot, and lets you
draw the population box. The normalized `[x, y, width, height]` values are
printed and you can choose to write them back to `config.json` automatically.

