import json
from pathlib import Path

import cv2

import campaign_bot as cb

ROOT = Path(__file__).resolve().parents[1]


def main():
    print("Waiting for HUD...")
    anchor, asset = cb.wait_hud()
    print(f"HUD detected using '{asset}' at {anchor}")
    frame = cb._grab_frame()

    r = cv2.selectROI(
        "Draw population box", frame, showCrosshair=True, fromCenter=False
    )
    cv2.destroyAllWindows()
    x, y, w, h = map(int, r)
    if w == 0 or h == 0:
        print("No region selected. Aborting.")
        return

    dx = (x - cb.HUD_ANCHOR["left"]) / cb.HUD_ANCHOR["width"]
    dy = (y - cb.HUD_ANCHOR["top"]) / cb.HUD_ANCHOR["height"]
    width = w / cb.HUD_ANCHOR["width"]
    height = h / cb.HUD_ANCHOR["height"]
    values = [round(dx, 2), round(dy, 2), round(width, 2), round(height, 2)]

    print("Suggested areas.pop_box:")
    print(values)

    cfg_path = ROOT / "config.json"
    resp = input(f"Update {cfg_path}? [y/N] ").strip().lower()
    if resp == "y":
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("areas", {})["pop_box"] = values
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print("config.json updated")
    else:
        print("config.json not modified")


if __name__ == "__main__":
    main()
