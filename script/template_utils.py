import logging
import cv2


def find_template(frame_bgr, tmpl_gray, threshold=0.82, scales=None):
    """Locate ``tmpl_gray`` within ``frame_bgr``.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Source image in BGR color space.
    tmpl_gray : np.ndarray
        Grayscale template to match against ``frame_bgr``.
    threshold : float, optional
        Minimum normalized correlation score to consider a match valid.
    scales : Iterable[float], optional
        Scales at which to search for the template. If ``None`` a single
        search at 1.0 scale is performed.

    Returns
    -------
    tuple
        ``(box, score, heatmap)`` where ``box`` is ``(x, y, w, h)`` of the best
        match or ``None`` if ``score`` falls below ``threshold``.
    """
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = tmpl_gray.shape[:2]
    best = (None, -1, None)

    for s in (scales or [1.0]):
        th, tw = int(h0 * s), int(w0 * s)
        if th < 10 or tw < 10:
            continue
        if th > frame_gray.shape[0] or tw > frame_gray.shape[1]:
            logging.debug(
                "Template %sx%s exceeds frame %sx%s at scale %.2f, skipping",
                tw,
                th,
                frame_gray.shape[1],
                frame_gray.shape[0],
                s,
            )
            continue
        tmpl = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best[1]:
            x, y = max_loc
            best = ((x, y, tw, th), max_val, res)

    box, score, heat = best
    if score >= threshold:
        return box, score, heat
    return None, score, heat
