"""
Segment a photographed page of handwriting into individual text-line crops,
using a horizontal projection profile (classical CV, no ML).
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _binarize(gray):
    # Adaptive threshold copes with uneven lighting/shadows across a
    # photographed page far better than a single global (Otsu) cutoff,
    # which tends to swallow whole background regions as "ink".
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )


def estimate_skew(image_bgr, angle_range=15.0, probe_width=900, min_blob_width_frac=0.03):
    """
    Estimate the page's rotation angle from the orientation of individual
    text-line blobs, not a single global "rotate and score" search.

    An earlier version maximized the row-projection's variance over
    candidate angles; that assumes one global rotation straightens every
    row equally, which doesn't hold for a handheld photo of an open
    notebook (the page isn't flat -- it curves near the spine), so the
    global optimum it found was often unrelated to the actual text angle.
    Measuring each line blob's own orientation and taking the median is
    robust to that: a few locally-warped lines don't drag the estimate
    off, and it directly measures what we actually care about (will this
    rotation make the row-projection line-finder work) instead of a proxy.
    """
    h0, w0 = image_bgr.shape[:2]
    scale = probe_width / w0
    small = cv2.resize(image_bgr, (probe_width, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)

    # Same horizontal-joining step segment_lines() uses, so each contour
    # below corresponds to one text line rather than one letter.
    k_w = max(3, probe_width // 70)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width = probe_width * min_blob_width_frac
    angles = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), angle = rect
        # normalize to "how far off horizontal", using the long side as
        # width regardless of which dimension OpenCV happened to report
        if rw < rh:
            rw, rh = rh, rw
            angle += 90
        if rw < min_width or rw < rh * 2:
            continue  # too small / not line-shaped to trust its angle
        # OpenCV reports angle in (-90, 0]; fold into (-45, 45]
        if angle <= -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) <= angle_range:
            angles.append(angle)

    if not angles:
        return 0.0
    # Empirically (not just by convention on paper): passing this value
    # straight into cv2.getRotationMatrix2D(angle) and applying it is what
    # actually straightens the image -- verified by re-measuring residual
    # skew after correction rather than trusting the documented sign
    # convention, which didn't match observed behavior here.
    return float(np.median(angles))


def deskew(image_bgr, max_angle=6.0):
    """Rotate the full-resolution image to correct camera/page tilt. Returns
    (deskewed_image, angle_degrees) -- angle is reported so callers/tests can
    see what was applied; a near-zero angle means the page was already
    straight and rotation was skipped to avoid a pointless resample."""
    angle = estimate_skew(image_bgr, angle_range=max_angle)
    if abs(angle) < 0.15:
        return image_bgr, 0.0

    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_bgr, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def split_pages(image_bgr, center_frac=0.28, min_gutter_frac=0.28):
    """
    Detect a two-page book/notebook spread and split it into separate page
    images so a horizontal line projection never has to span two
    independently-flowing columns of text at once (which produces
    meaningless line boundaries). Looks for a low-ink vertical gutter near
    the horizontal center; if none is clearly there, returns the image
    unchanged as a single page -- most photos are already a single page,
    and this must not fire on those.
    """
    h0, w0 = image_bgr.shape[:2]
    probe_width = 900
    scale = probe_width / w0
    small = cv2.resize(image_bgr, (probe_width, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)

    col_density = binary.mean(axis=0).astype(np.float64)
    col_density = gaussian_filter1d(col_density, sigma=4)

    w = col_density.shape[0]
    lo = int(w * (0.5 - center_frac))
    hi = int(w * (0.5 + center_frac))
    if hi <= lo:
        return [image_bgr]

    window = col_density[lo:hi]
    gutter_col = lo + int(np.argmin(window))
    gutter_density = col_density[gutter_col]
    typical_density = np.median(col_density)

    # A real spine gap has noticeably less ink than the page body around
    # it; a single page's center (which may run straight through a
    # sentence) usually doesn't dip nearly this far.
    if typical_density <= 1e-6 or gutter_density > typical_density * (1 - min_gutter_frac):
        return [image_bgr]

    split_x = int(gutter_col / scale)
    left = image_bgr[:, :split_x]
    right = image_bgr[:, split_x:]
    # Reject a lopsided split (e.g. a dark margin/shadow near one edge
    # mistaken for a gutter) -- a real spread splits roughly down the middle.
    ratio = left.shape[1] / max(1, right.shape[1])
    if not (0.5 < ratio < 2.0):
        return [image_bgr]

    return [left, right]


def segment_lines(image_bgr, target_width=1400, margin_frac=0.1, pad_frac=0.006):
    """
    Find text-line bands in a single-column page photo. Deskews first so a
    tilted photo doesn't smear adjacent lines together in the row profile.
    """
    image_bgr, _ = deskew(image_bgr)

    h0, w0 = image_bgr.shape[:2]
    scale = target_width / w0
    small = cv2.resize(image_bgr, (target_width, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)

    # Ignore the outer margins: notebook rule lines commonly extend past
    # the written text and would otherwise dominate the row profile.
    margin = int(target_width * margin_frac)
    central = binary[:, margin: target_width - margin]

    # Join letters/words within a line horizontally so each line forms one
    # solid band in the row projection; kernel size scales with image width.
    k_w = max(3, target_width // 70)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
    closed = cv2.morphologyEx(central, cv2.MORPH_CLOSE, kernel)

    row_sums = (closed > 0).sum(axis=1).astype(np.float64)
    smoothed = gaussian_filter1d(row_sums, sigma=3)

    # Real photos rarely have a profile that drops to zero between lines
    # (cursive ascenders/descenders, rule-line bleed), so find line centers
    # as peaks rather than thresholding for empty bands.
    min_distance = max(8, int(small.shape[0] * 0.02))
    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=smoothed.max() * 0.15)

    if len(peaks) == 0:
        return []

    # Line boundaries: midpoints between consecutive peaks, plus half a
    # line's worth of padding above the first / below the last peak.
    if len(peaks) > 1:
        avg_gap = int(np.mean(np.diff(peaks)))
    else:
        avg_gap = int(small.shape[0] * 0.05)

    bounds = [max(0, peaks[0] - avg_gap // 2)]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append((a + b) // 2)
    bounds.append(min(small.shape[0], peaks[-1] + avg_gap // 2))

    pad = max(2, int(small.shape[0] * pad_frac))
    crops = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        top = max(0, int((s - pad) / scale))
        bottom = min(h0, int((e + pad) / scale))
        if bottom - top < 3:
            continue
        crops.append(((0, top, w0, bottom), image_bgr[top:bottom, 0:w0]))
    return crops
