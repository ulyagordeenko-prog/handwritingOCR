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
    #
    # The minimum spacing is measured from the text itself rather than taken
    # as a fraction of the image height: on a tightly-spaced form the
    # height-based figure was larger than the real line pitch and merged
    # rows in pairs, so every second line vanished into its neighbour.
    prominence = smoothed.max() * 0.15
    loose, _ = find_peaks(smoothed, distance=max(3, int(small.shape[0] * 0.004)),
                          prominence=prominence)
    if len(loose) > 2:
        pitch = float(np.median(np.diff(loose)))
        min_distance = max(3, int(pitch * 0.6))
    else:
        min_distance = max(8, int(small.shape[0] * 0.02))

    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=prominence)

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


def crop_to_page(image_bgr, min_area_frac=0.25, max_area_frac=0.98, pad_frac=0.01):
    """
    Trim the desk, fingers and shadow around the sheet, keeping only the paper.

    This is not cosmetic. A vision model converts the picture into a fixed
    budget of visual tokens, so every pixel of desk is budget not spent on
    handwriting -- cropping is free resolution.

    Deliberately timid: the crop is accepted only when a bright, roughly
    rectangular region covers a plausible share of the frame. A photo that is
    already all paper (the common case) has no boundary to find, and a wrong
    crop would cut off text -- far worse than not cropping at all.
    """
    h0, w0 = image_bgr.shape[:2]
    probe_width = 800
    scale = probe_width / w0
    small = cv2.resize(image_bgr, (probe_width, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Paper is the bright part; Otsu finds the paper/surroundings cut when one
    # exists and degenerates harmlessly into "almost everything" when it doesn't
    # -- which the area guard below then rejects.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_bgr, False

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    area_frac = (w * h) / (small.shape[0] * small.shape[1])
    if not (min_area_frac < area_frac < max_area_frac):
        return image_bgr, False

    # Keep a small margin: ascenders and the first/last stroke of a line often
    # sit just outside the detected paper blob.
    pad = int(max(small.shape) * pad_frac)
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(small.shape[1] - x, w + 2 * pad)
    h = min(small.shape[0] - y, h + 2 * pad)

    left, top = int(x / scale), int(y / scale)
    right, bottom = min(w0, int((x + w) / scale)), min(h0, int((y + h) / scale))
    if right - left < w0 * 0.3 or bottom - top < h0 * 0.3:
        return image_bgr, False
    return image_bgr[top:bottom, left:right], True


def split_strips(image_bgr, count, min_gap_rows=6):
    """
    Cut a page into `count` horizontal strips, placing every cut on a boundary
    the line finder already identified, never at a fixed height.

    A fixed cut lands mid-line as often as not, and a line sliced in half is
    unreadable in both halves -- the split would cost more than the added
    resolution gains. Two weaker versions of this were measured and rejected:
    picking the quietest row within a window of the ideal position put one cut
    on a row carrying 127 ink pixels against a page average of 156 (i.e.
    through text), and averaging the profile over a band made it worse (211),
    because in a densely written region no quiet row exists to find. Asking
    segment_lines() where the lines actually are removes the guesswork: the cut
    goes on a real boundary, however far that sits from the ideal position.
    """
    if count < 2:
        return [image_bgr]

    crops = segment_lines(image_bgr)
    h0 = image_bgr.shape[0]
    # Midway between two consecutive line centres -- the point furthest from
    # both. The crop boundaries themselves are not good enough: they carry the
    # padding segment_lines adds and sit where uneven spacing puts them, which
    # measured as little as 0.32 of a line pitch from a line centre, close
    # enough to clip descenders.
    centres = [(bbox[1] + bbox[3]) // 2 for bbox, _ in crops]
    boundaries = [(a + b) // 2 for a, b in zip(centres[:-1], centres[1:])]
    if not boundaries:
        return [image_bgr]

    cuts = []
    for i in range(1, count):
        ideal = h0 * i / count
        best = min(boundaries, key=lambda y: abs(y - ideal))
        # two strips must not share a boundary, and a sliver is not a strip
        if all(abs(best - c) > min_gap_rows for c in cuts):
            cuts.append(best)

    bounds = [0] + sorted(cuts) + [h0]
    strips = [image_bgr[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 10]
    return strips or [image_bgr]
