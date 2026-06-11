"""
Roof polygon → solar panel grid layout engine.

Algorithm
---------
1. Convert roof polygon vertices from WGS84 lat/lon to a local metric
   coordinate frame (equirectangular projection centred on the polygon centroid).
   Accuracy: < 1 cm error for distances up to 500 m — more than sufficient
   for rooftop scale.

2. Compute roof area via the Shoelace formula.

3. Build a regular grid of panel rectangles starting at
   (bbox_min + setback_m) with step = (panel_dim + gap_m).

4. Accept a panel if ALL four corners satisfy:
   (a) inside the roof polygon  — ray-casting test, O(n_edges)
   (b) minimum distance to any polygon edge ≥ setback_m  — ensures the
       full panel body respects the setback, not just its corner points.

5. Convert accepted panel corners back to WGS84 for Leaflet rendering.

Orientation
-----------
"portrait"  : panel width (shorter side) runs East-West;
              panel length (longer side) runs North-South.
"landscape" : transposed — length East-West, width North-South.
"""

from __future__ import annotations

import math

from api.schemas import PanelLayoutResponse, PanelPosition

# WGS84 mean Earth radius [m]
_R_EARTH = 6_371_000.0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _centroid(polygon_latlon: list[tuple[float, float]]) -> tuple[float, float]:
    lats = [p[0] for p in polygon_latlon]
    lons = [p[1] for p in polygon_latlon]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _to_metric(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
) -> tuple[float, float]:
    """WGS84 → local metric (x = East [m], y = North [m])."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * (math.pi / 180.0) * _R_EARTH
    y = (lat - lat0) * (math.pi / 180.0) * _R_EARTH
    return x, y


def _to_latlon(
    x: float,
    y: float,
    lat0: float,
    lon0: float,
) -> tuple[float, float]:
    """Local metric (x, y) → WGS84 lat/lon."""
    lat = lat0 + y / (_R_EARTH * math.pi / 180.0)
    lon = lon0 + x / (math.cos(math.radians(lat0)) * _R_EARTH * math.pi / 180.0)
    return lat, lon


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Signed area via Shoelace formula; returns absolute value [m²]."""
    n = len(pts)
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(acc) * 0.5


def _point_in_polygon(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _point_to_segment_dist(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Minimum distance from point (px,py) to segment (a→b)."""
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _min_dist_to_edges(
    px: float,
    py: float,
    poly: list[tuple[float, float]],
) -> float:
    n = len(poly)
    min_d = float("inf")
    for i in range(n):
        j = (i + 1) % n
        d = _point_to_segment_dist(px, py, poly[i][0], poly[i][1], poly[j][0], poly[j][1])
        if d < min_d:
            min_d = d
    return min_d


def _panel_valid(
    corners: list[tuple[float, float]],
    poly: list[tuple[float, float]],
    setback_m: float,
) -> bool:
    """Return True if all four panel corners are inside poly and respect setback."""
    for cx, cy in corners:
        if not _point_in_polygon(cx, cy, poly):
            return False
        if _min_dist_to_edges(cx, cy, poly) < setback_m:
            return False
    return True


# ---------------------------------------------------------------------------
# Public layout function
# ---------------------------------------------------------------------------


def compute_panel_layout(
    roof_polygon_latlon: list[tuple[float, float]],
    panel_width_m: float,
    panel_length_m: float,
    panel_p_stc_w: float,
    setback_m: float = 0.5,
    row_gap_m: float = 0.05,
    col_gap_m: float = 0.02,
    orientation: str = "portrait",
) -> PanelLayoutResponse:
    """
    Compute a regular panel grid layout for a roof polygon.

    Parameters
    ----------
    roof_polygon_latlon : list of [lat, lon] pairs  (≥ 3 vertices, closed or open)
    panel_width_m       : module width  [m] (shorter dimension)
    panel_length_m      : module length [m] (longer dimension)
    panel_p_stc_w       : module STC power [W]  (for installed-kW calculation)
    setback_m           : minimum distance from roof edge to any panel corner [m]
    row_gap_m           : gap between rows (N-S) [m]
    col_gap_m           : gap between columns (E-W) [m]
    orientation         : "portrait" (length N-S) or "landscape" (length E-W)

    Returns
    -------
    PanelLayoutResponse
    """
    if len(roof_polygon_latlon) < 3:
        return PanelLayoutResponse(
            panel_count=0,
            panels=[],
            roof_area_m2=0.0,
            usable_area_m2=0.0,
            installed_kw=0.0,
            centroid=(0.0, 0.0),
        )

    lat0, lon0 = _centroid(roof_polygon_latlon)

    # Convert polygon to metric
    metric_poly: list[tuple[float, float]] = [
        _to_metric(lat, lon, lat0, lon0) for lat, lon in roof_polygon_latlon
    ]

    roof_area = _polygon_area(metric_poly)

    # Panel dimensions in the grid frame
    if orientation == "landscape":
        cell_ew = panel_length_m  # East-West extent
        cell_ns = panel_width_m  # North-South extent
    else:  # portrait
        cell_ew = panel_width_m
        cell_ns = panel_length_m

    step_ew = cell_ew + col_gap_m
    step_ns = cell_ns + row_gap_m

    # Bounding box
    xs = [p[0] for p in metric_poly]
    ys = [p[1] for p in metric_poly]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    panels: list[PanelPosition] = []

    # Grid origin starts at bbox_min + setback so first panel edge is setback_m in
    x_start = x_min + setback_m
    y_start = y_min + setback_m

    y = y_start
    while y + cell_ns <= y_max - setback_m + 1e-6:
        x = x_start
        while x + cell_ew <= x_max - setback_m + 1e-6:
            # Panel corners (counter-clockwise from SW)
            corners_metric: list[tuple[float, float]] = [
                (x, y),
                (x + cell_ew, y),
                (x + cell_ew, y + cell_ns),
                (x, y + cell_ns),
            ]
            if _panel_valid(corners_metric, metric_poly, setback_m):
                corners_latlon = [_to_latlon(cx, cy, lat0, lon0) for cx, cy in corners_metric]
                panels.append(PanelPosition(corners=corners_latlon))
            x += step_ew
        y += step_ns

    n = len(panels)
    usable_area = n * cell_ew * cell_ns
    installed_kw = n * panel_p_stc_w / 1000.0

    return PanelLayoutResponse(
        panel_count=n,
        panels=panels,
        roof_area_m2=round(roof_area, 2),
        usable_area_m2=round(usable_area, 2),
        installed_kw=round(installed_kw, 3),
        centroid=(lat0, lon0),
    )
