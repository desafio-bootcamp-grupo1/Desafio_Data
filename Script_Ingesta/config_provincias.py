import json
from pathlib import Path

def _ensure_dirs():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

_CACHE = Path("data/processed/provincias_bbox.json")

def _load_cached():
    if _CACHE.exists():
        with _CACHE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_cache(obj):
    with _CACHE.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _build_from_opendatasoft():
    import requests
    from shapely.geometry import shape

    url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "georef-spain-provincia/exports/geojson?lang=es&timezone=UTC"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    gj = r.json()
    feats = gj.get("features") or []

    boxes = {}
    for f in feats:
        props = f.get("properties", {})
        geom = f.get("geometry")
        if not geom:
            continue
        prov_name = (
            props.get("prov_name") or props.get("name") or
            props.get("provincia") or props.get("label_es") or props.get("label")
        )
        if not prov_name:
            continue
        g = shape(geom)  # EPSG:4326
        minx, miny, maxx, maxy = g.bounds
        boxes[str(prov_name)] = {
            "north": round(float(maxy), 6),
            "south": round(float(miny), 6),
            "west":  round(float(minx), 6),
            "east":  round(float(maxx), 6),
        }

    renames = {
        "Araba/Álava": "Álava",
        "Araba": "Álava",
        "Girona": "Girona",
        "Lleida": "Lleida",
        "Castelló": "Castellón",
        "València": "Valencia",
        "Illes Balears": "Illes Balears",
        "A Coruña": "A Coruña",
        "Ourense": "Ourense",
        "Bizkaia": "Bizkaia",
        "Gipuzkoa": "Gipuzkoa",
    }
    for k_old, k_new in list(renames.items()):
        if k_old in boxes:
            boxes[k_new] = boxes.pop(k_old)

    for k in list(boxes.keys()):
        if "Ceuta" in k and k != "Ceuta":
            boxes["Ceuta"] = boxes.pop(k)
        if "Melilla" in k and k != "Melilla":
            boxes["Melilla"] = boxes.pop(k)

    return boxes

def refresh_cache() -> dict:
    _ensure_dirs()
    boxes = _build_from_opendatasoft()
    _save_cache(boxes)
    return boxes

_ensure_dirs()
_cached = _load_cached()
if _cached:
    PROVINCIAS_BOUNDING_BOXES = _cached
else:
    PROVINCIAS_BOUNDING_BOXES = refresh_cache()
