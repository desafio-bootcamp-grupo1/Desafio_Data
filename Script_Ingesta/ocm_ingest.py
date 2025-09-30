import os, re, json, time, argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any, List

import requests
import pandas as pd
from dotenv import load_dotenv

from config_provincias import PROVINCIAS_BOUNDING_BOXES

API_URL = "https://api.openchargemap.io/v3/poi"

def slug(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
         .replace("*", "_")
         .replace("?", "_")
         .replace('"', "_")
         .replace("<", "_")
         .replace(">", "_")
         .replace("|", "_")
    )

def ensure_dirs():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/qc").mkdir(parents=True, exist_ok=True)

def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OCM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("No se encontró OCM_API_KEY en .env")
    return api_key

def usagecost_to_eur_per_kwh(text: str, treat_free_as_zero: bool = False) -> Tuple[Any, str]:
    if text is None:
        return None, "empty"
    s = re.sub(r"\s+", " ", str(text).strip(), flags=re.UNICODE)
    if treat_free_as_zero and re.search(r"\b(free|gratis)\b", s, flags=re.I):
        return 0.0, "free->0"
    pattern = r"([0-9]+[.,]?[0-9]*)\s*(?:€|EUR)\s*/?\s*kW?h"
    matches = re.findall(pattern, s, flags=re.I)
    if not matches:
        m2 = re.search(r"([0-9]+[.,]?[0-9]*)\s*(?:€|EUR).*kW?h", s, flags=re.I)
        if m2:
            matches = [m2.group(1)]
        else:
            if re.search(r"(€|EUR)\s*/\s*(min|minute)", s, flags=re.I):
                return None, "per_minute"
            if re.search(r"(€|EUR)\s*/\s*(session|sesión)", s, flags=re.I):
                return None, "per_session"
            if re.search(r"\b(free|gratis)\b", s, flags=re.I):
                return None, "free"
            return None, "no_kwh"
    try:
        val = float(matches[0].replace(",", "."))
    except ValueError:
        return None, "parse_error"
    if not (0.05 <= val <= 2.0):
        return None, "out_of_range"
    return val, "ok"

def band_from_power_kw(power_kw, level_id=None):
    try:
        p = float(power_kw) if power_kw is not None else None
        if p is None:
            return None
        return "AC" if p <= 22 else "DC"
    except Exception:
        return None

def make_session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def fetch_bbox(bbox: Dict[str, float], api_key: str, maxresults=10000, sleep_s=1.0) -> List[Dict]:
    params = {
        "output": "json",
        "countrycode": "ES",
        "boundingbox": f"({bbox['north']},{bbox['west']}),({bbox['south']},{bbox['east']})",
        "maxresults": str(maxresults),
        "compact": "true",
        "verbose": "false",
        "key": api_key,
    }
    session = make_session()
    headers = {"X-API-Key": api_key}
    r = session.get(API_URL, params=params, headers=headers, timeout=60)
    if r.status_code in (429, 503):
        time.sleep(5)
        r = session.get(API_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    time.sleep(sleep_s)
    return r.json()

def normalize_rows(pois_json: List[Dict], prov_name: str, month_label: str, treat_free_as_zero=False) -> List[Dict]:
    rows = []
    qc_rows = []
    for poi in pois_json:
        usage_cost = poi.get("UsageCost")
        lat = poi.get("AddressInfo", {}).get("Latitude")
        lon = poi.get("AddressInfo", {}).get("Longitude")
        operator = (poi.get("OperatorInfo") or {}).get("Title")
        connections = poi.get("Connections") or []
        for c in connections:
            power = c.get("PowerKW")
            level_id = c.get("LevelID")
            band = band_from_power_kw(power, level_id)
            val, why = usagecost_to_eur_per_kwh(usage_cost, treat_free_as_zero=treat_free_as_zero)
            qc_rows.append({
                "month": month_label, "provincia": prov_name, "operator": operator,
                "power_kW": power, "band": band, "usagecost_original": usage_cost,
                "parsed": val, "reason": why
            })
            if (val is not None) and (band is not None):
                rows.append({
                    "month": month_label, "provincia": prov_name, "band": band,
                    "eur_kWh": val, "lat": lat, "lon": lon,
                    "operator": operator, "power_kW": power
                })
    qc_path = Path(f"data/qc/qc_{slug(prov_name)}_{month_label}.csv")
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8")
    return rows

def aggregate_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month","provincia","band","median_eur_kWh","mean_eur_kWh","n"])
    agg = (
        df.groupby(["month","provincia","band"])["eur_kWh"]
          .agg(median="median", mean="mean", n="count")
          .reset_index()
          .rename(columns={"median": "median_eur_kWh", "mean": "mean_eur_kWh"})
    )
    return agg

def month_range_for_year(year: int):
    now = datetime.now(timezone.utc)
    end_month = 12 if year < now.year else now.month
    return [f"{year}-{m:02d}" for m in range(1, end_month+1)]

def process_one_month(month_label: str, only_prov, api_key: str, maxresults: int, overwrite: bool, treat_free_as_zero=False):
    ensure_dirs()
    all_rows = []
    prov_items = PROVINCIAS_BOUNDING_BOXES.items()
    if only_prov:
        subset = set(only_prov)
        prov_items = [(k, v) for k, v in PROVINCIAS_BOUNDING_BOXES.items() if k in subset]

    for prov_name, bbox in prov_items:
        raw_path = Path(f"data/raw/ocm_{slug(prov_name)}_{month_label}.json")
        if raw_path.exists() and not overwrite:
            print(f"[SKIP] {prov_name} {month_label}: raw ya existe, leyendo…")
            pois = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            print(f"[INFO] Descargando {prov_name} {month_label} …")
            pois = fetch_bbox(bbox, api_key=api_key, maxresults=maxresults)
            raw_path.write_text(json.dumps(pois, ensure_ascii=False, indent=2), encoding="utf-8")

        rows = normalize_rows(pois, prov_name, month_label, treat_free_as_zero=treat_free_as_zero)
        print(f"[INFO] {prov_name} {month_label}: filas con precio parseado = {len(rows)}")
        all_rows.extend(rows)

    df_rows = pd.DataFrame(all_rows)
    df_agg = aggregate_month(df_rows)

    rows_path = Path(f"data/processed/ocm_rows_{month_label}.csv")
    agg_path = Path(f"data/processed/ocm_agg_{month_label}.csv")
    df_rows.to_csv(rows_path, index=False, encoding="utf-8")
    df_agg.to_csv(agg_path, index=False, encoding="utf-8")
    print(f"[OK] Mes {month_label} -> rows: {rows_path.name}, agg: {agg_path.name}")
    return df_rows, df_agg

def combine_year(year: int):
    ensure_dirs()
    files = sorted(Path("data/processed").glob(f"ocm_agg_{year}-*.csv"))
    if not files:
        print(f"[WARN] No hay agregados mensuales para {year}")
        return None
    parts = [pd.read_csv(fp) for fp in files]
    annual = (
        pd.concat(parts, ignore_index=True)
          .sort_values(["month","provincia","band"])
          .reset_index(drop=True)
    )
    out = Path(f"data/processed/ocm_agg_{year}.csv")
    annual.to_csv(out, index=False, encoding="utf-8")
    print(f"[OK] Agregado anual -> {out.name}")
    return annual

def main():
    parser = argparse.ArgumentParser(description="Ingesta OCM por provincias y agregado mensual/anual")
    parser.add_argument("--month", type=str, help="YYYY-MM. Solo ese mes.")
    parser.add_argument("--year", type=int, help="YYYY. Todos los meses hasta el actual.")
    parser.add_argument("--provincias", "--only-provincias", "--only",
                        dest="only_prov", type=str, nargs="*",
                        metavar="PROVINCIA",
                        help="Lista de provincias tal y como aparecen en config_provincias.py")
    parser.add_argument("--overwrite", action="store_true", help="Forzar descarga si ya existe el raw del mes.")
    parser.add_argument("--maxresults", type=int, default=10000)
    parser.add_argument("--free0", action="store_true", help="Tratar 'free/gratis' como 0.0")
    args = parser.parse_args()

    api_key = load_api_key()
    if args.month and args.year:
        raise SystemExit("Usa --month O --year, no ambos.")

    if args.month:
        process_one_month(args.month, args.only_prov, api_key, args.maxresults, args.overwrite, treat_free_as_zero=args.free0)
        return

    if args.year:
        for m in month_range_for_year(args.year):
            print(f"\n===== Procesando {m} =====")
            process_one_month(m, args.only_prov, api_key, args.maxresults, args.overwrite, treat_free_as_zero=args.free0)
        combine_year(args.year)
        return

    month_label = datetime.utcnow().strftime("%Y-%m")
    process_one_month(month_label, args.only_prov, api_key, args.maxresults, args.overwrite, treat_free_as_zero=args.free0)

if __name__ == "__main__":
    main()