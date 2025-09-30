import os, re, requests, argparse
from dotenv import load_dotenv
from config import CCAA_BOUNDING_BOXES
from config_provincias import PROVINCIAS_BOUNDING_BOXES  # <- añade esto

API_URL = "https://api.openchargemap.io/v3/poi"

def usagecost_to_eur_per_kwh(text: str):
    if not text:
        return None
    s = re.sub(r"\s+", " ", text)
    m = re.search(r"([0-9]+[.,]?[0-9]*)\s*(?:€|EUR)\s*/?\s*kWh", s, flags=re.I)
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    return val if 0.05 <= val <= 2.0 else None

def main():
    parser = argparse.ArgumentParser(description="Test de normalización de UsageCost por CCAA o provincia")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--ccaa", type=str, help="Nombre exacto de la CCAA (config.py)")
    g.add_argument("--provincia", type=str, help="Nombre exacto de la provincia (config_provincias.py)")
    parser.add_argument("--n", type=int, default=10, help="Número de POIs a mostrar")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OCM_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OCM_API_KEY en .env")

    if args.ccaa:
        if args.ccaa not in CCAA_BOUNDING_BOXES:
            print(f"CCAA '{args.ccaa}' no encontrada en config.py"); return
        name = args.ccaa
        bbox = CCAA_BOUNDING_BOXES[name]
        scope = "CCAA"
    else:
        if args.provincia not in PROVINCIAS_BOUNDING_BOXES:
            print(f"Provincia '{args.provincia}' no encontrada en config_provincias.py"); return
        name = args.provincia
        bbox = PROVINCIAS_BOUNDING_BOXES[name]
        scope = "provincia"

    params = {
        "output": "json",
        "countrycode": "ES",
        "maxresults": args.n,
        "boundingbox": f"({bbox['north']},{bbox['west']}),({bbox['south']},{bbox['east']})",
        "compact": "true",
        "verbose": "false",
        "key": api_key,
    }

    print(f"\n=== {scope.upper()}: {name} ===")
    r = requests.get(API_URL, params=params, headers={"X-API-Key": api_key}, timeout=60)
    r.raise_for_status()
    pois = r.json()

    if not pois:
        print("Sin resultados"); return

    for poi in pois:
        uc = poi.get("UsageCost")
        norm = usagecost_to_eur_per_kwh(uc)
        print(f"Original: {uc!r} -> Normalizado: {norm}")

if __name__ == "__main__":
    main()
