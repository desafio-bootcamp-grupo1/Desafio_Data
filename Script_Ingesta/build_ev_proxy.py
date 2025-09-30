from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import pandas as pd

def find_repo_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    markers = {".git", "ocm_ingest.py", "config.py"}
    p = start.resolve()
    for _ in range(10):
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        return Path(top)
    except Exception:
        return start.resolve()

REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PVPC_2025_DEFAULT = {
    "2025-01": 0.1599,
    "2025-02": 0.1680,
    "2025-03": 0.1233,
    "2025-04": 0.1084,
    "2025-05": 0.1130,
    "2025-06": 0.1339,
    "2025-07": 0.1370,
    "2025-08": 0.1331,
    "2025-09": 0.1393,  
}

def load_pvpc(year: int, pvpc_file: str | None) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas: month (YYYY-MM), pvpc_eur_kWh, pvpc_index (pendiente de normalizar).
    Si pvpc_file es None y year==2025, usa PVPC_2025_DEFAULT.
    """
    if pvpc_file:
        fp = Path(pvpc_file)
        if not fp.exists():
            raise SystemExit(f"[ERR] No existe PVPC file: {fp}")
        pvpc = pd.read_csv(fp)
        if not {"month", "pvpc_eur_kWh"}.issubset(pvpc.columns):
            raise SystemExit("[ERR] pvpc_file debe tener columnas: month, pvpc_eur_kWh")
    else:
        if year != 2025:
            raise SystemExit("[ERR] Para años != 2025 especifica --pvpc-file con month,pvpc_eur_kWh")
        pvpc = (pd.Series(PVPC_2025_DEFAULT, name="pvpc_eur_kWh")
                  .rename_axis("month").reset_index())
    pvpc["month"] = pd.to_datetime(pvpc["month"], format="%Y-%m")
    pvpc = pvpc.sort_values("month").reset_index(drop=True)
    return pvpc

def main():
    ap = argparse.ArgumentParser(description="Genera histórico proxy EV por CCAA usando índice PVPC")
    ap.add_argument("--year", type=int, required=True, help="Año a reconstruir (YYYY), p.ej. 2025")
    ap.add_argument("--base-month", type=str, required=True, help="Mes base (YYYY-MM) que ya tienes como CSV mensual")
    ap.add_argument("--metric", choices=["median", "mean"], default="median", help="Columna de baseline a usar")
    ap.add_argument("--pvpc-file", type=str, default=None, help="CSV con month,pvpc_eur_kWh (si no 2025)")
    ap.add_argument("--long", action="store_true", help="Además del wide, guardar CSV long")
    args = ap.parse_args()

    year = args.year
    base_month = args.base_month
    metric = args.metric

    pvpc = load_pvpc(year, args.pvpc_file)
    pvpc = pvpc[pvpc["month"].dt.year == year].copy()
    if pvpc.empty:
        raise SystemExit(f"[ERR] PVPC no tiene meses para {year}")
    try:
        base_val = pvpc.loc[pvpc["month"] == pd.Timestamp(base_month), "pvpc_eur_kWh"].iloc[0]
    except IndexError:
        raise SystemExit(f"[ERR] El mes base {base_month} no está en PVPC. Revisa pvpc_file o el dict por defecto.")
    pvpc["pvpc_index"] = pvpc["pvpc_eur_kWh"] / base_val

    agg_path = DATA_DIR / f"ocm_agg_{base_month}.csv"
    if not agg_path.exists():
        raise SystemExit(f"[ERR] No encuentro {agg_path}. Genera el mensual con ocm_ingest.py --month {base_month}")
    df = pd.read_csv(agg_path)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    val_col = "median_eur_kWh" if metric == "median" else "mean_eur_kWh"
    needed = {"ccaa", "band", val_col}
    if not needed.issubset(df.columns):
        raise SystemExit(f"[ERR] {agg_path.name} debe contener columnas {needed}")

    baseline = (
        df[df["month"] == pd.Timestamp(base_month)]
          .pivot_table(index="ccaa", columns="band", values=val_col)
          .reset_index()
    )
    for col in ["AC", "DC"]:
        if col not in baseline.columns:
            baseline[col] = pd.NA

    hist_proxy = (
        baseline.assign(key=1)
                .merge(pvpc[["month","pvpc_index"]].assign(key=1), on="key")
                .drop(columns="key")
                .sort_values(["ccaa", "month"])
                .reset_index(drop=True)
    )
    for col in ["AC", "DC"]:
        hist_proxy[col] = hist_proxy[col] * hist_proxy["pvpc_index"]

    out_wide = hist_proxy.copy()
    out_wide["month"] = out_wide["month"].dt.strftime("%Y-%m")
    out_wide = out_wide[["month", "ccaa", "AC", "DC"]]
    out_wide_path = DATA_DIR / f"ocm_proxy_{year}_wide.csv"
    out_wide.to_csv(out_wide_path, index=False, encoding="utf-8")
    print(f"[OK] Guardado: {out_wide_path.relative_to(REPO_ROOT)}")

    if args.long:
        out_long = []
        for band in ["AC", "DC"]:
            tmp = out_wide[["month", "ccaa", band]].rename(columns={band: "eur_kWh"}).copy()
            tmp["band"] = band
            out_long.append(tmp)
        out_long = pd.concat(out_long, ignore_index=True)[["month","ccaa","band","eur_kWh"]]
        out_long_path = DATA_DIR / f"ocm_proxy_{year}_long.csv"
        out_long.to_csv(out_long_path, index=False, encoding="utf-8")
        print(f"[OK] Guardado: {out_long_path.relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    main()