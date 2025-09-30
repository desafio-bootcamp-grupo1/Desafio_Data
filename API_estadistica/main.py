# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import pandas as pd
import numpy as np
import os, re, json, ast, traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Dict, Any

# ==============================
# Cargar .env (del mismo dir que main.py) y configurar Mongo
# ==============================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
MONGO_URI = os.getenv("DB_URL")
DB_NAME = os.getenv("DB_NAME", "Prueba1")
COL_FUEL = os.getenv("COL_FUEL", "Combustible")
COL_EV = os.getenv("COL_EV", "electrico")
COL_TOLL = os.getenv("COL_TOLL", "Peaje")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll_fuel = db[COL_FUEL]
coll_ev = db[COL_EV]
coll_toll = db[COL_TOLL]

# ==============================
# FastAPI + CORS
# ==============================
app = FastAPI(title="API KPIs: Combustible, EV, Peaje")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Utilidades
# ==============================
dias_map = {
    "Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles","Thursday":"Jueves",
    "Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"
}

def _to_num_eur(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan
    s = str(x).strip().replace("€","").replace("\xa0"," ").strip()
    s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", s)
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except:
        return np.nan

def _fix_jsonish(s):
    s = re.sub(r"(?<!\\)'", '"', str(s))
    return s.replace("None","null").replace("True","true").replace("False","false")

def parse_lineas(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = _fix_jsonish(x)
    for fn in (json.loads, lambda t: ast.literal_eval(t)):
        try:
            out = fn(s)
            if isinstance(out, dict):
                return [out]
            if isinstance(out, list):
                return out
        except Exception:
            continue
    return []

def _ser(d: dict):
    out = {}
    for k, v in d.items():
        out[k] = v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v
    return out

def _filter_payload(payload: dict, section: Optional[str], fields: Optional[List[str]]) -> dict:
    if not payload:
        return {}
    if section in ("empresa", "usuario"):
        payload = {section: payload.get(section, {})}
    if fields:
        fset = set(fields)
        for sec in list(payload.keys()):
            sec_data = payload.get(sec) or {}
            payload[sec] = {k: v for k, v in sec_data.items() if k in fset}
    return payload

# ==============================
# Filtros Mongo por fechas
# ==============================
def _build_filter_fechas_cabecera(start_date: str|None, end_date: str|None):
    f = {}
    if start_date or end_date:
        rango = {}
        if start_date:
            rango["$gte"] = start_date
        if end_date:
            rango["$lte"] = end_date
        f["fechaEmision"] = rango
    return f

def _build_filter_fechas_toll(start_date: str|None, end_date: str|None):
    f = {}
    if start_date or end_date:
        rango = {}
        if start_date:
            rango["$gte"] = start_date
        if end_date:
            dt = datetime.fromisoformat(end_date) + timedelta(days=1) - timedelta(seconds=1)
            rango["$lte"] = dt.isoformat()
        f["fechaHora"] = rango
    return f

def load_df_mongo(collection, filtros: dict) -> pd.DataFrame:
    try:
        data = list(collection.find(filtros))
    except Exception as e:
        print("Error Mongo:", repr(e))
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

# ==============================
# Normalización temporal
# ==============================
def add_time_cols_fuel_ev(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "fechaEmision" not in df.columns:
        df["fechaEmision"] = pd.NaT
    df["fechaEmision"] = pd.to_datetime(df["fechaEmision"], errors="coerce")
    df["horaEmision"] = df.get("horaEmision", "00:00:00")
    df["dt"] = pd.to_datetime(
        df["fechaEmision"].astype(str) + " " + df["horaEmision"].fillna("00:00:00"),
        errors="coerce"
    )
    df["mes"] = df["dt"].dt.to_period("M").astype(str)
    df["hora"] = df["dt"].dt.hour
    df["dia_semana"] = df["dt"].dt.day_name().map(dias_map)
    return df

def add_time_cols_toll(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "fechaHora" not in df.columns:
        df["fechaHora"] = pd.NaT
    df["fechaHora"] = pd.to_datetime(df["fechaHora"], errors="coerce")
    df["mes"] = df["fechaHora"].dt.to_period("M").astype(str)
    df["hora"] = df["fechaHora"].dt.hour
    df["dia_semana"] = df["fechaHora"].dt.day_name().map(dias_map)
    df["is_weekend"] = df["dia_semana"].isin(["Sábado","Domingo"])
    return df

# ==============================
# Explode líneas
# ==============================
def explode_fuel_lines(df_fuel: pd.DataFrame) -> pd.DataFrame:
    if df_fuel.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df_fuel.iterrows():
        lineas = r.get("lineas_parsed") or parse_lineas(r.get("lineas"))
        for li in (lineas or []):
            litros = li.get("litros")
            ppu = li.get("precioPorLitro") or li.get("precio_unitario") or li.get("precio")
            importe_li = li.get("importe")
            try:
                l = float(litros) if litros is not None else np.nan
            except:
                l = np.nan
            try:
                p = float(ppu) if ppu is not None else np.nan
            except:
                p = np.nan
            if pd.isna(p) and importe_li is not None and pd.notna(l) and l > 0:
                try:
                    p = float(importe_li)/l
                except:
                    pass
            importe = (l*p) if (pd.notna(l) and pd.notna(p)) else np.nan
            rows.append({
                "idTicket": r.get("idTicket"),
                "idUsuario": r.get("idUsuario"),
                "empresaTransporte": r.get("empresaNombre"),
                "dt": r.get("dt"),
                "mes": r.get("mes"),
                "hora": r.get("hora"),
                "dia_semana": r.get("dia_semana"),
                "producto": li.get("producto"),
                "litros": l,
                "precio_litro": p,
                "importe_linea": importe,
                "baseImponible": r.get("baseImponible"),
                "iva": r.get("iva"),
                "total": r.get("total"),
                "metodoPago": r.get("metodoPago"),
            })
    df_lines = pd.DataFrame(rows)
    for c in ["litros","precio_litro","importe_linea","baseImponible","iva","total"]:
        if c in df_lines.columns:
            df_lines[c] = pd.to_numeric(df_lines[c], errors="coerce")
    return df_lines

def explode_ev_lines(df_ev: pd.DataFrame) -> pd.DataFrame:
    if df_ev.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df_ev.iterrows():
        lineas = r.get("lineas_parsed") or parse_lineas(r.get("lineas"))
        for li in (lineas or []):
            kwh_raw = li.get("kwh") or li.get("energia") or li.get("energia_kwh")
            ppu_raw = li.get("precio_kwh") or li.get("precioUnitario") or li.get("precio_unitario") or li.get("precio")
            importe_li = li.get("importe") or li.get("importeLinea") or li.get("importe_linea")
            try:
                k = float(kwh_raw) if kwh_raw is not None else np.nan
            except:
                k = np.nan
            try:
                p = float(ppu_raw) if ppu_raw is not None else np.nan
            except:
                p = np.nan
            if pd.isna(p) and (importe_li is not None) and pd.notna(k) and k > 0:
                try:
                    p = float(importe_li) / k
                except:
                    pass
            importe = (k * p) if (pd.notna(k) and pd.notna(p)) else np.nan
            rows.append({
                "idTicket": r.get("idTicket"),
                "idUsuario": r.get("idUsuario"),
                "empresaTransporte": r.get("empresaNombre"),
                "dt": r.get("dt"),
                "mes": r.get("mes"),
                "hora": r.get("hora"),
                "dia_semana": r.get("dia_semana"),
                "producto": li.get("producto"),
                "kwh": k,
                "precio_kwh": p,
                "importe_linea": importe,
                "baseImponible": r.get("baseImponible"),
                "iva": r.get("iva"),
                "total": r.get("total"),
                "metodoPago": r.get("metodoPago"),
                "tipoCorriente": li.get("tipoCorriente") or li.get("tarifa") or r.get("estacion_tarifa"),
                "potenciaKW": li.get("potenciaKW") or li.get("potenciaMaxKW") or li.get("power_kw") or r.get("estacion_potencia_max_kw"),
            })
    df_lines = pd.DataFrame(rows)
    for c in ["kwh","precio_kwh","importe_linea","baseImponible","iva","total","potenciaKW"]:
        if c in df_lines.columns:
            df_lines[c] = pd.to_numeric(df_lines[c], errors="coerce")
    return df_lines

# ==============================
# KPIs helpers
# ==============================
def _dias_mediana(series_dt):
    s = pd.to_datetime(pd.Series(series_dt).dropna()).sort_values().unique()
    if len(s) < 2:
        return np.nan
    d = np.diff(s).astype("timedelta64[D]").astype(float)
    return float(np.median(d))

# ---------- KPIs Combustible ----------
def kpis_usuario_fuel(df_tickets: pd.DataFrame, df_lines: pd.DataFrame):
    if df_tickets.empty or df_lines.empty:
        return {}

    gasto_usuario_mes = df_tickets.groupby(["idUsuario","mes"])["total"].sum().reset_index()
    tickets_usuario_mes = df_tickets.groupby(["idUsuario","mes"])["idTicket"].nunique().reset_index(name="tickets")
    litros_usuario_mes = df_lines.groupby(["idUsuario","mes"])["litros"].sum().reset_index()

    litros_ticket_usuario = df_lines.groupby(["idUsuario","idTicket"])["litros"].sum().reset_index()
    litros_medio_ticket_usuario = litros_ticket_usuario.groupby("idUsuario")["litros"].mean().reset_index(name="litros_medio_ticket")

    precio_usuario_marca = (
        df_lines.assign(w=df_lines["litros"])
        .groupby(["idUsuario","empresaTransporte"])
        .apply(lambda x: (x["precio_litro"].mul(x["w"]).sum())/x["w"].sum())
        .rename("eur_l").reset_index()
    )

    metodos_usuario = df_tickets.groupby(["idUsuario","metodoPago"])["idTicket"].nunique().reset_index(name="tickets")
    metodos_usuario_mes = df_tickets.groupby(["idUsuario","mes","metodoPago"])["idTicket"].nunique().reset_index(name="tickets")
    metodos_usuario_mes["pct"] = 100 * metodos_usuario_mes["tickets"] / metodos_usuario_mes.groupby(["idUsuario","mes"])["tickets"].transform("sum")

    dias_mediana = df_tickets.groupby("idUsuario")["dt"].apply(_dias_mediana).reset_index(name="dias_mediana")

    anomalias_usuario = (
        df_lines[(df_lines["precio_litro"] < 0.8) | (df_lines["precio_litro"] > 3.0)]
        .groupby("idUsuario")["idTicket"].nunique().reset_index(name="tickets_anomalos")
    )

    gasto_dia_semana = df_tickets.groupby(["idUsuario","dia_semana"])["total"].sum().reset_index()

    return {
        "gasto_usuario_mes": gasto_usuario_mes,
        "tickets_usuario_mes": tickets_usuario_mes,
        "litros_usuario_mes": litros_usuario_mes,
        "litros_medio_ticket_usuario": litros_medio_ticket_usuario,
        "precio_usuario_marca": precio_usuario_marca,
        "metodos_usuario": metodos_usuario,
        "metodos_usuario_mes": metodos_usuario_mes,
        "dias_mediana": dias_mediana,
        "anomalias_usuario": anomalias_usuario,
        "gasto_dia_semana": gasto_dia_semana,
    }

# ---------- KPIs Combustible (EMPRESA con desglose por usuario/mes) ----------
def kpis_empresa_fuel(df_tickets: pd.DataFrame, df_lines: pd.DataFrame):
    if df_tickets.empty or df_lines.empty:
        return {}

    gasto_mes_emp = (
        df_tickets.groupby(["empresaNombre","mes"])["total"]
        .sum().reset_index()
        .rename(columns={"empresaNombre":"empresa"})
    )
    gasto_total_emp = (
        df_tickets.groupby("empresaNombre")["total"]
        .sum().reset_index()
        .rename(columns={"empresaNombre":"empresa","total":"gasto_total_periodo"})
    )
    vehiculos_emp = (
        df_tickets.groupby("empresaNombre")["idUsuario"]
        .nunique().reset_index()
        .rename(columns={"empresaNombre":"empresa","idUsuario":"num_vehiculos"})
    )
    gasto_medio_veh = (
        df_tickets.groupby(["empresaNombre","idUsuario"])["total"]
        .sum().groupby("empresaNombre").mean().reset_index()
        .rename(columns={"empresaNombre":"empresa","total":"gasto_medio_por_vehiculo"})
    )
    ranking_veh = (
        df_tickets.groupby(["empresaNombre","idUsuario"])["total"]
        .sum().reset_index()
        .rename(columns={"empresaNombre":"empresa","total":"gasto_usuario"})
    )

    litros_mes_emp = (
        df_lines.groupby(["empresaTransporte","mes"])["litros"]
        .sum().reset_index().rename(columns={"empresaTransporte":"empresa"})
    )
    litros_veh_emp = (
        df_lines.groupby(["empresaTransporte","idUsuario"])["litros"]
        .sum().reset_index().rename(columns={"empresaTransporte":"empresa"})
    )
    precio_global_emp = (
        df_lines.assign(w=df_lines["litros"])
        .groupby("empresaTransporte")
        .apply(lambda x: (x["precio_litro"].mul(x["w"]).sum())/x["w"].sum())
        .reset_index(name="eur_l")
        .rename(columns={"empresaTransporte":"empresa"})
    )
    gasto_emp_user_mes = (
        df_tickets.groupby(["empresaNombre","idUsuario","mes"])["total"]
        .sum().reset_index()
        .rename(columns={"empresaNombre":"empresa"})
    )
    tickets_emp_user_mes = (
        df_tickets.groupby(["empresaNombre","idUsuario","mes"])["idTicket"]
        .nunique().reset_index(name="tickets")
        .rename(columns={"empresaNombre":"empresa"})
    )
    litros_emp_user_mes = (
        df_lines.groupby(["empresaTransporte","idUsuario","mes"])["litros"]
        .sum().reset_index()
        .rename(columns={"empresaTransporte":"empresa"})
    )
    precio_emp_user = (
        df_lines.assign(w=df_lines["litros"])
        .groupby(["empresaTransporte","idUsuario"])
        .apply(lambda x: (x["precio_litro"].mul(x["w"]).sum())/x["w"].sum())
        .reset_index(name="eur_l")
        .rename(columns={"empresaTransporte":"empresa"})
    )

    return {
        "gasto_mes_emp": gasto_mes_emp,
        "gasto_total_emp": gasto_total_emp,
        "vehiculos_emp": vehiculos_emp,
        "gasto_medio_veh": gasto_medio_veh,
        "ranking_veh": ranking_veh,
        "litros_mes_emp": litros_mes_emp,
        "litros_veh_emp": litros_veh_emp,
        "precio_global_emp": precio_global_emp,
        "gasto_emp_user_mes": gasto_emp_user_mes,
        "tickets_emp_user_mes": tickets_emp_user_mes,
        "litros_emp_user_mes": litros_emp_user_mes,
        "precio_emp_user": precio_emp_user,
        "consumo_emp_user_mes": litros_emp_user_mes,
    }

# ---------- KPIs EV ----------
def kpis_usuario_ev(df_tickets: pd.DataFrame, df_lines: pd.DataFrame):
    if df_tickets.empty or df_lines.empty:
        return {}

    gasto_usuario_mes = df_tickets.groupby(["idUsuario","mes"])["total"].sum().reset_index()
    tickets_usuario_mes = df_tickets.groupby(["idUsuario","mes"])["idTicket"].nunique().reset_index(name="tickets")
    kwh_usuario_mes = df_lines.groupby(["idUsuario","mes"])["kwh"].sum().reset_index()

    kwh_ticket_usuario = df_lines.groupby(["idUsuario","idTicket"])["kwh"].sum().reset_index()
    kwh_medio_ticket_usuario = kwh_ticket_usuario.groupby("idUsuario")["kwh"].mean().reset_index(name="kwh_medio_ticket")

    precio_usuario_cpo = (
        df_lines.assign(w=df_lines["kwh"])
        .groupby(["idUsuario","empresaTransporte"])
        .apply(lambda x: (x["precio_kwh"].mul(x["w"]).sum())/x["w"].sum())
        .rename("eur_kwh").reset_index()
    )

    metodos_usuario = df_tickets.groupby(["idUsuario","metodoPago"])["idTicket"].nunique().reset_index(name="tickets")
    metodos_usuario_mes = df_tickets.groupby(["idUsuario","mes","metodoPago"])["idTicket"].nunique().reset_index(name="tickets")
    metodos_usuario_mes["pct"] = 100 * metodos_usuario_mes["tickets"] / metodos_usuario_mes.groupby(["idUsuario","mes"])["tickets"].transform("sum")

    dias_mediana = df_tickets.groupby("idUsuario")["dt"].apply(_dias_mediana).reset_index(name="dias_mediana")

    anomalias_usuario = (
        df_lines[(df_lines["precio_kwh"] < 0.20) | (df_lines["precio_kwh"] > 1.50)]
        .groupby("idUsuario")["idTicket"].nunique().reset_index(name="tickets_anomalos")
    )

    gasto_dia_semana = df_tickets.groupby(["idUsuario","dia_semana"])["total"].sum().reset_index()

    return {
        "gasto_usuario_mes": gasto_usuario_mes,
        "tickets_usuario_mes": tickets_usuario_mes,
        "kwh_usuario_mes": kwh_usuario_mes,
        "kwh_medio_ticket_usuario": kwh_medio_ticket_usuario,
        "precio_usuario_cpo": precio_usuario_cpo,
        "metodos_usuario": metodos_usuario,
        "metodos_usuario_mes": metodos_usuario_mes,
        "dias_mediana": dias_mediana,
        "anomalias_usuario": anomalias_usuario,
        "gasto_dia_semana": gasto_dia_semana,
    }

# ---------- KPIs EV (EMPRESA con desglose por usuario/mes) ----------
def kpis_empresa_ev(df_tickets: pd.DataFrame, df_lines: pd.DataFrame):
    if df_tickets.empty or df_lines.empty:
        return {}

    gasto_mes_emp = (
        df_tickets.groupby(["empresaNombre","mes"])["total"]
        .sum().reset_index().rename(columns={"empresaNombre":"empresa"})
    )
    gasto_total_emp = (
        df_tickets.groupby("empresaNombre")["total"]
        .sum().reset_index().rename(columns={"empresaNombre":"empresa","total":"gasto_total_periodo"})
    )
    vehiculos_emp = (
        df_tickets.groupby("empresaNombre")["idUsuario"]
        .nunique().reset_index().rename(columns={"empresaNombre":"empresa","idUsuario":"num_vehiculos"})
    )
    gasto_medio_veh = (
        df_tickets.groupby(["empresaNombre","idUsuario"])["total"]
        .sum().groupby("empresaNombre").mean().reset_index()
        .rename(columns={"empresaNombre":"empresa","total":"gasto_medio_por_vehiculo"})
    )
    ranking_veh = (
        df_tickets.groupby(["empresaNombre","idUsuario"])["total"]
        .sum().reset_index().rename(columns={"empresaNombre":"empresa","total":"gasto_usuario"})
    )

    kwh_mes_emp = (
        df_lines.groupby(["empresaTransporte","mes"])["kwh"]
        .sum().reset_index().rename(columns={"empresaTransporte":"empresa"})
    )
    kwh_veh_emp = (
        df_lines.groupby(["empresaTransporte","idUsuario"])["kwh"]
        .sum().reset_index().rename(columns={"empresaTransporte":"empresa"})
    )
    precio_global_emp = (
        df_lines.assign(w=df_lines["kwh"])
        .groupby("empresaTransporte")
        .apply(lambda x: (x["precio_kwh"].mul(x["w"]).sum())/x["w"].sum())
        .reset_index(name="eur_kwh").rename(columns={"empresaTransporte":"empresa"})
    )
    gasto_emp_user_mes = (
        df_tickets.groupby(["empresaNombre","idUsuario","mes"])["total"]
        .sum().reset_index().rename(columns={"empresaNombre":"empresa"})
    )
    tickets_emp_user_mes = (
        df_tickets.groupby(["empresaNombre","idUsuario","mes"])["idTicket"]
        .nunique().reset_index(name="tickets").rename(columns={"empresaNombre":"empresa"})
    )
    kwh_emp_user_mes = (
        df_lines.groupby(["empresaTransporte","idUsuario","mes"])["kwh"]
        .sum().reset_index().rename(columns={"empresaTransporte":"empresa"})
    )
    precio_emp_user = (
        df_lines.assign(w=df_lines["kwh"])
        .groupby(["empresaTransporte","idUsuario"])
        .apply(lambda x: (x["precio_kwh"].mul(x["w"]).sum())/x["w"].sum())
        .reset_index(name="eur_kwh").rename(columns={"empresaTransporte":"empresa"})
    )

    return {
        "gasto_mes_emp": gasto_mes_emp,
        "gasto_total_emp": gasto_total_emp,
        "vehiculos_emp": vehiculos_emp,
        "gasto_medio_veh": gasto_medio_veh,
        "ranking_veh": ranking_veh,
        "kwh_mes_emp": kwh_mes_emp,
        "kwh_veh_emp": kwh_veh_emp,
        "precio_global_emp": precio_global_emp,
        "gasto_emp_user_mes": gasto_emp_user_mes,
        "tickets_emp_user_mes": tickets_emp_user_mes,
        "kwh_emp_user_mes": kwh_emp_user_mes,
        "precio_emp_user": precio_emp_user,
        "consumo_emp_user_mes": kwh_emp_user_mes,
    }

# ==============================
# Dominios y combinador ALL
# ==============================
VALID_DOMAINS = {"combustible", "ev", "peaje", "all"}
VALID_SECTIONS = {"empresa", "usuario"}

def _compute_all_empresa(start_date: str|None, end_date: str|None, empresa: str|None, idUsuario: str|None) -> dict:
    # Combustible
    filt_fuel = _build_filter_fechas_cabecera(start_date, end_date)
    if empresa:   filt_fuel["empresaNombre"] = empresa
    if idUsuario: filt_fuel["idUsuario"] = idUsuario
    df_fuel = load_df_mongo(coll_fuel, filt_fuel)
    if not df_fuel.empty:
        df_fuel = add_time_cols_fuel_ev(df_fuel)
        gasto_mes_fuel = (df_fuel.groupby(["empresaNombre","mes"])["total"].sum()
                          .reset_index().rename(columns={"empresaNombre":"empresa","total":"gasto"}))
        gasto_user_mes_fuel = (df_fuel.groupby(["empresaNombre","idUsuario","mes"])["total"].sum()
                               .reset_index().rename(columns={"empresaNombre":"empresa","total":"gasto"}))
        gasto_mes_fuel["domain"] = "combustible"
        gasto_user_mes_fuel["domain"] = "combustible"
    else:
        gasto_mes_fuel = pd.DataFrame(columns=["empresa","mes","gasto","domain"])
        gasto_user_mes_fuel = pd.DataFrame(columns=["empresa","idUsuario","mes","gasto","domain"])

    # EV
    filt_ev = _build_filter_fechas_cabecera(start_date, end_date)
    if empresa:   filt_ev["empresaNombre"] = empresa
    if idUsuario: filt_ev["idUsuario"] = idUsuario
    df_ev = load_df_mongo(coll_ev, filt_ev)
    if not df_ev.empty:
        df_ev = add_time_cols_fuel_ev(df_ev)
        gasto_mes_ev = (df_ev.groupby(["empresaNombre","mes"])["total"].sum()
                        .reset_index().rename(columns={"empresaNombre":"empresa","total":"gasto"}))
        gasto_user_mes_ev = (df_ev.groupby(["empresaNombre","idUsuario","mes"])["total"].sum()
                             .reset_index().rename(columns={"empresaNombre":"empresa","total":"gasto"}))
        gasto_mes_ev["domain"] = "ev"
        gasto_user_mes_ev["domain"] = "ev"
    else:
        gasto_mes_ev = pd.DataFrame(columns=["empresa","mes","gasto","domain"])
        gasto_user_mes_ev = pd.DataFrame(columns=["empresa","idUsuario","mes","gasto","domain"])

    # Peaje
    filt_toll = _build_filter_fechas_toll(start_date, end_date)
    if empresa:   filt_toll["empresaNombre"] = empresa
    if idUsuario: filt_toll["idUsuario"] = idUsuario
    df_toll = load_df_mongo(coll_toll, filt_toll)
    if not df_toll.empty:
        for col, default in [("empresaNombre","EMPRESA_UNICA"), ("fechaHora", pd.NaT)]:
            if col not in df_toll.columns: df_toll[col] = default
        cand_imp = [c for c in df_toll.columns if any(p in c.lower() for p in
                   ["importe","total","precio","coste","amount","valor","pago"])]
        if cand_imp:
            col = cand_imp[0]
            num = pd.to_numeric(df_toll[col], errors="coerce")
            if num.isna().any():
                parsed = df_toll[col].apply(_to_num_eur)
                num = num.where(num.notna(), parsed)
            df_toll["importe"] = num
        else:
            df_toll["importe"] = np.nan
        df_toll = add_time_cols_toll(df_toll)

        gasto_mes_toll = (df_toll.groupby(["empresaNombre","mes"])["importe"].sum()
                          .reset_index().rename(columns={"empresaNombre":"empresa","importe":"gasto"}))
        gasto_user_mes_toll = (df_toll.groupby(["empresaNombre","idUsuario","mes"])["importe"].sum()
                               .reset_index().rename(columns={"empresaNombre":"empresa","importe":"gasto"}))
        gasto_mes_toll["domain"] = "peaje"
        gasto_user_mes_toll["domain"] = "peaje"
    else:
        gasto_mes_toll = pd.DataFrame(columns=["empresa","mes","gasto","domain"])
        gasto_user_mes_toll = pd.DataFrame(columns=["empresa","idUsuario","mes","gasto","domain"])

    gasto_mes_emp_all = pd.concat([gasto_mes_fuel, gasto_mes_ev, gasto_mes_toll], ignore_index=True)
    gasto_emp_user_mes_all = pd.concat([gasto_user_mes_fuel, gasto_user_mes_ev, gasto_user_mes_toll], ignore_index=True)

    return {
        "empresa": {
            "gasto_mes_emp_all": gasto_mes_emp_all,
            "gasto_emp_user_mes_all": gasto_emp_user_mes_all,
        },
        "usuario": {}
    }

# ==============================
# Endpoints de diagnóstico
# ==============================
@app.get("/health")
def health():
    try:
        return {
            "db": DB_NAME,
            "collections": {
                COL_FUEL: coll_fuel.estimated_document_count(),
                COL_EV: coll_ev.estimated_document_count(),
                COL_TOLL: coll_toll.estimated_document_count()
            }
        }
    except Exception as e:
        return {"status": "degraded", "error": repr(e)}

@app.get("/debug/peek")
def debug_peek(n: int = 3):
    info = {}
    for name, coll in [(COL_FUEL, coll_fuel), (COL_EV, coll_ev), (COL_TOLL, coll_toll)]:
        df = load_df_mongo(coll, {})
        info[name] = {
            "docs": len(df),
            "cols": list(df.columns),
            "sample": df.head(n).to_dict(orient="records")
        }
    return info

@app.get("/debug/config")
def debug_config():
    try:
        return {
            "DB_URL_loaded": bool(MONGO_URI),
            "DB_NAME": DB_NAME,
            "COLLECTIONS": {
                "COL_FUEL": COL_FUEL,
                "COL_EV": COL_EV,
                "COL_TOLL": COL_TOLL,
            },
            "counts": {
                COL_FUEL: coll_fuel.estimated_document_count(),
                COL_EV: coll_ev.estimated_document_count(),
                COL_TOLL: coll_toll.estimated_document_count(),
            },
        }
    except Exception as e:
        return {"error": repr(e)}

@app.get("/debug/distinct")
def debug_distinct(collection: str, field: str, limit: int = 50):
    """
    Ejemplos:
    /debug/distinct?collection=Peaje&field=idUsuario
    /debug/distinct?collection=Combustible&field=empresaNombre
    /debug/distinct?collection=electrico&field=idUsuario
    """
    name_to_coll = {
        COL_FUEL: coll_fuel, "Combustible": coll_fuel,
        COL_EV: coll_ev, "electrico": coll_ev, "Electricos": coll_ev,
        COL_TOLL: coll_toll, "Peaje": coll_toll,
    }
    coll = name_to_coll.get(collection)
    if coll is None:
        return {"error": f"Colección desconocida: {collection}. Usa {list(name_to_coll.keys())}"}
    try:
        vals = coll.distinct(field)
        pipeline = [
            {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        top = list(coll.aggregate(pipeline))
        return {
            "collection": collection,
            "field": field,
            "distinct_count": len(vals),
            "sample_values": vals[:min(limit, 20)],
            "top_counts": top,
        }
    except Exception as e:
        return {"error": repr(e)}

@app.get("/debug/peaje_sample")
def debug_peaje_sample(n: int = 5):
    df = load_df_mongo(coll_toll, {})
    return {
        "docs": len(df),
        "cols": list(df.columns),
        "head": df.head(n).to_dict(orient="records"),
    }

@app.get("/debug/peaje_after")
def debug_peaje_after(
    start_date: str|None = None,
    end_date: str|None = None,
    empresa: str|None = None,
    idUsuario: str|None = None,
    n: int = 5
):
    filt = _build_filter_fechas_toll(start_date, end_date)
    if empresa:
        filt["empresaNombre"] = empresa
    if idUsuario:
        filt["idUsuario"] = idUsuario

    df = load_df_mongo(coll_toll, filt)
    if df.empty:
        return {"df_empty": True, "filtro": filt}

    for col, default in [
        ("idUsuario", "DESCONOCIDO"),
        ("empresaNombre","EMPRESA_UNICA"),
        ("autopista", "SIN_AUTOPISTA"),
        ("formaPago", "DESCONOCIDO"),
        ("fechaHora", pd.NaT),
    ]:
        if col not in df.columns:
            df[col] = default

    cand_imp = [c for c in df.columns if any(p in c.lower() for p in ["importe","total","precio","coste","amount","valor","pago"])]
    if cand_imp:
        col = cand_imp[0]
        imp_series = df[col]
        num = pd.to_numeric(imp_series, errors="coerce")
        if num.isna().any():
            parsed = imp_series.apply(_to_num_eur)
            num = num.where(num.notna(), parsed)
        df["importe"] = num
    else:
        df["importe"] = np.nan

    df = add_time_cols_toll(df)
    return {
        "filtro": filt,
        "cols": list(df.columns),
        "nulls_por_col": df.isna().sum().to_dict(),
        "head": df.head(n).to_dict(orient="records"),
        "importe_all_nan": bool(df["importe"].isna().all()),
        "fechaHora_all_nat": bool(pd.to_datetime(df["fechaHora"], errors="coerce").isna().all()),
        "mes_unique_sample": df["mes"].dropna().unique().tolist()[:10],
        "idUsuario_sample": df["idUsuario"].dropna().unique().tolist()[:10],
        "empresaNombre_sample": df["empresaNombre"].dropna().unique().tolist()[:10],
    }

# ==============================
# Endpoints KPIs (con selector de claves)
# ==============================
@app.get("/kpis/combustible")
def ep_kpis_combustible(
    start_date: str|None = Query(None, description="YYYY-MM-DD (fechaEmision)"),
    end_date: str|None = Query(None, description="YYYY-MM-DD (fechaEmision)"),
    empresa: str|None = Query(None),
    idUsuario: str|None = Query(None),
    section: str|None = Query(None, pattern="^(empresa|usuario)$"),
    fields: str|None = Query(None, description="Claves separadas por coma, p.ej.: gasto_mes_emp,precio_global_emp"),
):
    try:
        filt = _build_filter_fechas_cabecera(start_date, end_date)
        if empresa:
            filt["empresaNombre"] = empresa
        if idUsuario:
            filt["idUsuario"] = idUsuario

        df = load_df_mongo(coll_fuel, filt)
        if df.empty:
            return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

        df = add_time_cols_fuel_ev(df)
        df["lineas_parsed"] = df.get("lineas", pd.Series([[]]*len(df))).apply(parse_lineas)
        df_lines = explode_fuel_lines(df)
        if df_lines.empty:
            return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

        payload = {
            "usuario": _ser(kpis_usuario_fuel(df, df_lines)),
            "empresa": _ser(kpis_empresa_fuel(df, df_lines)),
        }
        fields_list = [s.strip() for s in fields.split(",")] if fields else None
        return _filter_payload(payload, section, fields_list)
    except Exception as e:
        print("ERROR /kpis/combustible:", repr(e))
        return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

@app.get("/kpis/ev")
def ep_kpis_ev(
    start_date: str|None = Query(None, description="YYYY-MM-DD (fechaEmision)"),
    end_date: str|None = Query(None, description="YYYY-MM-DD (fechaEmision)"),
    empresa: str|None = Query(None),
    idUsuario: str|None = Query(None),
    section: str|None = Query(None, pattern="^(empresa|usuario)$"),
    fields: str|None = Query(None, description="Claves separadas por coma, p.ej.: kwh_mes_emp,precio_global_emp"),
):
    try:
        filt = _build_filter_fechas_cabecera(start_date, end_date)
        if empresa:
            filt["empresaNombre"] = empresa
        if idUsuario:
            filt["idUsuario"] = idUsuario

        df = load_df_mongo(coll_ev, filt)
        if df.empty:
            return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

        df = add_time_cols_fuel_ev(df)
        df["lineas_parsed"] = df.get("lineas", pd.Series([[]]*len(df))).apply(parse_lineas)
        df_lines = explode_ev_lines(df)
        if df_lines.empty:
            return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

        payload = {
            "usuario": _ser(kpis_usuario_ev(df, df_lines)),
            "empresa": _ser(kpis_empresa_ev(df, df_lines)),
        }
        fields_list = [s.strip() for s in fields.split(",")] if fields else None
        return _filter_payload(payload, section, fields_list)
    except Exception as e:
        print("ERROR /kpis/ev:", repr(e))
        return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

@app.get("/kpis/peaje")
def ep_kpis_peaje(
    start_date: str|None = Query(None, description="YYYY-MM-DD (fechaHora)"),
    end_date: str|None = Query(None, description="YYYY-MM-DD (fechaHora)"),
    empresa: str|None = Query(None),
    idUsuario: str|None = Query(None),
    section: str|None = Query(None, pattern="^(empresa|usuario)$"),
    fields: str|None = Query(None, description="Claves separadas por coma, p.ej.: gasto_mes_emp,emp_autopista_mes"),
):
    try:
        filt = _build_filter_fechas_toll(start_date, end_date)
        if empresa:
            filt["empresaNombre"] = empresa
        if idUsuario:
            filt["idUsuario"] = idUsuario

        df = load_df_mongo(coll_toll, filt)
        if df.empty:
            return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

        for col, default in [
            ("idUsuario", "DESCONOCIDO"),
            ("empresaNombre","EMPRESA_UNICA"),
            ("autopista", "SIN_AUTOPISTA"),
            ("formaPago", "DESCONOCIDO"),
            ("fechaHora", pd.NaT),
        ]:
            if col not in df.columns:
                df[col] = default

        cand_imp = [c for c in df.columns if any(p in c.lower() for p in ["importe","total","precio","coste","amount","valor","pago"])]
        if cand_imp:
            col = cand_imp[0]
            imp_series = df[col]
            num = pd.to_numeric(imp_series, errors="coerce")
            if num.isna().any():
                parsed = imp_series.apply(_to_num_eur)
                num = num.where(num.notna(), parsed)
            df["importe"] = num
        else:
            df["importe"] = np.nan

        df = add_time_cols_toll(df)
        if df["importe"].isna().all():
            return _filter_payload({"usuario": {}, "empresa": {}}, section, None)

        usuario = kpis_usuario_toll(df)
        empresa_kpi = kpis_empresa_toll(df)

        payload = {
            "usuario": _ser(usuario) if usuario else {},
            "empresa": _ser(empresa_kpi) if empresa_kpi else {},
        }
        fields_list = [s.strip() for s in fields.split(",")] if fields else None
        return _filter_payload(payload, section, fields_list)
    except Exception as e:
        return _filter_payload(
            {"usuario": {}, "empresa": {}, "error": str(e), "traceback": traceback.format_exc()},
            section, None
        )

# ==============================
# KPI "todo en uno" (documentación visible)
# ==============================
@app.get("/kpis/all")
def ep_kpis_all(
    start_date: str|None = Query(None, description="YYYY-MM-DD"),
    end_date: str|None = Query(None, description="YYYY-MM-DD"),
    empresa: str|None = Query(None, description="Nombre de la empresa"),
    idUsuario: str|None = Query(None, description="(Opcional) filtrar por usuario concreto"),
    fields: str|None = Query(None, description="Claves (datasets) separadas por coma para filtrar dentro de cada dominio"),
):
    """
    Devuelve TODOS los KPIs de nivel EMPRESA para combustible, ev y peaje en una sola respuesta.
    Estructura:
    {
      "combustible": { ...kpis empresa fuel... },
      "ev":          { ...kpis empresa ev... },
      "peaje":       { ...kpis empresa peaje... }
    }
    Si se pasa `fields`, se filtran las claves dentro de cada dominio (p.ej. gasto_mes_emp, ranking_veh, consumo_emp_user_mes).
    """
    domains = ["combustible", "ev", "peaje"]
    fields_list = [s.strip() for s in fields.split(",")] if fields else None

    out = {}
    for d in domains:
        payload = _compute_payload(d, start_date, end_date, empresa, idUsuario)
        emp = payload.get("empresa", {}) or {}
        if fields_list:
            emp = {k: v for k, v in emp.items() if k in fields_list}
        out[d] = emp
    return out

# ==============================
# Aliases: "un endpoint por gráfica"
# ==============================
def _compute_payload(domain: str, start_date: str|None, end_date: str|None, empresa: str|None, idUsuario: str|None) -> dict:
    if domain == "combustible":
        filt = _build_filter_fechas_cabecera(start_date, end_date)
        if empresa:
            filt["empresaNombre"] = empresa
        if idUsuario:
            filt["idUsuario"] = idUsuario
        df = load_df_mongo(coll_fuel, filt)
        if df.empty:
            return {"usuario": {}, "empresa": {}}
        df = add_time_cols_fuel_ev(df)
        df["lineas_parsed"] = df.get("lineas", pd.Series([[]]*len(df))).apply(parse_lineas)
        df_lines = explode_fuel_lines(df)
        if df_lines.empty:
            return {"usuario": {}, "empresa": {}}
        return {
            "usuario": _ser(kpis_usuario_fuel(df, df_lines)),
            "empresa": _ser(kpis_empresa_fuel(df, df_lines))
        }

    elif domain == "ev":
        filt = _build_filter_fechas_cabecera(start_date, end_date)
        if empresa:
            filt["empresaNombre"] = empresa
        if idUsuario:
            filt["idUsuario"] = idUsuario
        df = load_df_mongo(coll_ev, filt)
        if df.empty:
            return {"usuario": {}, "empresa": {}}
        df = add_time_cols_fuel_ev(df)
        df["lineas_parsed"] = df.get("lineas", pd.Series([[]]*len(df))).apply(parse_lineas)
        df_lines = explode_ev_lines(df)
        if df_lines.empty:
            return {"usuario": {}, "empresa": {}}
        return {
            "usuario": _ser(kpis_usuario_ev(df, df_lines)),
            "empresa": _ser(kpis_empresa_ev(df, df_lines))
        }

    elif domain == "peaje":
        filt = _build_filter_fechas_toll(start_date, end_date)
        if empresa:
            filt["empresaNombre"] = empresa
        if idUsuario:
            filt["idUsuario"] = idUsuario
        df = load_df_mongo(coll_toll, filt)
        if df.empty:
            return {"usuario": {}, "empresa": {}}

        for col, default in [
            ("idUsuario","DESCONOCIDO"),("empresaNombre","EMPRESA_UNICA"),
            ("autopista","SIN_AUTOPISTA"),("formaPago","DESCONOCIDO"),("fechaHora",pd.NaT)
        ]:
            if col not in df.columns:
                df[col] = default

        cand_imp = [c for c in df.columns if any(p in c.lower() for p in ["importe","total","precio","coste","amount","valor","pago"])]
        if cand_imp:
            col = cand_imp[0]
            num = pd.to_numeric(df[col], errors="coerce")
            if num.isna().any():
                parsed = df[col].apply(_to_num_eur)
                num = num.where(num.notna(), parsed)
            df["importe"] = num
        else:
            df["importe"] = np.nan

        df = add_time_cols_toll(df)
        if df["importe"].isna().all():
            return {"usuario": {}, "empresa": {}}
        return {
            "usuario": _ser(kpis_usuario_toll(df)),
            "empresa": _ser(kpis_empresa_toll(df))
        }

    elif domain == "all":
        return _compute_all_empresa(start_date, end_date, empresa, idUsuario)

    else:
        raise HTTPException(status_code=400, detail="domain inválido")

@app.get("/charts/{domain}/{section}/{field}")
def chart_endpoint(
    domain: str,
    section: str,
    field: str,
    start_date: str|None = Query(None, description="YYYY-MM-DD"),
    end_date: str|None = Query(None, description="YYYY-MM-DD"),
    empresa: str|None = Query(None),
    idUsuario: str|None = Query(None),
):
    domain = domain.lower()
    section = section.lower()
    if domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"domain debe ser uno de {sorted(VALID_DOMAINS)}")
    if section not in VALID_SECTIONS:
        raise HTTPException(status_code=400, detail=f"section debe ser uno de {sorted(VALID_SECTIONS)}")

    payload = _compute_payload(domain, start_date, end_date, empresa, idUsuario)
    data = payload.get(section, {})
    if field not in data:
        return {
            "error": f"field '{field}' no disponible para section '{section}' en '{domain}'",
            "available_fields": sorted(list(data.keys()))
        }
    return {field: data[field]}

# ==============================
# Root
# ==============================
@app.get("/")
def root():
    return {
        "message": "API KPIs lista. Endpoints: /kpis/combustible, /kpis/ev, /kpis/peaje, /kpis/all | Aliases por gráfica: /charts/{domain}/{section}/{field} (dominios: combustible, ev, peaje, all)",
        "health": "/health",
        "debug_config": "/debug/config",
        "debug_peek": "/debug/peek",
        "debug_distinct": "/debug/distinct?collection=Peaje&field=idUsuario",
        "debug_peaje_sample": "/debug/peaje_sample",
        "debug_peaje_after": "/debug/peaje_after",
        "kpis_all_example": "/kpis/all?empresa=ACME&start_date=2025-01-01&end_date=2025-12-31&fields=gasto_mes_emp,consumo_emp_user_mes",
        "charts_example": "/charts/all/empresa/gasto_mes_emp_all?empresa=ACME&start_date=2025-01-01&end_date=2025-12-31"
    }

# ==============================
# ENDPOINTS SUSTAINABILITY (EV + ICE unificados) — SECCIÓN AUTÓNOMA
# ==============================

COL_SUS = os.getenv("COL_SUS", "Sustainability")
coll_sus = db[COL_SUS]

try:
    KGCO2_PER_L_GASOLINA
except NameError:
    KGCO2_PER_L_GASOLINA = 2.35
try:
    KGCO2_PER_L_DIESEL
except NameError:
    KGCO2_PER_L_DIESEL = 2.69
try:
    GRID_KGCO2_PER_KWH
except NameError:
    GRID_KGCO2_PER_KWH = 0.283
try:
    LOSS_FACTOR_TD
except NameError:
    LOSS_FACTOR_TD = 1.096

def sus_safe_ratio(a: float, b: float) -> float | None:
    a = 0.0 if pd.isna(a) else float(a)
    b = 0.0 if pd.isna(b) else float(b)
    return (a / b) if b > 0 else None

def sus_match_date_range_mes(df: pd.DataFrame, from_: str | None, to_: str | None) -> pd.DataFrame:
    if "mes" not in df.columns:
        return df
    out = df.copy()
    if from_:
        out = out[out["mes"] >= from_]
    if to_:
        out = out[out["mes"] <= to_]
    return out

def sus_materialize_from_domain_collections(
    from_: str | None, to_: str | None,
    idEmpresa: str | None, idUsuario: str | None,
    propulsion: str | None
) -> pd.DataFrame:
    # Combustible
    filt_fuel = _build_filter_fechas_cabecera(from_, to_)
    if idEmpresa:   filt_fuel["empresaNombre"] = idEmpresa
    if idUsuario:   filt_fuel["idUsuario"] = idUsuario
    df_fuel = load_df_mongo(coll_fuel, filt_fuel)

    if not df_fuel.empty:
        df_fuel = add_time_cols_fuel_ev(df_fuel)
        df_fuel["lineas_parsed"] = df_fuel.get("lineas", pd.Series([[]] * len(df_fuel))).apply(parse_lineas)
        lines_f = explode_fuel_lines(df_fuel)

        if lines_f.empty:
            fuel_tx = df_fuel[["idUsuario", "empresaNombre", "mes"]].copy()
            fuel_tx["litros"] = np.nan
        else:
            fuel_tx = lines_f[["idUsuario", "empresaTransporte", "mes", "litros"]].rename(
                columns={"empresaTransporte": "empresaNombre"}
            )

        fuel_tx["idEmpresa"] = fuel_tx["empresaNombre"].astype(str)
        if "idVehiculo" in df_fuel.columns:
            veh_map_f = (df_fuel.dropna(subset=["idUsuario", "idVehiculo"])
                               .drop_duplicates("idUsuario")[["idUsuario", "idVehiculo"]])
            fuel_tx = fuel_tx.merge(veh_map_f, on="idUsuario", how="left")
        fuel_tx["idVehiculo"] = fuel_tx["idVehiculo"].fillna(
            fuel_tx["idEmpresa"].astype(str) + "-" + fuel_tx["idUsuario"].astype(str) + "-ICE"
        )

        fuel_tx["propulsion"] = "ICE"
        factor_ice = KGCO2_PER_L_DIESEL
        fuel_tx["litros"] = pd.to_numeric(fuel_tx["litros"], errors="coerce")
        fuel_tx["kgCO2_ice"] = fuel_tx["litros"] * factor_ice
        fuel_tx["kwh"] = np.nan
        fuel_tx["kgCO2e_ev"] = np.nan
        fuel_tx["kgCO2_total"] = fuel_tx["kgCO2_ice"]

        fuel_tx = fuel_tx[[
            "idEmpresa", "idUsuario", "idVehiculo", "propulsion", "mes", "kwh", "litros",
            "kgCO2e_ev", "kgCO2_ice", "kgCO2_total"
        ]]
    else:
        fuel_tx = pd.DataFrame(columns=[
            "idEmpresa", "idUsuario", "idVehiculo", "propulsion", "mes", "kwh", "litros",
            "kgCO2e_ev", "kgCO2_ice", "kgCO2_total"
        ])

    # EV
    filt_ev = _build_filter_fechas_cabecera(from_, to_)
    if idEmpresa:   filt_ev["empresaNombre"] = idEmpresa
    if idUsuario:   filt_ev["idUsuario"] = idUsuario
    df_ev = load_df_mongo(coll_ev, filt_ev)

    if not df_ev.empty:
        df_ev = add_time_cols_fuel_ev(df_ev)
        df_ev["lineas_parsed"] = df_ev.get("lineas", pd.Series([[]] * len(df_ev))).apply(parse_lineas)
        lines_e = explode_ev_lines(df_ev)

        if lines_e.empty:
            ev_tx = df_ev[["idUsuario", "empresaNombre", "mes"]].copy()
            ev_tx["kwh"] = np.nan
        else:
            ev_tx = lines_e[["idUsuario", "empresaTransporte", "mes", "kwh"]].rename(
                columns={"empresaTransporte": "empresaNombre"}
            )

        ev_tx["idEmpresa"] = ev_tx["empresaNombre"].astype(str)
        if "idVehiculo" in df_ev.columns:
            veh_map_e = (df_ev.dropna(subset=["idUsuario", "idVehiculo"])
                              .drop_duplicates("idUsuario")[["idUsuario", "idVehiculo"]])
            ev_tx = ev_tx.merge(veh_map_e, on="idUsuario", how="left")
        ev_tx["idVehiculo"] = ev_tx["idVehiculo"].fillna(
            ev_tx["idEmpresa"].astype(str) + "-" + ev_tx["idUsuario"].astype(str) + "-EV"
        )

        ev_tx["propulsion"] = "EV"
        ev_tx["kwh"] = pd.to_numeric(ev_tx["kwh"], errors="coerce")
        ev_tx["kgCO2e_ev"] = ev_tx["kwh"] * LOSS_FACTOR_TD * GRID_KGCO2_PER_KWH
        ev_tx["litros"] = np.nan
        ev_tx["kgCO2_ice"] = np.nan
        ev_tx["kgCO2_total"] = ev_tx["kgCO2e_ev"]

        ev_tx = ev_tx[[
            "idEmpresa", "idUsuario", "idVehiculo", "propulsion", "mes", "kwh", "litros",
            "kgCO2e_ev", "kgCO2_ice", "kgCO2_total"
        ]]
    else:
        ev_tx = pd.DataFrame(columns=[
            "idEmpresa", "idUsuario", "idVehiculo", "propulsion", "mes", "kwh", "litros",
            "kgCO2e_ev", "kgCO2_ice", "kgCO2_total"
        ])

    df = pd.concat([fuel_tx, ev_tx], ignore_index=True, sort=False)

    if propulsion:
        df = df[df["propulsion"].astype(str).str.upper() == propulsion.upper()]

    if "mes" not in df.columns or df["mes"].isna().all():
        df["mes"] = "NA"
    df["mes"] = df["mes"].astype(str)

    df = sus_match_date_range_mes(df, from_, to_)
    return df

def sus_get_df(
    from_: str | None, to_: str | None,
    idEmpresa: str | None, idUsuario: str | None,
    propulsion: str | None
) -> pd.DataFrame:
    try:
        if coll_sus.estimated_document_count() > 0:
            q: Dict[str, Any] = {}
            if idEmpresa:  q["idEmpresa"]  = idEmpresa
            if idUsuario:  q["idUsuario"]  = idUsuario
            if propulsion: q["propulsion"] = propulsion
            docs = list(coll_sus.find(q, {"_id": 0}))
            df = pd.DataFrame(docs)
            if df.empty:
                return df
            for c in ["kwh", "litros", "kgCO2e_ev", "kgCO2_ice", "kgCO2_total"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "mes" in df.columns:
                df["mes"] = df["mes"].astype(str)
                df = sus_match_date_range_mes(df, from_, to_)
            else:
                df["mes"] = "NA"
            return df
    except Exception:
        pass
    return sus_materialize_from_domain_collections(from_, to_, idEmpresa, idUsuario, propulsion)

def sus_paginate(items: list, limit: int, offset: int) -> list:
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    return items[offset:offset + limit]

@app.get("/v1/sustainability/companies")
def sus_companies(
    from_: str | None = Query(None, alias="from", description="YYYY-MM"),
    to_:   str | None = Query(None, alias="to",   description="YYYY-MM"),
    propulsion: str | None = Query(None, description="EV|ICE"),
    limit: int = 100,
    offset: int = 0,
):
    df = sus_get_df(from_, to_, None, None, propulsion)
    if df.empty:
        return {"count": 0, "items": []}
    g = (df.groupby(["idEmpresa"], dropna=False)
           .agg(kwh=("kwh", "sum"),
                litros=("litros", "sum"),
                kgCO2e_ev=("kgCO2e_ev", "sum"),
                kgCO2_ice=("kgCO2_ice", "sum"),
                kgCO2_total=("kgCO2_total", "sum"),
                n_usuarios=("idUsuario", pd.Series.nunique),
                n_vehiculos=("idVehiculo", pd.Series.nunique))
           .reset_index())
    g["share_co2_ev"] = g.apply(lambda r: sus_safe_ratio(r["kgCO2e_ev"], r["kgCO2e_ev"] + r["kgCO2_ice"]), axis=1)
    items = sus_paginate(g.to_dict(orient="records"), limit, offset)
    return {"count": len(items), "items": items}

@app.get("/v1/sustainability/companies/{idEmpresa}/summary")
def sus_company_summary(
    idEmpresa: str,
    from_: str | None = Query(None, alias="from"),
    to_:   str | None = Query(None, alias="to"),
    propulsion: str | None = Query(None, description="EV|ICE"),
):
    df = sus_get_df(from_, to_, idEmpresa, None, propulsion)
    if df.empty:
        return {"idEmpresa": idEmpresa, "kwh": 0, "litros": 0, "kgCO2e_ev": 0, "kgCO2_ice": 0, "kgCO2_total": 0,
                "n_usuarios": 0, "n_vehiculos": 0, "share_co2_ev": None}
    g = (df.groupby(["idEmpresa"], dropna=False)
           .agg(kwh=("kwh", "sum"),
                litros=("litros", "sum"),
                kgCO2e_ev=("kgCO2e_ev", "sum"),
                kgCO2_ice=("kgCO2_ice", "sum"),
                kgCO2_total=("kgCO2_total", "sum"),
                n_usuarios=("idUsuario", pd.Series.nunique),
                n_vehiculos=("idVehiculo", pd.Series.nunique))
           .reset_index())
    r = g.iloc[0].to_dict()
    r["share_co2_ev"] = sus_safe_ratio(r["kgCO2e_ev"], r["kgCO2e_ev"] + r["kgCO2_ice"])
    return r

@app.get("/v1/sustainability/companies/{idEmpresa}/months")
def sus_company_months(
    idEmpresa: str,
    from_: str | None = Query(None, alias="from"),
    to_:   str | None = Query(None, alias="to"),
    propulsion: str | None = Query(None, description="EV|ICE"),
    limit: int = 500, offset: int = 0
):
    df = sus_get_df(from_, to_, idEmpresa, None, propulsion)
    if df.empty:
        return {"count": 0, "items": []}
    g = (df.groupby(["idEmpresa", "mes"], dropna=False)
           .agg(kwh=("kwh", "sum"),
                litros=("litros", "sum"),
                kgCO2e_ev=("kgCO2e_ev", "sum"),
                kgCO2_ice=("kgCO2_ice", "sum"),
                kgCO2_total=("kgCO2_total", "sum"))
           .reset_index().sort_values(["mes"]))
    g["share_co2_ev"] = g.apply(lambda r: sus_safe_ratio(r["kgCO2e_ev"], r["kgCO2e_ev"] + r["kgCO2_ice"]), axis=1)
    items = sus_paginate(g.to_dict(orient="records"), limit, offset)
    return {"count": len(items), "items": items}

@app.get("/v1/sustainability/companies/{idEmpresa}/users")
def sus_company_users(
    idEmpresa: str,
    from_: str | None = Query(None, alias="from"),
    to_:   str | None = Query(None, alias="to"),
    propulsion: str | None = Query(None, description="EV|ICE"),
    limit: int = 200, offset: int = 0
):
    df = sus_get_df(from_, to_, idEmpresa, None, propulsion)
    if df.empty:
        return {"count": 0, "items": []}
    g = (df.groupby(["idEmpresa", "idUsuario"], dropna=False)
           .agg(kwh=("kwh", "sum"),
                litros=("litros", "sum"),
                kgCO2e_ev=("kgCO2e_ev", "sum"),
                kgCO2_ice=("kgCO2_ice", "sum"),
                kgCO2_total=("kgCO2_total", "sum"),
                n_vehiculos=("idVehiculo", pd.Series.nunique))
           .reset_index().sort_values(["idUsuario"]))
    g["share_co2_ev"] = g.apply(lambda r: sus_safe_ratio(r["kgCO2e_ev"], r["kgCO2e_ev"] + r["kgCO2_ice"]), axis=1)
    items = sus_paginate(g.to_dict(orient="records"), limit, offset)
    return {"count": len(items), "items": items}

@app.get("/v1/sustainability/companies/{idEmpresa}/users/{idUsuario}/months")
def sus_company_user_months(
    idEmpresa: str, idUsuario: str,
    from_: str | None = Query(None, alias="from"),
    to_:   str | None = Query(None, alias="to"),
    propulsion: str | None = Query(None, description="EV|ICE"),
    limit: int = 200, offset: int = 0
):
    df = sus_get_df(from_, to_, idEmpresa, idUsuario, propulsion)
    if df.empty:
        return {"count": 0, "items": []}
    g = (df.groupby(["idEmpresa", "idUsuario", "mes"], dropna=False)
           .agg(kwh=("kwh", "sum"),
                litros=("litros", "sum"),
                kgCO2e_ev=("kgCO2e_ev", "sum"),
                kgCO2_ice=("kgCO2_ice", "sum"),
                kgCO2_total=("kgCO2_total", "sum"))
           .reset_index().sort_values(["mes"]))
    g["share_co2_ev"] = g.apply(lambda r: sus_safe_ratio(r["kgCO2e_ev"], r["kgCO2e_ev"] + r["kgCO2_ice"]), axis=1)
    items = sus_paginate(g.to_dict(orient="records"), limit, offset)
    return {"count": len(items), "items": items}

@app.get("/v1/sustainability/vehicles")
def sus_vehicles(
    idEmpresa: str | None = None,
    idUsuario: str | None = None,
    propulsion: str | None = Query(None, description="EV|ICE"),
    from_: str | None = Query(None, alias="from"),
    to_:   str | None = Query(None, alias="to"),
    limit: int = 500, offset: int = 0
):
    df = sus_get_df(from_, to_, idEmpresa, idUsuario, propulsion)
    if df.empty:
        return {"count": 0, "items": []}
    g = (df.groupby(["idEmpresa", "idUsuario", "idVehiculo", "propulsion"], dropna=False)
           .agg(kwh=("kwh", "sum"),
                litros=("litros", "sum"),
                kgCO2e_ev=("kgCO2e_ev", "sum"),
                kgCO2_ice=("kgCO2_ice", "sum"),
                kgCO2_total=("kgCO2_total", "sum"))
           .reset_index())
    items = sus_paginate(g.to_dict(orient="records"), limit, offset)
    return {"count": len(items), "items": items}
