from __future__ import annotations

import io
import itertools
import math
import re
from pathlib import Path

import dash
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CHAIN_COLORS = {
    "Amanhecer": "#e63946",
    "Meu Super": "#3b82f6",
    "Volta": "#fb923c",
}
DEFAULT_RADIUS = 500


def clean_text(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&amp;", "&")
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def clean_coord(value: object, kind: str, lat_for_lon: float | None = None) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        s = str(value).strip().replace(",", ".")
        if not s or s.lower() == "nan":
            return None
        v = float(s)
    except Exception:
        return None

    if kind == "lat":
        if -90 <= v <= 90:
            return v
        return None

    # longitude corrections for common malformed CSV values, e.g. -8073513 -> -8.073513
    if abs(v) > 180:
        digits = re.sub(r"[^0-9]", "", str(value))
        if digits:
            sign = -1 if str(value).strip().startswith("-") else 1
            for denom in (1e6, 1e7, 1e8):
                cand = sign * (float(digits) / denom)
                if -32 <= cand <= -5:
                    return cand
    if lat_for_lon is not None and 36 <= lat_for_lon <= 43.5 and 0 < v < 10:
        return -v
    if -32 <= v <= -5:
        return v
    if -180 <= v <= 180:
        return v
    return None


def google_streetview_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.7f},{lon:.7f}"


def apple_lookaround_url(lat: float, lon: float) -> str:
    # Opens Apple Maps centered at the coordinate; where Apple Look Around exists, user can enter Look Around.
    return f"https://maps.apple.com/?ll={lat:.7f},{lon:.7f}&q=Look%20Around"


def normalize_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    fname = path.stem.lower()
    if "amanhecer" in fname:
        cadeia = "Amanhecer"
        df = pd.DataFrame({
            "id": raw.get("storeUrl", raw.index),
            "nome": raw.get("name", ""),
            "morada": raw.get("address", ""),
            "codigo_postal": raw.get("postalCode", ""),
            "municipio": raw.get("city", ""),
            "distrito": raw.get("zone", ""),
            "telefone": raw.get("phone", ""),
            "email": raw.get("email", ""),
            "horario": raw.get("hours", ""),
            "servicos": "",
            "latitude": raw.get("latitude", np.nan),
            "longitude": raw.get("longitude", np.nan),
        })
    elif "meusuper" in fname or "meu_super" in fname:
        cadeia = "Meu Super"
        df = pd.DataFrame({
            "id": raw.get("id", raw.index),
            "nome": raw.get("nome", raw.get("name", "")),
            "morada": raw.get("morada", raw.get("address", "")),
            "codigo_postal": "",
            "municipio": "",
            "distrito": "",
            "telefone": raw.get("telefone", ""),
            "email": "",
            "horario": raw.get("horario", raw.get("horarios", "")),
            "servicos": raw.get("servicos", raw.get("servico", "")),
            "latitude": raw.get("latitude", raw.get("lat", np.nan)),
            "longitude": raw.get("longitude", raw.get("lng", np.nan)),
        })
    elif "volta" in fname:
        cadeia = "Volta"
        df = pd.DataFrame({
            "id": raw.get("id", raw.index),
            "nome": raw.get("titulo", raw.get("nome", "")),
            "morada": raw.get("morada", raw.get("address", "")),
            "codigo_postal": raw.get("cp", ""),
            "municipio": raw.get("municipio", ""),
            "distrito": raw.get("distrito", ""),
            "telefone": raw.get("telefone", ""),
            "email": raw.get("email", ""),
            "horario": raw.get("horario", raw.get("horarios", "")),
            "servicos": raw.get("servico", raw.get("servicos", "")),
            "latitude": raw.get("lat", raw.get("latitude", np.nan)),
            "longitude": raw.get("lng", raw.get("longitude", np.nan)),
        })
    else:
        cadeia = path.stem.title()
        cols = {c.lower(): c for c in raw.columns}
        def pick(*names):
            for n in names:
                if n in cols:
                    return raw[cols[n]]
            return ""
        df = pd.DataFrame({
            "id": pick("id"),
            "nome": pick("nome", "name", "titulo"),
            "morada": pick("morada", "address"),
            "codigo_postal": pick("cp", "postalcode", "codigo_postal"),
            "municipio": pick("municipio", "city", "concelho"),
            "distrito": pick("distrito", "zone"),
            "telefone": pick("telefone", "phone"),
            "email": pick("email"),
            "horario": pick("horario", "horarios", "hours"),
            "servicos": pick("servico", "servicos"),
            "latitude": pick("latitude", "lat"),
            "longitude": pick("longitude", "lng", "lon"),
        })

    df["cadeia"] = cadeia
    for c in ["nome", "morada", "codigo_postal", "municipio", "distrito", "telefone", "email", "horario", "servicos"]:
        df[c] = df[c].apply(clean_text)
    lat = [clean_coord(x, "lat") for x in df["latitude"]]
    lon = [clean_coord(x, "lon", la) for x, la in zip(df["longitude"], lat)]
    df["latitude"] = lat
    df["longitude"] = lon
    # Keep rows for volta, amanhecer, and meu super only
    df = df[df["cadeia"].isin(["Volta", "Amanhecer", "Meu Super"])]
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[(df["latitude"].between(30, 43.8)) & (df["longitude"].between(-32, -5))].copy()
    df["street_view"] = [google_streetview_url(a, b) for a, b in zip(df.latitude, df.longitude)]
    df["apple_lookaround"] = [apple_lookaround_url(a, b) for a, b in zip(df.latitude, df.longitude)]
    df["coords"] = df.apply(lambda r: f"{r.latitude:.6f}, {r.longitude:.6f}", axis=1)
    return df.reset_index(drop=True)


def load_data() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        frames.append(normalize_file(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["row_id"] = np.arange(len(df))
    return df


DF = load_data()
CHAINS = sorted(DF["cadeia"].unique().tolist()) if not DF.empty and "cadeia" in DF.columns else []

# A aplicação considera apenas interseções com a Volta.
# Não calcular/mostrar Amanhecer x Meu Super.
ALLOWED_INTERSECTION_PAIRS = [("Amanhecer", "Volta"), ("Meu Super", "Volta")]
PAIR_OPTIONS = ["Todas"] + [f"{a} x {b}" for a, b in ALLOWED_INTERSECTION_PAIRS]


def hdist_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - lat1)
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def intersection_pairs(chain_a: str, chain_b: str, radius: int, max_rows: int = 5000) -> pd.DataFrame:
    """Return pairs from two different chains within radius meters.

    radius == 0 is treated as exact same coordinate after 6 decimal rounding.
    This is useful to identify the same physical point / same geocoded location.
    """
    radius = int(radius or 0)
    a = DF[DF.cadeia == chain_a].reset_index(drop=True)
    b = DF[DF.cadeia == chain_b].reset_index(drop=True)
    if a.empty or b.empty:
        return pd.DataFrame()

    rows = []

    if radius <= 0:
        bb = b.copy()
        bb["coord_key"] = bb.apply(lambda r: f"{float(r.latitude):.6f}|{float(r.longitude):.6f}", axis=1)
        lookup = {k: g for k, g in bb.groupby("coord_key")}
        for _, ra in a.iterrows():
            key = f"{float(ra.latitude):.6f}|{float(ra.longitude):.6f}"
            if key not in lookup:
                continue
            for _, rb in lookup[key].iterrows():
                rows.append({
                    "row_id_a": int(ra.row_id) if "row_id" in ra.index else int(ra.name),
                    "cadeia_a": str(chain_a),
                    "loja_a": str(ra.nome),
                    "morada_a": str(ra.morada),
                    "lat_a": float(ra.latitude),
                    "lon_a": float(ra.longitude),
                    "row_id_b": int(rb.row_id) if "row_id" in rb.index else int(rb.name),
                    "cadeia_b": str(chain_b),
                    "loja_b": str(rb.nome),
                    "morada_b": str(rb.morada),
                    "lat_b": float(rb.latitude),
                    "lon_b": float(rb.longitude),
                    "dist_m": 0.0,
                    "street_view_a": str(ra.street_view),
                    "apple_lookaround_a": str(ra.apple_lookaround),
                    "street_view_b": str(rb.street_view),
                    "apple_lookaround_b": str(rb.apple_lookaround),
                })
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break
    else:
        b_lat = b.latitude.to_numpy(dtype=float)
        b_lon = b.longitude.to_numpy(dtype=float)
        for _, ra in a.iterrows():
            d = hdist_m(float(ra.latitude), float(ra.longitude), b_lat, b_lon)
            idx = np.where(d <= radius)[0]
            for j in idx:
                rb = b.iloc[int(j)]
                rows.append({
                    "row_id_a": int(ra.row_id) if "row_id" in ra.index else int(ra.name),
                    "cadeia_a": str(chain_a),
                    "loja_a": str(ra.nome),
                    "morada_a": str(ra.morada),
                    "lat_a": float(ra.latitude),
                    "lon_a": float(ra.longitude),
                    "row_id_b": int(rb.row_id) if "row_id" in rb.index else int(rb.name),
                    "cadeia_b": str(chain_b),
                    "loja_b": str(rb.nome),
                    "morada_b": str(rb.morada),
                    "lat_b": float(rb.latitude),
                    "lon_b": float(rb.longitude),
                    "dist_m": float(d[j]),
                    "street_view_a": str(ra.street_view),
                    "apple_lookaround_a": str(ra.apple_lookaround),
                    "street_view_b": str(rb.street_view),
                    "apple_lookaround_b": str(rb.apple_lookaround),
                })
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("dist_m", kind="stable").reset_index(drop=True)
    return out

def all_intersections(radius: int, pair_value: str) -> pd.DataFrame:
    if pair_value != "Todas":
        a, b = pair_value.split(" x ")
        return intersection_pairs(a, b, radius)
    frames = []
    for a, b in ALLOWED_INTERSECTION_PAIRS:
        x = intersection_pairs(a, b, radius, max_rows=3000)
        if not x.empty:
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nearest_competition(df: pd.DataFrame, radius: int = DEFAULT_RADIUS) -> pd.DataFrame:
    rows = []
    for i, r in df.iterrows():
        other = df[df.cadeia != r.cadeia]
        if other.empty:
            rows.append((np.nan, "", 0))
            continue
        d = hdist_m(r.latitude, r.longitude, other.latitude.to_numpy(), other.longitude.to_numpy())
        if len(d) == 0:
            rows.append((np.nan, "", 0))
            continue
        j = int(np.argmin(d))
        rows.append((float(d[j]), other.iloc[j].cadeia, int((d <= radius).sum())))
    out = df.copy()
    out[["dist_concorrente_m", "concorrente_mais_proximo", "concorrentes_no_raio"]] = pd.DataFrame(rows, index=out.index)
    return out



def competition_dataset(base_chain: str = "Todas", competitor_chain: str = "Todas", radius: int = DEFAULT_RADIUS) -> pd.DataFrame:
    """One row per base store with closest competitor and competitors in radius.

    radius == 0 means exact same coordinate (rounded to 6 decimals).
    """
    radius = int(radius or 0)
    base = DF.copy() if base_chain == "Todas" else DF[DF.cadeia == base_chain].copy()
    rows = []

    for _, r in base.iterrows():
        other = DF[DF.cadeia != r.cadeia].copy()
        if competitor_chain != "Todas":
            other = other[other.cadeia == competitor_chain]
        if other.empty:
            rows.append({**r.to_dict(), "dist_concorrente_m": np.nan, "concorrente_mais_proximo": "", "loja_concorrente": "", "concorrentes_no_raio": 0})
            continue

        if radius <= 0:
            key = f"{float(r.latitude):.6f}|{float(r.longitude):.6f}"
            keys = other.apply(lambda x: f"{float(x.latitude):.6f}|{float(x.longitude):.6f}", axis=1)
            exact = other[keys == key]
            if exact.empty:
                rows.append({**r.to_dict(), "dist_concorrente_m": np.nan, "concorrente_mais_proximo": "", "loja_concorrente": "", "concorrentes_no_raio": 0})
            else:
                rb = exact.iloc[0]
                rows.append({**r.to_dict(), "dist_concorrente_m": 0.0, "concorrente_mais_proximo": str(rb.cadeia), "loja_concorrente": str(rb.nome), "concorrentes_no_raio": int(len(exact))})
        else:
            d = hdist_m(float(r.latitude), float(r.longitude), other.latitude.to_numpy(dtype=float), other.longitude.to_numpy(dtype=float))
            if len(d) == 0:
                rows.append({**r.to_dict(), "dist_concorrente_m": np.nan, "concorrente_mais_proximo": "", "loja_concorrente": "", "concorrentes_no_raio": 0})
                continue
            j = int(np.argmin(d))
            rb = other.iloc[j]
            rows.append({**r.to_dict(), "dist_concorrente_m": float(d[j]), "concorrente_mais_proximo": str(rb.cadeia), "loja_concorrente": str(rb.nome), "concorrentes_no_raio": int((d <= radius).sum())})

    out = pd.DataFrame(rows)
    text_cols = ["cadeia", "nome", "morada", "telefone", "email", "horario", "servicos", "concorrente_mais_proximo", "loja_concorrente"]
    for c in text_cols:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)
    return out


def competition_map(df: pd.DataFrame, title: str):
    if df.empty:
        return empty_fig("Sem dados de concorrência")
    tmp = df.copy()
    tmp = tmp[tmp["concorrentes_no_raio"].fillna(0).astype(int) > 0].copy()
    if tmp.empty:
        return empty_fig("Nenhuma coincidência/interseção para os filtros selecionados")
    tmp["dist_plot"] = tmp["dist_concorrente_m"].fillna(-1)
    tmp["texto"] = tmp.apply(lambda r: f"<b>{r.nome}</b><br>Cadeia: {r.cadeia}<br>Concorrente: {r.concorrente_mais_proximo}<br>Loja concorrente: {r.loja_concorrente}<br>Distância: {r.dist_concorrente_m:.1f} m<br>Concorrentes no raio: {int(r.concorrentes_no_raio)}", axis=1)
    fig = px.scatter_mapbox(
        tmp, lat="latitude", lon="longitude", color="cadeia", size="concorrentes_no_raio",
        color_discrete_map=CHAIN_COLORS, zoom=5.2, height=620,
        hover_name="nome", hover_data={"latitude": False, "longitude": False, "row_id": False, "dist_plot": False},
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", customdata=tmp[["texto"]].to_numpy())
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0), title=title, legend_title="Cadeia")
    return fig

DF_COMP = nearest_competition(DF) if not DF.empty else DF.copy()


def kpi_card(label, value):
    return dbc.Col(html.Div([html.Div(label, className="label"), html.Div(value, className="value")], className="kpi"), md=3)


def empty_fig(message="Sem dados"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#64748b"))
    fig.update_layout(template="plotly_white", height=520, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def map_fig(df: pd.DataFrame, title: str = "", color_by="cadeia", mode="Pontos"):
    if df.empty:
        return empty_fig("Sem lojas para apresentar")
    if mode == "Hexbin / Densidade":
        tmp = df.copy()
        tmp["lat_bin"] = (tmp.latitude / 0.12).round() * 0.12
        tmp["lon_bin"] = (tmp.longitude / 0.12).round() * 0.12
        agg = tmp.groupby(["lat_bin", "lon_bin"], as_index=False).size().rename(columns={"size": "lojas"})
        fig = px.scatter_mapbox(
            agg, lat="lat_bin", lon="lon_bin", size="lojas", color="lojas",
            color_continuous_scale="Viridis", size_max=32, zoom=5.2, height=620,
            hover_data={"lojas": True, "lat_bin": False, "lon_bin": False},
        )
    elif mode == "Distância ao concorrente":
        tmp = DF_COMP[DF_COMP.row_id.isin(df.row_id)].copy()
        tmp["dist_plot"] = tmp["dist_concorrente_m"].fillna(tmp["dist_concorrente_m"].max())
        fig = px.scatter_mapbox(
            tmp, lat="latitude", lon="longitude", color="dist_plot", size="concorrentes_no_raio",
            color_continuous_scale="RdYlGn", zoom=5.2, height=620,
            hover_name="nome",
            hover_data={"cadeia": True, "morada": True, "dist_concorrente_m": ":.0f", "concorrente_mais_proximo": True, "latitude": False, "longitude": False, "row_id": False, "dist_plot": False},
        )
    else:
        fig = px.scatter_mapbox(
            df, lat="latitude", lon="longitude", color=color_by,
            color_discrete_map=CHAIN_COLORS, zoom=5.2, height=620,
            hover_name="nome",
            hover_data={"cadeia": True, "morada": True, "telefone": True, "email": True, "latitude": ":.6f", "longitude": ":.6f", "row_id": False},
        )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0), title=title, legend_title="Cadeia")
    fig.update_traces(marker=dict(opacity=0.82))
    return fig


def chain_table(df: pd.DataFrame):
    cols = ["nome", "cadeia", "morada", "telefone", "email", "horario", "servicos", "coords", "street_view", "apple_lookaround"]
    out = df[cols].copy()
    out["street_view"] = out["street_view"].apply(lambda u: f"[Street View]({u})")
    out["apple_lookaround"] = out["apple_lookaround"].apply(lambda u: f"[Apple Look Around]({u})")
    out = out.fillna("").replace("", "—")
    return dash_table.DataTable(
        data=out.to_dict("records"),
        columns=[
            {"name": "Loja", "id": "nome"}, {"name": "Cadeia", "id": "cadeia"}, {"name": "Morada", "id": "morada"},
            {"name": "Telefone", "id": "telefone"}, {"name": "Email", "id": "email"}, {"name": "Horário", "id": "horario"},
            {"name": "Serviços", "id": "servicos"}, {"name": "Coordenadas", "id": "coords"},
            {"name": "Street View", "id": "street_view", "presentation": "markdown"},
            {"name": "Apple Look Around", "id": "apple_lookaround", "presentation": "markdown"},
        ],
        page_size=12,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Inter, Segoe UI, Arial", "fontSize": 13, "padding": "9px", "textAlign": "left", "maxWidth": 320, "whiteSpace": "normal"},
        style_header={"fontWeight": "800", "backgroundColor": "#f8fafc"},
        markdown_options={"link_target": "_blank"},
    )


def intersection_table(df: pd.DataFrame):
    if df.empty:
        return html.Div("Nenhuma interseção encontrada para o raio selecionado.", className="cardx small-note")
    out = df[["cadeia_a", "loja_a", "cadeia_b", "loja_b", "dist_m", "street_view_a", "apple_lookaround_a", "street_view_b", "apple_lookaround_b"]].copy()
    out["dist_m"] = out["dist_m"].round(1)
    for c in ["street_view_a", "street_view_b"]:
        out[c] = out[c].apply(lambda u: f"[Street View]({u})")
    for c in ["apple_lookaround_a", "apple_lookaround_b"]:
        out[c] = out[c].apply(lambda u: f"[Apple Look Around]({u})")
    return dash_table.DataTable(
        data=out.to_dict("records"),
        columns=[
            {"name": "Cadeia A", "id": "cadeia_a"}, {"name": "Loja A", "id": "loja_a"},
            {"name": "Cadeia B", "id": "cadeia_b"}, {"name": "Loja B", "id": "loja_b"},
            {"name": "Distância (m)", "id": "dist_m", "type": "numeric"},
            {"name": "Street View A", "id": "street_view_a", "presentation": "markdown"},
            {"name": "Look Around A", "id": "apple_lookaround_a", "presentation": "markdown"},
            {"name": "Street View B", "id": "street_view_b", "presentation": "markdown"},
            {"name": "Look Around B", "id": "apple_lookaround_b", "presentation": "markdown"},
        ],
        page_size=12,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Inter, Segoe UI, Arial", "fontSize": 13, "padding": "9px", "textAlign": "left", "maxWidth": 280, "whiteSpace": "normal"},
        style_header={"fontWeight": "800", "backgroundColor": "#f8fafc"},
        markdown_options={"link_target": "_blank"},
    )


def inter_map(df: pd.DataFrame):
    if df.empty:
        return empty_fig("Sem interseções para o raio selecionado")
    fig = go.Figure()
    # Lines limited for performance
    for _, r in df.head(700).iterrows():
        fig.add_trace(go.Scattermapbox(
            lat=[r.lat_a, r.lat_b], lon=[r.lon_a, r.lon_b], mode="lines",
            line=dict(width=1, color="rgba(124,58,237,.35)"), hoverinfo="skip", showlegend=False
        ))
    pts_a = df[["cadeia_a", "loja_a", "lat_a", "lon_a"]].rename(columns={"cadeia_a": "cadeia", "loja_a": "loja", "lat_a": "lat", "lon_a": "lon"})
    pts_b = df[["cadeia_b", "loja_b", "lat_b", "lon_b"]].rename(columns={"cadeia_b": "cadeia", "loja_b": "loja", "lat_b": "lat", "lon_b": "lon"})
    pts = pd.concat([pts_a, pts_b], ignore_index=True).drop_duplicates()
    for cadeia, sub in pts.groupby("cadeia"):
        fig.add_trace(go.Scattermapbox(
            lat=sub.lat, lon=sub.lon, mode="markers", name=cadeia,
            marker=dict(size=10, color=CHAIN_COLORS.get(cadeia, "#111827"), opacity=.9),
            text=sub.loja, hovertemplate="<b>%{text}</b><br>" + cadeia + "<extra></extra>"
        ))
    fig.update_layout(mapbox_style="open-street-map", height=620, margin=dict(l=0, r=0, t=20, b=0), legend_title="Cadeia")
    fig.update_mapboxes(center=dict(lat=float(pts.lat.mean()), lon=float(pts.lon.mean())), zoom=6)
    return fig


def sidebar():
    chain_links = [dcc.Link(f"🏪 {c}", href=f"/cadeia/{c}", className="side-link") for c in CHAINS]
    return html.Div([
        html.H2("GIS Geomarketing"),
        dcc.Link("🏠 Dashboard", href="/", className="side-link"),
        html.Div("Cadeias", className="side-section"),
        *chain_links,
        html.Div("Análise espacial", className="side-section"),
        dcc.Link("📍 Interseções", href="/intersecoes", className="side-link"),
        dcc.Link("🔢 Matriz", href="/matriz", className="side-link"),
        dcc.Link("⬢ Hexbin / Densidade", href="/densidade", className="side-link"),
        dcc.Link("🎯 Concorrência", href="/concorrencia", className="side-link"),
        # Downloads desativados na versão Hugging Face Spaces
    ], className="sidebar")


def page_dashboard():
    counts = DF.groupby("cadeia").size().reset_index(name="lojas")
    fig_bar = px.bar(counts, x="cadeia", y="lojas", color="cadeia", color_discrete_map=CHAIN_COLORS, text="lojas", height=360)
    fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
    return html.Div([
        html.H1("Plataforma GIS de Geomarketing", className="page-title"),
        html.P("Distribuição, concorrência, interseções e análise espacial de redes de retalho.", className="subtitle"),
        dbc.Row([
            kpi_card("Total de lojas", f"{len(DF):,}".replace(",", ".")),
            kpi_card("Cadeias", len(CHAINS)),
            kpi_card("Com telefone", int((DF.telefone != "").sum())),
            kpi_card("Com email", int((DF.email != "").sum())),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=map_fig(DF, "Mapa geral", "cadeia", "Pontos")), className="cardx"), md=8),
            dbc.Col(html.Div(dcc.Graph(figure=fig_bar), className="cardx"), md=4),
        ], className="g-3"),
    ])


def page_chain(cadeia: str):
    df = DF[DF.cadeia == cadeia].copy()
    return html.Div([
        html.H1(cadeia, className="page-title"),
        html.P("Mapa, dados e estatísticas da cadeia selecionada.", className="subtitle"),
        dbc.Row([
            kpi_card("Lojas", len(df)),
            kpi_card("Com telefone", int((df.telefone != "").sum())),
            kpi_card("Com email", int((df.email != "").sum())),
            kpi_card("Municípios", int(df.municipio.replace('', np.nan).nunique())),
        ], className="g-3 mb-4"),
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(["Pontos", "Hexbin / Densidade", "Distância ao concorrente"], "Pontos", id="chain-map-mode", clearable=False), md=4),
            ], className="mb-3"),
            dcc.Graph(id="chain-map", figure=map_fig(df, f"Mapa - {cadeia}")),
        ], className="cardx mb-4"),
        html.Div(chain_table(df), className="cardx"),
    ])


def page_intersections():
    pair_opts = PAIR_OPTIONS
    return html.Div([
        html.H1("Interseções", className="page-title"),
        html.P("Identifica exatamente quais lojas ficam próximas entre redes. Sem heatmap borrado.", className="subtitle"),
        html.Div([
            dbc.Row([
                dbc.Col([html.Label("Par de cadeias", className="fw-bold"), dcc.Dropdown(pair_opts, "Todas", id="inter-pair", clearable=False)], md=3),
                dbc.Col([html.Label("Raio de interseção (m)", className="fw-bold"), dcc.Slider(0, 2000, value=500, marks={0:"0",50:"50",250:"250",500:"500",1000:"1 km",2000:"2 km"}, id="inter-radius")], md=7),
                dbc.Col([html.Label(" "), dbc.Input(id="inter-radius-input", value=500, type="number", min=0, max=5000)], md=2),
            ])
        ], className="cardx mb-4"),
        html.Div(id="inter-kpis", className="mb-4"),
        html.Div(dcc.Graph(id="inter-map"), className="cardx mb-4"),
        html.Div(id="inter-table-wrap", className="cardx"),
    ])


def page_matrix():
    return html.Div([
        html.H1("Matriz de interseções", className="page-title"),
        html.P("Número de pares de lojas por cadeia dentro do raio selecionado.", className="subtitle"),
        html.Div([html.Label("Raio (m)", className="fw-bold"), dcc.Slider(0, 2000, value=500, marks={0:"0",50:"50",250:"250",500:"500",1000:"1 km",2000:"2 km"}, id="matrix-radius")], className="cardx mb-4"),
        html.Div(id="matrix-table", className="cardx"),
    ])


def page_density():
    return html.Div([
        html.H1("Hexbin / Densidade", className="page-title"),
        html.P("Mapa de densidade legível. Substitui o heatmap saturado.", className="subtitle"),
        html.Div(dcc.Graph(figure=map_fig(DF, "Densidade por células", mode="Hexbin / Densidade")), className="cardx"),
    ])


def page_competition():
    chain_opts = ["Todas"] + CHAINS
    return html.Div([
        html.H1("Concorrência", className="page-title"),
        html.P("Analisa concorrentes próximos. Raio 0 m = mesmo ponto/coordenada exata.", className="subtitle"),
        html.Div([
            dbc.Row([
                dbc.Col([html.Label("Cadeia base", className="fw-bold"), dcc.Dropdown(chain_opts, "Todas", id="comp-base", clearable=False)], md=3),
                dbc.Col([html.Label("Concorrente", className="fw-bold"), dcc.Dropdown(chain_opts, "Todas", id="comp-target", clearable=False)], md=3),
                dbc.Col([html.Label("Raio (m)", className="fw-bold"), dcc.Slider(0, 2000, value=500, marks={0:"0",50:"50",250:"250",500:"500",1000:"1 km",2000:"2 km"}, id="comp-radius")], md=4),
                dbc.Col([html.Label(" "), dbc.Input(id="comp-radius-input", value=500, type="number", min=0, max=5000)], md=2),
            ])
        ], className="cardx mb-4"),
        html.Div(id="comp-kpis", className="mb-4"),
        html.Div(dcc.Graph(id="comp-map"), className="cardx mb-4"),
        html.Div(id="comp-table-wrap", className="cardx"),
    ])


def page_statistics():
    chain_opts = ["Todas"] + CHAINS
    return html.Div([
        html.H1("Estatísticas por Cadeia", className="page-title"),
        html.P("Comparativo detalhado entre cadeias, municipios e distritos.", className="subtitle"),

        # Filtros
        html.Div([
            dbc.Row([
                dbc.Col([html.Label("Cadeia", className="fw-bold"), dcc.Dropdown(chain_opts, "Todas", id="stat-chain", clearable=False)], md=3),
            ])
        ], className="cardx mb-4"),

        # KPIs
        html.Div(id="stat-kpis", className="mb-4"),

        # Gráficos
        dbc.Row([
            dbc.Col([dcc.Graph(id="stat-bar-chain")], md=6),
            dbc.Col([dcc.Graph(id="stat-bar-municipality")], md=6),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col([dcc.Graph(id="stat-bar-district")], md=12),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col([dcc.Graph(id="stat-pie-chain")], md=6),
            dbc.Col([dcc.Graph(id="stat-table-chain")], md=6),
        ], className="g-3"),
    ])


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
          suppress_callback_exceptions=True)
server = app.server
app.layout = html.Div([dcc.Location(id="url"), sidebar(), html.Main(id="page", className="main")])


@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(pathname):
    try:
        if not pathname or pathname == "/":
            return page_dashboard()
        if pathname.startswith("/cadeia/"):
            cadeia = pathname.split("/cadeia/", 1)[1].replace("%20", " ")
            return page_chain(cadeia if cadeia in CHAINS else CHAINS[0])
        if pathname == "/intersecoes":
            return page_intersections()
        if pathname == "/matriz":
            return page_matrix()
        if pathname == "/densidade":
            return page_density()
        if pathname == "/concorrencia":
            return page_competition()
        return page_dashboard()
    except Exception as e:
        return html.Div([html.H1("Erro"), html.Pre(str(e))], className="main")


@app.callback(Output("chain-map", "figure"), Input("chain-map-mode", "value"), State("url", "pathname"), prevent_initial_call=True)
def update_chain_map(mode, pathname):
    cadeia = pathname.split("/cadeia/", 1)[1].replace("%20", " ") if pathname and "/cadeia/" in pathname else CHAINS[0]
    df = DF[DF.cadeia == cadeia]
    return map_fig(df, f"Mapa - {cadeia}", mode=mode or "Pontos")


@app.callback(Output("inter-radius-input", "value"), Input("inter-radius", "value"))
def sync_radius(v):
    return v


@app.callback(
    Output("inter-kpis", "children"),
    Output("inter-map", "figure"),
    Output("inter-table-wrap", "children"),
    Input("inter-pair", "value"),
    Input("inter-radius", "value"),
)
def update_intersections(pair_value, radius):
    try:
        radius = int(radius or DEFAULT_RADIUS)
        inter = all_intersections(radius, pair_value or "Todas")
        unique_a = inter[["cadeia_a", "loja_a"]].drop_duplicates().shape[0] if not inter.empty else 0
        unique_b = inter[["cadeia_b", "loja_b"]].drop_duplicates().shape[0] if not inter.empty else 0
        avg = f"{inter.dist_m.mean():.0f} m" if not inter.empty else "—"
        kpis = dbc.Row([
            kpi_card("Pares encontrados", len(inter)),
            kpi_card("Lojas lado A", unique_a),
            kpi_card("Lojas lado B", unique_b),
            kpi_card("Distância média", avg),
        ], className="g-3")
        return kpis, inter_map(inter), intersection_table(inter)
    except Exception as e:
        kpis = dbc.Alert(f"Erro no cálculo: {e}", color="danger")
        return kpis, empty_fig("Erro ao calcular interseções"), html.Div(str(e), className="small-note")


@app.callback(Output("matrix-table", "children"), Input("matrix-radius", "value"))
def update_matrix(radius):
    try:
        radius=int(radius or DEFAULT_RADIUS)
        rows=[]
        totals={c: len(DF[DF["cadeia"]==c]) for c in CHAINS}
        for a,b in ALLOWED_INTERSECTION_PAIRS:
            inter=intersection_pairs(a,b,radius,max_rows=100000)
            if inter.empty:
                ua=ub=0
            else:
                ua=inter["row_id_a"].nunique() if "row_id_a" in inter.columns else inter[["cadeia_a","loja_a","morada_a"]].drop_duplicates().shape[0]
                ub=inter["row_id_b"].nunique() if "row_id_b" in inter.columns else inter[["cadeia_b","loja_b","morada_b"]].drop_duplicates().shape[0]
            rows.append({
                "Cadeia A":a,
                "Cadeia B":b,
                "Lojas Cadeia A":ua,
                "Lojas Cadeia B":ub,
                "% Cadeia A":f"{ua/totals[a]*100:.1f}%" if totals[a] else "0%",
                "% Cadeia B":f"{ub/totals[b]*100:.1f}%" if totals[b] else "0%",
            })
        return dash_table.DataTable(
            data=rows,
            columns=[{"name":c,"id":c} for c in rows[0].keys()],
            style_cell={"padding":"12px","textAlign":"center","fontFamily":"Inter, Segoe UI, Arial"},
            style_header={"fontWeight":"800","backgroundColor":"#f8fafc"},
        )
    except Exception as e:
        return dbc.Alert(str(e), color="danger")



@app.callback(Output("comp-radius-input", "value"), Input("comp-radius", "value"))
def sync_comp_radius(v):
    return v


@app.callback(
    Output("comp-kpis", "children"),
    Output("comp-map", "figure"),
    Output("comp-table-wrap", "children"),
    Input("comp-base", "value"),
    Input("comp-target", "value"),
    Input("comp-radius", "value"),
)
def update_competition(base_chain, competitor_chain, radius):
    try:
        radius = int(radius or 0)
        comp = competition_dataset(base_chain or "Todas", competitor_chain or "Todas", radius)
        hits = comp[comp["concorrentes_no_raio"].fillna(0).astype(int) > 0].copy()
        avg = f"{hits.dist_concorrente_m.mean():.1f} m" if not hits.empty else "—"
        mode_label = "Mesmo ponto" if radius <= 0 else f"≤ {radius} m"
        kpis = dbc.Row([
            kpi_card("Filtro", mode_label),
            kpi_card("Lojas com concorrência", len(hits)),
            kpi_card("Distância média", avg),
            kpi_card("Máx. concorrentes", int(hits.concorrentes_no_raio.max()) if not hits.empty else 0),
        ], className="g-3")
        table = chain_table(hits.sort_values(["concorrentes_no_raio", "dist_concorrente_m"], ascending=[False, True]).head(1000)) if not hits.empty else html.Div("Nenhuma loja com concorrência para os filtros selecionados.", className="small-note")
        return kpis, competition_map(comp, "Concorrência"), table
    except Exception as e:
        return dbc.Alert(f"Erro no cálculo de concorrência: {e}", color="danger"), empty_fig("Erro"), html.Div(str(e), className="small-note")


# Callback dl_csv desativado


# Callback dl_xlsx desativado


# Callbacks de Estatísticas
@app.callback(
    [
        Output("stat-kpis", "children"),
        Output("stat-bar-chain", "figure"),
        Output("stat-bar-municipality", "figure"),
        Output("stat-bar-district", "figure"),
        Output("stat-pie-chain", "figure"),
        Output("stat-table-chain", "children"),
    ],
    Input("stat-chain", "value"),
)
def update_statistics(chain_filter):
    try:
        df = DF.copy()

        # Filtrar por cadeia
        if chain_filter != "Todas":
            df = df[df["chain"] == chain_filter]

        # KPIs
        kpis = dbc.Row([
            kpi_card("Total de lojas", len(df)),
            kpi_card("Cadeia selecionada", chain_filter),
        ], className="g-3")

        # Gráfico de barras - Quantidade por cadeia (se filtro de cadeia)
        if chain_filter == "Todas":
            # Mostrar todas as cadeias
            chain_counts = df.groupby("chain").size().sort_values(ascending=False)
            fig_bar_chain = go.Figure([
                go.Bar(x=chain_counts.index, y=chain_counts.values, text=chain_counts.values, textposition="auto")
            ])
            fig_bar_chain.update_layout(
                title="Quantidade de lojas por cadeia",
                xaxis_title="Cadeia de supermercado",
                yaxis_title="Número de lojas",
                height=400,
                showlegend=False
            )
        else:
            # Mostrar apenas uma cadeia - não tem sentido gráfico
            fig_bar_chain = empty_fig("Selecione 'Todas' para ver o comparativo entre cadeias")

        # Gráfico de barras - Quantidade por município
        muni_counts = df.groupby("municipality").size().sort_values(ascending=False).head(15)
        fig_bar_muni = go.Figure([
            go.Bar(x=muni_counts.index, y=muni_counts.values, text=muni_counts.values, textposition="auto")
        ])
        fig_bar_muni.update_layout(
            title="Top 15 municípios com mais lojas",
            xaxis_title="Município",
            yaxis_title="Número de lojas",
            height=400,
            showlegend=False,
            xaxis_tickangle=-45
        )

        # Gráfico de barras - Quantidade por distrito
        district_counts = df.groupby("district").size().sort_values(ascending=False).head(15)
        fig_bar_district = go.Figure([
            go.Bar(x=district_counts.index, y=district_counts.values, text=district_counts.values, textposition="auto")
        ])
        fig_bar_district.update_layout(
            title="Top 15 distritos com mais lojas",
            xaxis_title="Distrito",
            yaxis_title="Número de lojas",
            height=400,
            showlegend=False,
            xaxis_tickangle=-45
        )

        # Gráfico de pizza - Distribuição por cadeia (se filtro for 'Todas')
        if chain_filter == "Todas":
            chain_pie_counts = df.groupby("chain").size()
            fig_pie_chain = go.Figure([
                go.Pie(labels=chain_pie_counts.index, values=chain_pie_counts.values, hole=0.4)
            ])
            fig_pie_chain.update_layout(
                title="Distribuição de lojas por cadeia",
                height=400,
                showlegend=True
            )
        else:
            fig_pie_chain = empty_fig("Selecione 'Todas' para ver a distribuição por cadeias")

        # Tabela - Comparativo por cadeia (se filtro for 'Todas')
        if chain_filter == "Todas":
            chain_stats = df.groupby("chain").agg({
                "name": "count",
                "municipality": lambda x: f"{x.nunique()} municípios",
                "district": lambda x: f"{x.nunique()} distritos"
            }).rename(columns={"name": "Total de lojas"})
            chain_stats.index.name = "Cadeia"
            chain_stats.reset_index(inplace=True)

            table = html.Div([
                html.H5("Comparativo por cadeia:", className="mt-3 mb-2"),
                dash_table.DataTable(
                    data=chain_stats.to_dict('records'),
                    columns=[{"name": c, "id": c} for c in chain_stats.columns],
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "rgb(230, 230, 230)",
                        "fontWeight": "bold"
                    },
                    style_cell={
                        "minWidth": "100px",
                        "width": "auto",
                        "textAlign": "left"
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(245, 245, 245)"
                        }
                    ]
                )
            ])
        else:
            table = html.Div([
                html.H5("Detalhes da cadeia:", className="mt-3 mb-2"),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": c, "id": c} for c in df.columns],
                    page_size=50,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "rgb(230, 230, 230)",
                        "fontWeight": "bold"
                    },
                    style_cell={
                        "minWidth": "100px",
                        "width": "auto",
                        "textAlign": "left"
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(245, 245, 245)"
                        }
                    ]
                )
            ])

        return kpis, fig_bar_chain, fig_bar_muni, fig_bar_district, fig_pie_chain, table

    except Exception as e:
        return dbc.Alert(f"Erro no cálculo de estatísticas: {e}", color="danger"), \
               empty_fig("Erro"), empty_fig("Erro"), empty_fig("Erro"), empty_fig("Erro"), \
               html.Div(str(e), className="small-note")



# Callback download_intersections desativado


# ── Arranque para Hugging Face Spaces ──
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False)