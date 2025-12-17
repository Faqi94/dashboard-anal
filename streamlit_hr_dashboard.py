import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
# =============================
# PAGE CONFIG + THEME
# =============================
st.set_page_config(page_title="Byru Decision System", layout="wide")

BYRU_BLUE = "#0ea5e9"
BYRU_ORANGE = "#f97316"
BYRU_DARK = "#0b1220"
BYRU_MUTED = "#64748b"
BG = "#f8fafc"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;600;700&display=swap');
html, body, [class*="css"] {{
  font-family: 'Comfortaa', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
  background: {BG};
}}
.byru-header {{
  background: linear-gradient(90deg, {BYRU_BLUE}, {BYRU_ORANGE});
  padding: 18px;
  border-radius: 18px;
  color: white;
  box-shadow: 0 10px 28px rgba(2, 132, 199, 0.18);
  margin-bottom: 14px;
}}
.byru-header h1 {{
  font-size: 26px; margin: 0; line-height: 1.1; font-weight: 800;
}}
.byru-header p {{
  margin: 6px 0 0 0; opacity: .95; font-size: 13px;
}}
.kpi {{
  background: rgba(255,255,255,0.97);
  border: 1px solid rgba(2, 132, 199, 0.18);
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 10px 24px rgba(2, 132, 199, 0.10);
  height: 100%;
}}
.kpi-title {{ font-size: 12px; color: {BYRU_MUTED}; font-weight: 800; }}
.kpi-value {{ font-size: 24px; font-weight: 900; color: {BYRU_DARK}; margin-top: 6px; }}
.kpi-sub {{ font-size: 12px; color: {BYRU_MUTED}; margin-top: 6px; line-height: 1.35; }}
.pill {{
  font-size: 11px; font-weight: 900; padding: 6px 10px; border-radius: 999px;
  background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.25); color: {BYRU_ORANGE};
  white-space: nowrap;
}}
.hrline {{ height:1px; background: rgba(2,132,199,.12); border:none; margin: 12px 0; }}
.smallnote {{ color: {BYRU_MUTED}; font-size: 12px; }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="byru-header">
  <h1>Byru Decision System Dashboard</h1>
  <p>Multi-source analytics: Rekap Perusahaan + Dataset Karyawan. Fokus: Growth, Churn, Adoption, Engagement, Compliance, Health Score.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =============================
# HELPERS
# =============================
def fmt_int_dot(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return ""

def fmt_rp_dot(x):
    try:
        return "Rp " + f"{float(x):,.0f}".replace(",", ".")
    except Exception:
        return "Rp 0"

def apply_value_labels(
    fig,
    show=True,
    currency_trace_names=("Nominal", "NOMINAL", "Total Kasbon"),
    integer_trace_names=("Transaksi", "TRX", "COUNT", "USER_UNIK", "COMPANY_UNIK"),
    bar_position="outside",
    scatter_position="top center",
    max_labels=9999,  # batasi kalau terlalu rame
):
    """
    Auto-label untuk Plotly figure:
    - Bar: text di atas bar (outside)
    - Line/Scatter: text di atas titik (top center)
    - Bisa dipakai untuk px.* maupun make_subplots (combo).
    """

    if not show:
        # hapus text kalau dimatikan
        for tr in fig.data:
            tr.text = None
            if getattr(tr, "type", "") == "scatter":
                # kembalikan mode tanpa text
                if hasattr(tr, "mode") and tr.mode:
                    tr.mode = tr.mode.replace("+text", "").replace("text+", "")
        return fig

    for tr in fig.data:
        ttype = getattr(tr, "type", "")
        name = (getattr(tr, "name", "") or "").strip()

        # ambil y values
        y = getattr(tr, "y", None)
        if y is None:
            continue

        # limit label biar ga overload
        if len(y) > max_labels:
            continue

        # tentukan formatter
        if name in currency_trace_names:
            txt = [fmt_rp_dot(v) for v in y]
        elif name in integer_trace_names:
            txt = [fmt_int_dot(v) for v in y]
        else:
            # default: integer
            txt = [fmt_int_dot(v) for v in y]

        if ttype == "bar":
            tr.text = txt
            tr.textposition = bar_position
        elif ttype == "scatter":
            # pastikan mode ada text
            mode = getattr(tr, "mode", "lines+markers")
            if "text" not in mode:
                tr.mode = mode + "+text"
            tr.text = txt
            tr.textposition = scatter_position

    # supaya text tidak saling tabrakan parah
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
    return fig

def format_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "0"

def format_pct(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):.1f}%"
    except Exception:
        return "—"

def kpi_card(title, value, subtitle="", pill=""):
    pill_html = f'<span class="pill">{pill}</span>' if pill else ""
    st.markdown(
        f"""
<div class="kpi">
  <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
    <div class="kpi-title">{title}</div>
    {pill_html}
  </div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================
# ICONS
# =============================
EWA_ICON_SVG = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
      <stop stop-color="#0ea5e9"/>
      <stop offset="1" stop-color="#f97316"/>
    </linearGradient>
  </defs>
  <path d="M3.5 7.5C3.5 5.843 4.843 4.5 6.5 4.5H17.5C19.157 4.5 20.5 5.843 20.5 7.5V16.5C20.5 18.157 19.157 19.5 17.5 19.5H6.5C4.843 19.5 3.5 18.157 3.5 16.5V7.5Z" stroke="url(#g)" stroke-width="1.8"/>
  <path d="M3.8 8.2H16.8C18.3 8.2 19.2 9.1 19.2 10.6V13.4C19.2 14.9 18.3 15.8 16.8 15.8H3.8" stroke="url(#g)" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M16.9 11.2H19.2" stroke="url(#g)" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M12.2 9.6L10.1 13.2H12.1L11 16.4L13.7 12.3H11.8L12.2 9.6Z" fill="url(#g)"/>
</svg>"""


def norm_text(s: pd.Series) -> pd.Series:
    return (s.astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())

def company_key(series: pd.Series) -> pd.Series:
    return norm_text(series).str.upper()

def clean_dates(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = series.copy()
        try:
            dt = dt.mask(dt.dt.date == datetime(1970, 1, 1).date())
        except Exception:
            pass
        return dt

    s = norm_text(series)
    s = s.replace({
        "00/00/0000": np.nan,
        "0000-00-00": np.nan,
        "0000/00/00": np.nan,
        "00-00-0000": np.nan,
        "1970-01-01": np.nan,
        "None": np.nan,
        "nan": np.nan,
        "NaT": np.nan,
        "": np.nan
    })

    # Rekap banyak dd/mm/yyyy; HR bisa yyyy-mm-dd atau mm/dd
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.notna().mean() < 0.5:
        dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False)
        if dt2.notna().mean() > dt.notna().mean():
            dt = dt2

    # Handle special if string has double spaces like "11/21/2023  11:26:14"
    if dt.notna().mean() < 0.5:
        s2 = s.str.replace(r"\s{2,}", " ", regex=True)
        dt3 = pd.to_datetime(s2, errors="coerce", infer_datetime_format=True, dayfirst=False)
        if dt3.notna().mean() > dt.notna().mean():
            dt = dt3

    try:
        dt = dt.mask(dt.dt.date == datetime(1970, 1, 1).date())
    except Exception:
        pass
    return dt

def is_missing_text(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([True] * 0)
    x = norm_text(s)
    return s.isna() | (x == "") | (x.str.lower().isin(["nan", "none", "null", "-"]))

# =============================
# DATA HUB (UPLOAD)
# =============================
st.sidebar.header("📦 Data Hub")

emp_file = st.sidebar.file_uploader("Upload Dataset Karyawan - Dashboard (CSV/XLSX) — header=1", type=["csv", "xlsx"], key="emp")
rekap_file = st.sidebar.file_uploader("Upload Rekap Perusahaan - Dashboard (XLSX)", type=["xlsx"], key="rekap")
ewa_file = st.sidebar.file_uploader("📥 Upload Data EWA/PPOB (Kasbon Vinjer) .xlsx", type=["xlsx"], key="ewa_file")

st.sidebar.markdown('<hr class="hrline"/>', unsafe_allow_html=True)
st.sidebar.caption("Default filter: ALL perusahaan & ALL karyawan. Kamu bisa filter sendiri di sidebar.")

now = pd.Timestamp.now()
today = now.normalize()

# =============================
# LOADER: EMPLOYEE DATASET
# =============================
@st.cache_data(show_spinner=False)
def load_employee(file) -> tuple[pd.DataFrame, dict]:
    if file is None:
        return None, {}

    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file, header=1)
    else:
        df = pd.read_excel(file, header=1)

    df.columns = [str(c).strip() for c in df.columns]

    # ===== FIXED mapping (sesuai file kamu)
    # Unnamed: 1  = PERUSAHAAN
    # Unnamed: 37 = NO_BPJSTK
    # Unnamed: 38 = MASA_KERJA (string)
    # Unnamed: 39 = LOGIN_TERAKHIR
    # Unnamed: 40 = TANGGAL_DIBUAT
    # Unnamed: 41 = TANGGAL_NONAKTIF
    # Unnamed: 42 = STATUS
    # + rename critical yang dulu
    rename_map = {
        "Unnamed: 1": "PERUSAHAAN",
        "Unnamed: 37": "NO_BPJSTK",
        "Unnamed: 38": "MASA_KERJA",
        "Unnamed: 39": "LOGIN_TERAKHIR",
        "Unnamed: 40": "TANGGAL_DIBUAT",
        "Unnamed: 41": "TANGGAL_NONAKTIF",
        "Unnamed: 42": "STATUS",

        "Unnamed: 34": "STATUS_KEPEGAWAIAN",
        "Unnamed: 35": "NPWP",
        "Unnamed: 36": "BPJS_KES",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Detect optional columns (if exist)
    cols = {c.upper(): c for c in df.columns}
    col_gender = cols.get("GENDER") or cols.get("JENIS KELAMIN") or cols.get("JK")
    col_dept = cols.get("DEPARTEMEN") or cols.get("DEPARTMENT") or cols.get("DEPT")
    col_job = cols.get("JABATAN") or cols.get("POSITION") or cols.get("ROLE") or cols.get("TITLE")
    col_join = cols.get("JOIN DATE") or cols.get("TANGGAL MASUK") or cols.get("HIRE DATE")
    col_end = cols.get("END DATE") or cols.get("TANGGAL KELUAR") or cols.get("RESIGN DATE")
    col_dob = cols.get("TANGGAL LAHIR") or cols.get("DOB") or cols.get("DATE OF BIRTH")
    col_bank = cols.get("BANK") or cols.get("NAMA BANK") or cols.get("BANK NAME")
    col_name = cols.get("NAMA") or cols.get("NAMA KARYAWAN") or cols.get("EMPLOYEE NAME") or cols.get("FULL NAME") or cols.get("NAME")

    # Standardize text
    if "PERUSAHAAN" in df.columns:
        df["PERUSAHAAN"] = norm_text(df["PERUSAHAAN"])
    if col_dept and col_dept in df.columns:
        df[col_dept] = norm_text(df[col_dept]).str.upper()
    if col_job and col_job in df.columns:
        df[col_job] = norm_text(df[col_job])

    # Standardize dates
    for c in [col_join, col_end, col_dob, "LOGIN_TERAKHIR", "TANGGAL_DIBUAT", "TANGGAL_NONAKTIF"]:
        if c and c in df.columns:
            df[c] = clean_dates(df[c])

    # Active flag (lebih akurat):
    # STATUS "AKTIF" => aktif, selain itu nonaktif. Fallback ke TANGGAL_NONAKTIF jika STATUS kosong.
    if "STATUS" in df.columns:
        s = norm_text(df["STATUS"]).str.upper()
        df["_IS_AKTIF"] = s.eq("AKTIF")
    elif "TANGGAL_NONAKTIF" in df.columns:
        df["_IS_AKTIF"] = df["TANGGAL_NONAKTIF"].isna()
    else:
        df["_IS_AKTIF"] = True

    # Tenure months (MASA_KERJA_BULAN_CALC)
    df["MASA_KERJA_BULAN_CALC"] = np.nan
    if col_join and col_join in df.columns:
        # aktif
        m_active = df["_IS_AKTIF"] & df[col_join].notna()
        df.loc[m_active, "MASA_KERJA_BULAN_CALC"] = (today - df.loc[m_active, col_join]).dt.days / 30.0
        # nonaktif
        if col_end and col_end in df.columns:
            m_inactive = (~df["_IS_AKTIF"]) & df[col_join].notna() & df[col_end].notna()
            df.loc[m_inactive, "MASA_KERJA_BULAN_CALC"] = (df.loc[m_inactive, col_end] - df.loc[m_inactive, col_join]).dt.days / 30.0
        else:
            # fallback: pakai TANGGAL_NONAKTIF
            if "TANGGAL_NONAKTIF" in df.columns:
                m_inactive = (~df["_IS_AKTIF"]) & df[col_join].notna() & df["TANGGAL_NONAKTIF"].notna()
                df.loc[m_inactive, "MASA_KERJA_BULAN_CALC"] = (df.loc[m_inactive, "TANGGAL_NONAKTIF"] - df.loc[m_inactive, col_join]).dt.days / 30.0

    meta = dict(
        col_gender=col_gender, col_dept=col_dept, col_job=col_job,
        col_join=col_join, col_end=col_end, col_dob=col_dob,
        col_bank=col_bank, col_name=col_name
    )
    return df, meta

def _fmt_rp(x):
    try:
        return f"Rp {float(x):,.0f}".replace(",", ".")
    except Exception:
        return "Rp 0"

def _fmt_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "0"

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

import pandas as pd
import numpy as np
import re

# =========================
# Robust Employee Loader
# =========================
EMP_REQUIRED_HEADERS = [
    "JOIN DATE", "END DATE", "TANGGAL LAHIR", "DEPARTEMEN"
]

# mapping rename dari "unnamed" atau variasi penamaan
EMP_RENAME_BY_NAME = {
    "UNNAMED: 34": "STATUS_KEPEGAWAIAN",
    "UNNAMED 34": "STATUS_KEPEGAWAIAN",
    "UNNAMED: 35": "NPWP",
    "UNNAMED 35": "NPWP",
    "UNNAMED: 36": "BPJS_KES",
    "UNNAMED 36": "BPJS_KES",
    "UNNAMED: 42": "STATUS",
    "UNNAMED 42": "STATUS",
    # tambahan dari info kamu
    "UNNAMED 1": "PERUSAHAAN",
    "UNNAMED: 1": "PERUSAHAAN",
    "UNNAMED 37": "NO. BPJSTK",
    "UNNAMED: 37": "NO. BPJSTK",
    "UNNAMED 40": "TANGGAL DIBUAT",
    "UNNAMED: 40": "TANGGAL DIBUAT",
    "UNNAMED 41": "TANGGAL NONAKTIF",
    "UNNAMED: 41": "TANGGAL NONAKTIF",
}

EMP_RENAME_BY_INDEX_0BASED = {
    1: "PERUSAHAAN",        # kolom ke-2
    34: "STATUS_KEPEGAWAIAN",
    35: "NPWP",
    36: "BPJS_KES",
    37: "NO. BPJSTK",
    40: "TANGGAL DIBUAT",
    41: "TANGGAL NONAKTIF",
    42: "STATUS",
    43: "IS AKTIF",
    44: "MASA KERJA BULAN",  # kalau memang ada kolom ke-45 (hati-hati bila total 44)
}

def _norm_col(x: str) -> str:
    x = "" if x is None else str(x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x.upper()

def _find_header_row_from_preview(preview_df: pd.DataFrame, required_headers=EMP_REQUIRED_HEADERS, min_hits=2):
    """Cari baris mana yang paling mirip header dengan menghitung overlap header penting."""
    best_row = 0
    best_score = -1
    req = set(_norm_col(h) for h in required_headers)

    for r in range(min(len(preview_df), 30)):
        row_vals = [_norm_col(v) for v in preview_df.iloc[r].tolist()]
        row_set = set([v for v in row_vals if v and v != "NAN"])
        score = len(req.intersection(row_set))
        if score > best_score:
            best_score = score
            best_row = r

    # kalau tidak ketemu yang meyakinkan, fallback ke header row 1 (sesuai instruksi kamu)
    if best_score < min_hits:
        return 1, best_score
    return best_row, best_score

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)  # kamu punya format campur; ini aman
    # paksa '1970-01-01' dan error sistem jadi NaT
    dt = dt.mask(dt == pd.Timestamp("1970-01-01"))
    return dt

def load_employee_data(uploaded_file, force_header_row=None):
    """
    Robust loader untuk file karyawan:
    - otomatis deteksi header row (kalau force_header_row None)
    - normalize nama kolom (uppercase + trim)
    - rename Unnamed by NAME dan fallback by INDEX
    - standardisasi tanggal + departemen uppercase + masa kerja bulan
    """
    info = {"header_row": None, "header_score": None, "source_type": None, "columns_before": None, "columns_after": None}

    if uploaded_file is None:
        return None, info

    name = getattr(uploaded_file, "name", "")
    is_csv = str(name).lower().endswith(".csv")

    # --- preview dulu tanpa header untuk deteksi baris header
    if is_csv:
        info["source_type"] = "csv"
        preview = pd.read_csv(uploaded_file, header=None, nrows=25, dtype=str, encoding_errors="ignore")
        header_row, score = (force_header_row, None) if force_header_row is not None else _find_header_row_from_preview(preview)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=header_row, dtype=str, encoding_errors="ignore")
    else:
        info["source_type"] = "excel"
        preview = pd.read_excel(uploaded_file, header=None, nrows=25, dtype=str)
        header_row, score = (force_header_row, None) if force_header_row is not None else _find_header_row_from_preview(preview)
        df = pd.read_excel(uploaded_file, header=header_row, dtype=str)

    info["header_row"] = header_row
    info["header_score"] = score

    # --- normalize column names
    cols_before = list(df.columns)
    df.columns = [_norm_col(c) for c in df.columns]
    info["columns_before"] = cols_before

    # --- rename by NAME (unnamed)
    rename_map = {}
    for c in df.columns:
        nc = _norm_col(c)
        if nc in EMP_RENAME_BY_NAME:
            rename_map[c] = EMP_RENAME_BY_NAME[nc]
    df = df.rename(columns=rename_map)

    # --- fallback rename by INDEX kalau masih belum ada kolom penting
    # (berguna kalau "Unnamed: xx" berubah labelnya tapi posisinya tetap)
    cols = list(df.columns)
    for idx0, newname in EMP_RENAME_BY_INDEX_0BASED.items():
        if idx0 < len(cols):
            old = cols[idx0]
            # jangan overwrite kalau target sudah ada
            if newname not in df.columns and (old.startswith("UNNAMED") or old.strip() == "" or old == "NAN"):
                df = df.rename(columns={old: newname})

    # --- standardisasi tanggal (kalau kolom ada)
    for dc in ["JOIN DATE", "END DATE", "TANGGAL LAHIR", "TANGGAL DIBUAT", "TANGGAL NONAKTIF"]:
        if dc in df.columns:
            df[dc] = _safe_to_datetime(df[dc])
            # handle '0000-00-00' atau tanggal “default sistem” -> NaT (sudah ke-coerce)
            df.loc[df[dc].astype(str).str.contains("0000-00-00", na=False), dc] = pd.NaT

    # --- departemen uppercase
    if "DEPARTEMEN" in df.columns:
        df["DEPARTEMEN"] = df["DEPARTEMEN"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})

    # --- status aktif: coba ambil dari IS AKTIF atau STATUS
    if "IS AKTIF" in df.columns:
        isaktif = df["IS AKTIF"].astype(str).str.strip().str.upper()
        aktif_mask = isaktif.isin(["1", "TRUE", "YA", "Y", "AKTIF", "ACTIVE"])
    elif "STATUS" in df.columns:
        stt = df["STATUS"].astype(str).str.strip().str.upper()
        aktif_mask = stt.isin(["AKTIF", "ACTIVE"])
    else:
        aktif_mask = pd.Series([True] * len(df), index=df.index)

    # --- MASA KERJA BULAN (recompute supaya konsisten)
    today = pd.Timestamp.today().normalize()
    if "JOIN DATE" in df.columns:
        join = df["JOIN DATE"]
        end = df["END DATE"] if "END DATE" in df.columns else pd.Series([pd.NaT]*len(df), index=df.index)

        masa = pd.Series([np.nan] * len(df), index=df.index, dtype="float")
        # aktif: today - join
        masa.loc[aktif_mask & join.notna()] = (today - join.loc[aktif_mask & join.notna()]).dt.days / 30.0
        # tidak aktif: end - join
        masa.loc[(~aktif_mask) & join.notna() & end.notna()] = (end.loc[(~aktif_mask) & join.notna() & end.notna()] - join.loc[(~aktif_mask) & join.notna() & end.notna()]).dt.days / 30.0

        df["MASA_KERJA_BULAN"] = masa.round(2)

    info["columns_after"] = list(df.columns)
    return df, info


def render_ewa_page(ewa_file, company_master=None):
    st.markdown(f"## {EWA_ICON_SVG} EWA & PPOB Analytics (Transaction-level)", unsafe_allow_html=True)
    st.caption("Upload file kasbon/EWA/PPOB. Visualisasi pakai Plotly + filter interaktif.")

    if ewa_file is None:
        st.info("Silakan upload file EWA/PPOB (.xlsx) dulu via sidebar.")
        return

    try:
        df = pd.read_excel(ewa_file)
    except Exception as e:
        st.error(f"Gagal membaca file EWA/PPOB: {e}")
        return

    # ---- Required columns (mengikuti pola app kamu)
    req = ["Tanggal Approved", "Username/ ID User", "Total Kasbon"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(miss)}")
        st.write("Kolom yang ada:", list(df.columns))
        return

    # ---- Clean
    df = df.copy()
    df["Tanggal Approved"] = pd.to_datetime(df["Tanggal Approved"], errors="coerce")
    df = df[df["Tanggal Approved"].notna()].copy()
    if df.empty:
        st.warning("Semua baris punya Tanggal Approved tidak valid.")
        return

    df["Total Kasbon"] = pd.to_numeric(df["Total Kasbon"], errors="coerce").fillna(0.0)
    df["Hari"] = df["Tanggal Approved"].dt.day_name()
    df["Date"] = df["Tanggal Approved"].dt.date
    df["Month"] = df["Tanggal Approved"].dt.to_period("M").astype(str)

    # company col (fleksibel)
    company_col = pick_col(df, ["Nama Perusahaan", "Nama Perushaan", "Company", "Nama Company", "PERUSAHAAN"])
    if company_col is None:
        df["PERUSAHAAN_CLEAN"] = "UNKNOWN"
    else:
        df["PERUSAHAAN_CLEAN"] = (
            df[company_col].astype(str).str.strip().str.upper().replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN"})
        )

    # jenis col (fleksibel) mengikuti pola kamu
    jenis_col = pick_col(df, ["Jenis EWA", "JENIS EWA", "Jenis", "JENIS", "Jenis Transaksi", "Jenis_Kasbon"])
    if jenis_col:
        df["JENIS_CLEAN"] = df[jenis_col].astype(str).str.strip().str.upper()
    else:
        df["JENIS_CLEAN"] = "GABUNGAN"

    # ---------- Filters (default = SEMUA)
    min_date = df["Tanggal Approved"].min().date()
    max_date = df["Tanggal Approved"].max().date()

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with c1:
        date_range = st.date_input("Periode", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d1, d2 = date_range
        else:
            d1, d2 = min_date, max_date

    with c2:
        jenis_pick = st.selectbox("Segmen", ["GABUNGAN", "EWA", "PPOB"], index=0)

    with c3:
        only_weekend = st.toggle("Weekend saja (Sabtu/Minggu)", value=False)

    with c4:
        top_n = st.slider("Top N", 5, 30, 10)

    comp_list = sorted(df["PERUSAHAAN_CLEAN"].dropna().unique().tolist())
    pick_companies = st.multiselect("Filter Perusahaan (default: semua)", options=comp_list, default=[])

    # apply filters
    view = df[(df["Tanggal Approved"].dt.date >= d1) & (df["Tanggal Approved"].dt.date <= d2)].copy()

    if jenis_pick != "GABUNGAN":
        view = view[view["JENIS_CLEAN"] == jenis_pick].copy()

    if only_weekend:
        view = view[view["Hari"].isin(["Saturday", "Sunday"])].copy()

    if pick_companies:
        view = view[view["PERUSAHAAN_CLEAN"].isin([x.upper().strip() for x in pick_companies])].copy()

    if view.empty:
        st.warning("Hasil filter kosong.")
        return

    # ---------- KPIs
    total_amt = float(view["Total Kasbon"].sum())
    total_trx = int(len(view))
    uniq_user = int(view["Username/ ID User"].nunique())
    uniq_comp = int(view["PERUSAHAAN_CLEAN"].nunique())
    avg_ticket = float(view["Total Kasbon"].mean()) if total_trx else 0.0

    # optional profit/admin if exist
    admin_col = pick_col(view, ["Biaya Admin", "ADMIN FEE", "Admin Fee", "Biaya Admin (Rp)"])
    profit_col = pick_col(view, ["Profit", "PROFIT", "Laba", "Margin"])
    transfer_col = pick_col(view, ["Biaya Transfer", "BIAYA TRANSFER", "Transfer Fee", "TRANSFER FEE", "Fee Transfer"])

    total_admin = float(pd.to_numeric(view[admin_col], errors="coerce").fillna(0).sum()) if admin_col else None
    total_profit = float(pd.to_numeric(view[profit_col], errors="coerce").fillna(0).sum()) if profit_col else None
    total_transfer = float(pd.to_numeric(view[transfer_col], errors="coerce").fillna(0).sum()) if transfer_col else None


    k2, k3, k4, k5 = st.columns(4)
    k2.metric("Total Transaksi", _fmt_int(total_trx))
    k3.metric("User Unik", _fmt_int(uniq_user))
    k4.metric("Company Unik", _fmt_int(uniq_comp))
    k5.metric("Avg Trx", _fmt_rp(avg_ticket))

    a0, a1 = st.columns(2)
    a0.metric("Total Nominal", _fmt_rp(total_amt))
    a1.metric("Total Biaya Transfer", _fmt_rp(total_transfer))
    if total_profit is not None or total_admin is not None:
        a1, a2 = st.columns(2)
        if total_profit is not None:
            a1.metric("Total Profit", _fmt_rp(total_profit))
        if total_admin is not None:
            a2.metric("Total Admin Fee", _fmt_rp(total_admin))
    st.markdown("---")
    
    # ---------- Trend Monthly: Nominal + Trx
    monthly = (
        view.groupby("Month")
        .agg(NOMINAL=("Total Kasbon", "sum"), TRX=("Total Kasbon", "size"),
             USER_UNIK=("Username/ ID User", "nunique"),
             COMPANY_UNIK=("PERUSAHAAN_CLEAN", "nunique"))
        .reset_index()
        .sort_values("Month")
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["NOMINAL"], name="Nominal"), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["TRX"], name="Transaksi", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title="Trend Bulanan — Nominal vs Transaksi", height=420, margin=dict(l=10,r=10,t=60,b=10))
    fig.update_yaxes(title_text="Nominal", secondary_y=False)
    fig.update_yaxes(title_text="Transaksi", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    fig_uc = px.line(
        monthly, x="Month", y=["USER_UNIK", "COMPANY_UNIK"],
        markers=True, title="Trend Bulanan — User Unik & Company Unik"
    )
    fig_uc.update_layout(height=380, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_uc, use_container_width=True)

    # ---------- Weekend vs Weekday contribution (mengikuti konsep weekend contribution)
    weekend_mask = view["Hari"].isin(["Saturday", "Sunday"])
    wk_amt = float(view.loc[weekend_mask, "Total Kasbon"].sum())
    wd_amt = float(view.loc[~weekend_mask, "Total Kasbon"].sum())

    donut = px.pie(
        pd.DataFrame({"Bucket": ["Weekday", "Weekend"], "Nominal": [wd_amt, wk_amt]}),
        names="Bucket", values="Nominal", hole=0.55,
        title="Kontribusi Nominal — Weekday vs Weekend"
    )
    donut.update_layout(height=360, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(donut, use_container_width=True)

    st.markdown("---")

    # ---------- Top Company & Top User
    left, right = st.columns(2)

    with left:
        st.markdown("### 🏢 Top Company")
        top_company = (
            view.groupby("PERUSAHAAN_CLEAN")
            .agg(NOMINAL=("Total Kasbon", "sum"), TRX=("Total Kasbon", "size"), USER_UNIK=("Username/ ID User", "nunique"))
            .reset_index()
            .sort_values("NOMINAL", ascending=False)
            .head(top_n)
        )

        fig_tc = px.bar(
            top_company.sort_values("NOMINAL"),
            x="NOMINAL", y="PERUSAHAAN_CLEAN", orientation="h",
            title=f"Top {top_n} Company — by Nominal"
        )
        fig_tc.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_tc, use_container_width=True)

        top_company_disp = top_company.copy()
        top_company_disp["NOMINAL"] = top_company_disp["NOMINAL"].apply(_fmt_rp)
        st.dataframe(top_company_disp, use_container_width=True, height=260)

        st.download_button(
            "⬇️ Download Top Company (CSV)",
            data=top_company.to_csv(index=False).encode("utf-8"),
            file_name="top_company_ewa.csv",
            mime="text/csv",
            use_container_width=True
        )

    with right:
        st.markdown("### 👤 Top User")
        top_user = (
            view.groupby("Username/ ID User")
            .agg(NOMINAL=("Total Kasbon", "sum"), TRX=("Total Kasbon", "size"),
                 COMPANY=("PERUSAHAAN_CLEAN", "nunique"))
            .reset_index()
            .sort_values("NOMINAL", ascending=False)
            .head(top_n)
        )

        fig_tu = px.bar(
            top_user.sort_values("NOMINAL"),
            x="NOMINAL", y="Username/ ID User", orientation="h",
            title=f"Top {top_n} User — by Nominal"
        )
        fig_tu.update_layout(height=520, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_tu, use_container_width=True)

        top_user_disp = top_user.copy()
        top_user_disp["NOMINAL"] = top_user_disp["NOMINAL"].apply(_fmt_rp)
        st.dataframe(top_user_disp, use_container_width=True, height=260)

        st.download_button(
            "⬇️ Download Top User (CSV)",
            data=top_user.to_csv(index=False).encode("utf-8"),
            file_name="top_user_ewa.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # ---------- New Users Growth (first transaction) — Company-level
    st.markdown("### 📈 New Users Growth (berdasarkan transaksi pertama)")
    first_tx = (
        view.groupby(["PERUSAHAAN_CLEAN", "Username/ ID User"])["Tanggal Approved"]
        .min()
        .reset_index()
    )
    first_tx["FirstMonth"] = first_tx["Tanggal Approved"].dt.to_period("M").astype(str)

    new_users_month = (
        first_tx.groupby(["FirstMonth", "PERUSAHAAN_CLEAN"])
        .size()
        .reset_index(name="NEW_USERS")
        .sort_values("FirstMonth")
    )

    # top companies by new users in selected period
    top_new = (
        new_users_month.groupby("PERUSAHAAN_CLEAN")["NEW_USERS"].sum()
        .sort_values(ascending=False)
        .head(min(10, new_users_month["PERUSAHAAN_CLEAN"].nunique()))
        .index.tolist()
    )
    new_users_focus = new_users_month[new_users_month["PERUSAHAAN_CLEAN"].isin(top_new)].copy()

    fig_new = px.line(
        new_users_focus, x="FirstMonth", y="NEW_USERS", color="PERUSAHAAN_CLEAN",
        markers=True, title="New Users per Bulan — Top Company (by New Users)"
    )
    fig_new.update_layout(height=480, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig_new, use_container_width=True)

    st.dataframe(
        new_users_month.sort_values(["FirstMonth","NEW_USERS"], ascending=[True, False]),
        use_container_width=True,
        height=260
    )

    st.markdown("---")

    # ---------- Optional: Join ke Company Master (Renewal impact)
    if company_master is not None and isinstance(company_master, pd.DataFrame) and len(company_master):
        st.markdown("### 🔗 Renewal Impact (EWA x SUBS_END)")
        cm = company_master.copy()

        # cari kolom company & subs_end di rekap
        cm_company_col = pick_col(cm, ["PERUSAHAAN", "COMPANY_NAME", "Nama Perusahaan", "Nama Perushaan"])
        cm_subs_end = pick_col(cm, ["SUBS_END", "AKHIR BERLANGGANAN", "SUBSCRIPTION END", "TANGGAL EXPIRED", "END DATE"])

        if cm_company_col and cm_subs_end:
            cm["PERUSAHAAN_CLEAN"] = cm[cm_company_col].astype(str).str.strip().str.upper()
            cm[cm_subs_end] = pd.to_datetime(cm[cm_subs_end], errors="coerce")
            cm["DAYS_TO_END"] = (cm[cm_subs_end] - pd.Timestamp.today().normalize()).dt.days

            # last 30d EWA nominal per company (from filtered view)
            last30_start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=30))
            v30 = view[view["Tanggal Approved"] >= last30_start].groupby("PERUSAHAAN_CLEAN")["Total Kasbon"].sum().reset_index(name="EWA_30D_NOMINAL")

            join = cm.merge(v30, on="PERUSAHAAN_CLEAN", how="left")
            join["EWA_30D_NOMINAL"] = join["EWA_30D_NOMINAL"].fillna(0.0)

            # expiring soon with impact
            soon = join[(join["DAYS_TO_END"].notna()) & (join["DAYS_TO_END"] >= 0) & (join["DAYS_TO_END"] <= 30)].copy()
            soon = soon.sort_values(["DAYS_TO_END", "EWA_30D_NOMINAL"], ascending=[True, False])

            show_cols = [c for c in [cm_company_col, cm_subs_end, "DAYS_TO_END", "EWA_30D_NOMINAL"] if c in soon.columns]
            if len(soon):
                tmp = soon[show_cols].copy()
                if "EWA_30D_NOMINAL" in tmp.columns:
                    tmp["EWA_30D_NOMINAL"] = tmp["EWA_30D_NOMINAL"].apply(_fmt_rp)
                st.dataframe(tmp, use_container_width=True, height=320)
            else:
                st.info("Tidak ada perusahaan yang akan expired ≤ 30 hari (berdasarkan data rekap yang terbaca).")
        else:
            st.info("Company master ada, tapi kolom PERUSAHAAN/SUBS_END belum terdeteksi.")


# =============================
# LOADER: REKAP PERUSAHAAN
# =============================
@st.cache_data(show_spinner=False)
def load_rekap(file) -> pd.DataFrame:
    if file is None:
        return None

    raw = pd.read_excel(file, sheet_name=0, header=None)

    hdr_idx = None
    for i in range(min(60, len(raw))):
        v0 = str(raw.iloc[i, 0]).strip().upper()
        v1 = str(raw.iloc[i, 1]).strip().upper() if raw.shape[1] > 1 else ""
        if v0 == "NO" and ("ID" in v1 or "PERUSAHAAN" in v1):
            hdr_idx = i
            break
    if hdr_idx is None:
        # fallback: cari row yang mengandung "ID PERUSAHAAN"
        for i in range(min(60, len(raw))):
            row = raw.iloc[i].astype(str).str.upper().tolist()
            if any("ID PERUSAHAAN" in x for x in row):
                hdr_idx = i
                break
    if hdr_idx is None:
        raise ValueError("Header Rekap Perusahaan tidak ditemukan (row NO/ID PERUSAHAAN).")

    df = pd.read_excel(file, sheet_name=0, header=hdr_idx)
    df.columns = [str(c).strip() for c in df.columns]

    # canonical rename
    rename = {}
    for c in df.columns:
        cu = c.strip().upper()
        if cu == "ID PERUSAHAAN": rename[c] = "COMPANY_ID"
        if cu == "NAMA PERUSAHAAN": rename[c] = "COMPANY_NAME"
        if cu == "ALAMAT": rename[c] = "ADDRESS"
        if cu == "TELEPON": rename[c] = "PHONE"
        if cu == "EMAIL": rename[c] = "EMAIL"
        if cu == "USER AKTIF": rename[c] = "USER_ACTIVE"
        if cu == "USER TIDAK AKTIF": rename[c] = "USER_INACTIVE"
        if cu == "TANGGAL DIBUAT": rename[c] = "CREATED_AT"
        if cu == "AWAL BERLANGGANAN": rename[c] = "SUBS_START"
        if cu == "AKHIR BERLANGGANAN": rename[c] = "SUBS_END"
        if cu == "STATUS": rename[c] = "STATUS"
    df = df.rename(columns=rename)

    # clean
    if "COMPANY_NAME" in df.columns:
        df["COMPANY_NAME"] = norm_text(df["COMPANY_NAME"])
        df["_CKEY"] = company_key(df["COMPANY_NAME"])
    else:
        df["_CKEY"] = "UNKNOWN"

    for c in ["USER_ACTIVE", "USER_INACTIVE", "COMPANY_ID"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["CREATED_AT", "SUBS_START", "SUBS_END"]:
        if c in df.columns:
            df[c] = clean_dates(df[c])

    if "STATUS" in df.columns:
        df["STATUS"] = norm_text(df["STATUS"]).str.upper()

    return df

# =============================
# LOAD
# =============================
emp_df, emp_meta = load_employee(emp_file) if emp_file else (None, {})
rekap_df = load_rekap(rekap_file) if rekap_file else None

if emp_df is None and rekap_df is None:
    st.info("Upload minimal 1 file: Dataset Karyawan atau Rekap Perusahaan.")
    st.stop()

# =============================
# BUILD COMPANY MASTER (Decision Table)
# =============================
company_master = None

# base company dimension from rekap
if rekap_df is not None:
    cm = rekap_df.copy()
    if "COMPANY_NAME" not in cm.columns:
        cm["COMPANY_NAME"] = "UNKNOWN"
    if "_CKEY" not in cm.columns:
        cm["_CKEY"] = company_key(cm["COMPANY_NAME"])
else:
    cm = pd.DataFrame(columns=["_CKEY", "COMPANY_NAME"])

# employee company aggregation
if emp_df is not None and "PERUSAHAAN" in emp_df.columns:
    e = emp_df.copy()
    e["_CKEY"] = company_key(e["PERUSAHAAN"])

    g = e.groupby("_CKEY", dropna=False)
    agg = pd.DataFrame({
        "_CKEY": g.size().index,
        "EMP_TOTAL": g.size().values.astype(int),
        "EMP_ACTIVE": g["_IS_AKTIF"].sum().values.astype(int),
    })
    agg["EMP_ACTIVE_RATE%"] = np.where(agg["EMP_TOTAL"] > 0, agg["EMP_ACTIVE"] / agg["EMP_TOTAL"] * 100, np.nan)

    # New employee accounts 30D (TANGGAL_DIBUAT)
    if "TANGGAL_DIBUAT" in e.columns and e["TANGGAL_DIBUAT"].notna().any():
        last30 = now - pd.Timedelta(days=30)
        tmp = e[e["TANGGAL_DIBUAT"].notna() & e["TANGGAL_DIBUAT"].between(last30, now)]
        new30 = tmp.groupby("_CKEY").size()
        agg["EMP_NEW_30D"] = agg["_CKEY"].map(new30).fillna(0).astype(int)
    else:
        agg["EMP_NEW_30D"] = 0

    # Employee churn 30D (TANGGAL_NONAKTIF)
    if "TANGGAL_NONAKTIF" in e.columns and e["TANGGAL_NONAKTIF"].notna().any():
        last30 = now - pd.Timedelta(days=30)
        tmp = e[e["TANGGAL_NONAKTIF"].notna() & e["TANGGAL_NONAKTIF"].between(last30, now)]
        ch30 = tmp.groupby("_CKEY").size()
        agg["EMP_CHURN_30D"] = agg["_CKEY"].map(ch30).fillna(0).astype(int)
    else:
        agg["EMP_CHURN_30D"] = 0

    agg["EMP_NET_30D"] = agg["EMP_NEW_30D"] - agg["EMP_CHURN_30D"]

    # Engagement 30D (LOGIN_TERAKHIR)
    if "LOGIN_TERAKHIR" in e.columns and e["LOGIN_TERAKHIR"].notna().any():
        last30 = now - pd.Timedelta(days=30)
        tmp = e[e["LOGIN_TERAKHIR"].notna()].copy()
        engaged30 = tmp.groupby("_CKEY")["LOGIN_TERAKHIR"].apply(lambda s: (s >= last30).mean() * 100.0)
        agg["EMP_ENGAGED_30D%"] = agg["_CKEY"].map(engaged30).round(1)
    else:
        agg["EMP_ENGAGED_30D%"] = np.nan

    # Compliance missing (ACTIVE)
    act = e[e["_IS_AKTIF"]].copy()
    if "NPWP" in act.columns and len(act):
        miss = act.groupby("_CKEY")["NPWP"].apply(lambda s: is_missing_text(s).mean() * 100.0)
        agg["NPWP_MISS%_AKTIF"] = agg["_CKEY"].map(miss).round(1)
    if "BPJS_KES" in act.columns and len(act):
        miss = act.groupby("_CKEY")["BPJS_KES"].apply(lambda s: is_missing_text(s).mean() * 100.0)
        agg["BPJSKES_MISS%_AKTIF"] = agg["_CKEY"].map(miss).round(1)
    if "NO_BPJSTK" in act.columns and len(act):
        miss = act.groupby("_CKEY")["NO_BPJSTK"].apply(lambda s: is_missing_text(s).mean() * 100.0)
        agg["BPJSTK_MISS%_AKTIF"] = agg["_CKEY"].map(miss).round(1)

else:
    agg = pd.DataFrame(columns=["_CKEY"])

# merge
if len(cm) and len(agg):
    company_master = cm.merge(agg, on="_CKEY", how="outer")
elif len(cm):
    company_master = cm.copy()
elif len(agg):
    company_master = agg.copy()

# fill company name from employee dataset if missing
if company_master is not None:
    if "COMPANY_NAME" not in company_master.columns:
        company_master["COMPANY_NAME"] = np.nan
    if emp_df is not None and "PERUSAHAAN" in emp_df.columns:
        name_map = (emp_df[["PERUSAHAAN"]].dropna()
                    .assign(_CKEY=company_key(emp_df["PERUSAHAAN"]))
                    .drop_duplicates("_CKEY"))
        name_map = dict(zip(name_map["_CKEY"], name_map["PERUSAHAAN"]))
        company_master["COMPANY_NAME"] = company_master["COMPANY_NAME"].fillna(company_master["_CKEY"].map(name_map))
    company_master["COMPANY_NAME"] = company_master["COMPANY_NAME"].fillna("UNKNOWN")

# =============================
# SIDEBAR FILTERS (DEFAULT ALL)
# =============================
st.sidebar.header("⚙️ Filters (Default: ALL)")
show_only_active_emp = st.sidebar.checkbox("Hanya karyawan aktif (untuk analisis HR)", value=False)

company_list = sorted(company_master["COMPANY_NAME"].astype(str).unique().tolist()) if company_master is not None else []
pick_companies = st.sidebar.multiselect("Filter perusahaan (kosong = semua)", company_list, default=[])

# date filter for employee-created (growth)
enable_emp_created_filter = False
date_range_emp = None
if emp_df is not None and "TANGGAL_DIBUAT" in emp_df.columns and emp_df["TANGGAL_DIBUAT"].notna().any():
    enable_emp_created_filter = st.sidebar.checkbox("Aktifkan filter TANGGAL_DIBUAT (karyawan)", value=False)
    if enable_emp_created_filter:
        mn = emp_df["TANGGAL_DIBUAT"].min()
        mx = emp_df["TANGGAL_DIBUAT"].max()
        date_range_emp = st.sidebar.date_input("Range TANGGAL_DIBUAT (karyawan)", value=(mn.date(), mx.date()))

st.sidebar.markdown('<hr class="hrline"/>', unsafe_allow_html=True)

# health score weights
st.sidebar.subheader("🏥 Health Score Weights")
w_adoption = st.sidebar.slider("Adoption (USER_ACTIVE / employee active)", 0, 50, 25)
w_engage = st.sidebar.slider("Engagement (Login 30D)", 0, 50, 25)
w_net = st.sidebar.slider("Net Growth 30D", 0, 50, 25)
w_compliance = st.sidebar.slider("Compliance (NPWP/BPJS missing)", 0, 50, 25)

# =============================
# APPLY FILTERED VIEWS
# =============================
cm_view = company_master.copy() if company_master is not None else pd.DataFrame()
if pick_companies:
    cm_view = cm_view[cm_view["COMPANY_NAME"].astype(str).isin(pick_companies)].copy()

emp_view = None
if emp_df is not None:
    emp_view = emp_df.copy()
    if pick_companies and "PERUSAHAAN" in emp_view.columns:
        emp_view = emp_view[emp_view["PERUSAHAAN"].astype(str).isin(pick_companies)].copy()
    if show_only_active_emp:
        emp_view = emp_view[emp_view["_IS_AKTIF"]].copy()
    if enable_emp_created_filter and date_range_emp is not None:
        start_d = pd.to_datetime(date_range_emp[0])
        end_d = pd.to_datetime(date_range_emp[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        emp_view = emp_view[emp_view["TANGGAL_DIBUAT"].notna() & emp_view["TANGGAL_DIBUAT"].between(start_d, end_d)].copy()

# =============================
# EXEC METRICS (ACCURATE)
# =============================
total_companies = int(company_master["_CKEY"].nunique()) if company_master is not None else 0

rekap_active_companies = None
if company_master is not None and "STATUS" in company_master.columns:
    rekap_active_companies = int((company_master["STATUS"] == "AKTIF").sum())

user_active_total = None
user_inactive_total = None
if company_master is not None and "USER_ACTIVE" in company_master.columns:
    user_active_total = float(pd.to_numeric(company_master["USER_ACTIVE"], errors="coerce").fillna(0).sum())
if company_master is not None and "USER_INACTIVE" in company_master.columns:
    user_inactive_total = float(pd.to_numeric(company_master["USER_INACTIVE"], errors="coerce").fillna(0).sum())

emp_total = int(emp_df.shape[0]) if emp_df is not None else None
emp_active = int(emp_df["_IS_AKTIF"].sum()) if (emp_df is not None and "_IS_AKTIF" in emp_df.columns) else None
emp_active_rate = (emp_active / emp_total * 100.0) if (emp_total and emp_active is not None) else None

# Growth company 30D (rekap created)
new_company_30d = None
if company_master is not None and "CREATED_AT" in company_master.columns and company_master["CREATED_AT"].notna().any():
    last30 = now - pd.Timedelta(days=30)
    new_company_30d = int(company_master["CREATED_AT"].between(last30, now).sum())

# Employee growth 30D
new_emp_30d = None
if emp_df is not None and "TANGGAL_DIBUAT" in emp_df.columns and emp_df["TANGGAL_DIBUAT"].notna().any():
    last30 = now - pd.Timedelta(days=30)
    new_emp_30d = int((emp_df["TANGGAL_DIBUAT"].between(last30, now)).sum())

# Employee churn 30D
churn_emp_30d = None
if emp_df is not None and "TANGGAL_NONAKTIF" in emp_df.columns and emp_df["TANGGAL_NONAKTIF"].notna().any():
    last30 = now - pd.Timedelta(days=30)
    churn_emp_30d = int((emp_df["TANGGAL_NONAKTIF"].between(last30, now)).sum())

# Subscription risk: ending in next 30/60/90 days
subs_end_30 = subs_end_60 = subs_end_90 = None
if company_master is not None and "SUBS_END" in company_master.columns and company_master["SUBS_END"].notna().any():
    d = company_master["SUBS_END"]
    subs_end_30 = int(((d >= now) & (d <= now + pd.Timedelta(days=30))).sum())
    subs_end_60 = int(((d >= now) & (d <= now + pd.Timedelta(days=60))).sum())
    subs_end_90 = int(((d >= now) & (d <= now + pd.Timedelta(days=90))).sum())


# =============================
# HEALTH SCORE (0-100) — Robust (rekap saja / + karyawan)
# =============================
cm_score = cm_view.copy()

def minmax(series):
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series([0.0]*len(series), index=series.index)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn == 0:
        return pd.Series([50.0]*len(series), index=series.index)
    return (x - mn) / (mx - mn) * 100.0

def get_series(df, col, default=0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index)

# --- Adoption (utama)
if "USER_ACTIVE" in cm_score.columns:
    adoption_raw = get_series(cm_score, "USER_ACTIVE", 0).fillna(0)
elif "EMP_ACTIVE" in cm_score.columns:
    adoption_raw = get_series(cm_score, "EMP_ACTIVE", 0).fillna(0)
else:
    adoption_raw = pd.Series([0]*len(cm_score), index=cm_score.index)

adoption_score = minmax(adoption_raw)

# --- Engagement (kalau ada EMP_ENGAGED_30D% pakai itu; kalau rekap saja, pakai proxy active ratio)
if "EMP_ENGAGED_30D%" in cm_score.columns:
    engage_score = get_series(cm_score, "EMP_ENGAGED_30D%", 0).fillna(0).clip(0, 100)
elif "USER_ACTIVE" in cm_score.columns and "USER_INACTIVE" in cm_score.columns:
    ua = get_series(cm_score, "USER_ACTIVE", 0).fillna(0)
    ui = get_series(cm_score, "USER_INACTIVE", 0).fillna(0)
    denom = (ua + ui).replace(0, np.nan)
    engage_score = (ua / denom * 100.0).fillna(0).clip(0, 100)  # proxy engagement
else:
    engage_score = pd.Series([0]*len(cm_score), index=cm_score.index)

# --- Net Growth (kalau nggak ada, 0)
net_raw = get_series(cm_score, "EMP_NET_30D", 0).fillna(0)
net_score = minmax(net_raw)

# --- Compliance (kalau kolom missing-rate tidak ada, anggap 100 supaya tidak mengurangi score)
has_comp = any(c in cm_score.columns for c in ["NPWP_MISS%_AKTIF", "BPJSKES_MISS%_AKTIF", "BPJSTK_MISS%_AKTIF"])
if has_comp:
    npwp_miss    = get_series(cm_score, "NPWP_MISS%_AKTIF", 0).fillna(0)
    bpjskes_miss = get_series(cm_score, "BPJSKES_MISS%_AKTIF", 0).fillna(0)
    bpjstk_miss  = get_series(cm_score, "BPJSTK_MISS%_AKTIF", 0).fillna(0)
    compliance_penalty = (npwp_miss + bpjskes_miss + bpjstk_miss) / 3.0
    compliance_score = (100.0 - compliance_penalty).clip(0, 100)
else:
    compliance_score = pd.Series([100.0]*len(cm_score), index=cm_score.index)

# --- Weighted Score (adaptive: bobot komponen yang datanya "kosong semua" akan diabaikan)
components = {
    "adoption": (adoption_score, w_adoption),
    "engage": (engage_score, w_engage),
    "net": (net_score, w_net),
    "compliance": (compliance_score, w_compliance),
}

active_weights = []
for name, (sc, w) in components.items():
    # kalau seluruh score 0 dan kolom memang tidak ada (indikasi kosong), skip bobotnya
    if sc.notna().sum() == 0:
        continue
    active_weights.append(w)

W = sum(active_weights) if sum(active_weights) > 0 else 1

cm_score["HEALTH_SCORE"] = (
    adoption_score * (w_adoption / W) +
    engage_score * (w_engage / W) +
    net_score * (w_net / W) +
    compliance_score * (w_compliance / W)
).round(1)


# =============================
# TABS (many reports)
# =============================
tab_exec, tab_portfolio, tab_growth, tab_ewa, tab_engage, tab_hr, tab_quality, tab_debug = st.tabs([
    "📌 Executive",
    "🏢 Portfolio & Score",
    "📈 Growth / Churn / Cohort",
    "⚡💳 EWA / PPOB",
    "⚡ Engagement",
    "👥 HR Insights",
    "🛡️ Data Quality",
    "🧪 Debug & Export",
])

# =============================
# TAB: EXECUTIVE
# =============================
with tab_exec:
    st.subheader("Executive Snapshot (Most Decision-Relevant)")

    r1 = st.columns(4)
    with r1[0]: kpi_card("Total Perusahaan", format_int(total_companies), "Unik (gabungan rekap + employee)", pill="CLIENTS")
    with r1[1]: kpi_card("Perusahaan AKTIF (Rekap)", format_int(rekap_active_companies) if rekap_active_companies is not None else "—", "Berdasarkan kolom STATUS rekap", pill="PORTFOLIO")
    with r1[2]: kpi_card("Total User Aktif (Rekap)", format_int(user_active_total) if user_active_total is not None else "—", "Karyawan Aktif", pill="ADOPTION")
    with r1[3]: kpi_card("Total User Tidak Aktif (Rekap)", format_int(user_inactive_total) if user_inactive_total is not None else "—", "Karyawan Non Aktif", pill="ADOPTION")

    r2 = st.columns(4)
    with r2[0]: kpi_card("Total Karyawan", format_int(emp_total) if emp_total is not None else "—", "Total Seluruh Karyawan", pill="HR")
    with r2[1]: kpi_card("Karyawan Aktif", format_int(emp_active) if emp_active is not None else "—", "Berdasarkan STATUS (karyawan)", pill="HR")
    with r2[2]: kpi_card("Employee Active Rate", format_pct(emp_active_rate), "Aktif / total", pill="HR")
    with r2[3]:
        kpi_card("New Employee Accounts (30D)", format_int(new_emp_30d) if new_emp_30d is not None else "—",
                 "Total Karyawan Baru", pill="GROWTH")

    r3 = st.columns(4)
    with r3[0]:
        kpi_card("New Companies (30D)", format_int(new_company_30d) if new_company_30d is not None else "—",
                 "Total Perusahaan Baru", pill="GROWTH")
    with r3[1]:
        kpi_card("Employee Churn (30D)", format_int(churn_emp_30d) if churn_emp_30d is not None else "—",
                 "Berdasarkan TANGGAL_NONAKTIF (karyawan)", pill="RISK")
    with r3[2]:
        kpi_card("Subs End ≤ 30D", format_int(subs_end_30) if subs_end_30 is not None else "—",
                 "Berlangganan Akan Habis Bulan Ini", pill="RISK")
    with r3[3]:
        kpi_card("Subs End ≤ 90D", format_int(subs_end_90) if subs_end_90 is not None else "—",
                 "Berlangganan Akan Habis <3 Bulan", pill="RISK")

    st.markdown('<hr class="hrline"/>', unsafe_allow_html=True)
    st.markdown("#### Quick Insights (auto)")
    insights = []
    if subs_end_30 is not None and subs_end_30 > 0:
        insights.append(f"⚠️ Ada **{format_int(subs_end_30)}** perusahaan dengan **Akhir Berlangganan dalam 30 hari** (prioritas renewal).")
    if churn_emp_30d is not None and churn_emp_30d > 0:
        insights.append(f"📉 Ada **{format_int(churn_emp_30d)}** karyawan nonaktif dalam 30 hari terakhir (indikasi churn internal / offboarding).")
    if user_inactive_total is not None and user_active_total is not None and (user_active_total + user_inactive_total) > 0:
        inactive_rate = user_inactive_total / (user_active_total + user_inactive_total) * 100
        insights.append(f"🧯 Inactive user ratio (rekap) ≈ **{inactive_rate:.1f}%** (semakin tinggi = adopsi menurun).")
    if not insights:
        insights.append("✅ Data cukup stabil. Gunakan tab Portfolio & Score untuk prioritas tindakan per perusahaan.")
    st.write("\n".join(insights))

    # ============================
    # Renewal Watchlist Tables
    # ============================
    st.markdown('<hr class="hrline"/>', unsafe_allow_html=True)
    st.markdown("### Renewal Watchlist (SUBS_END)")

    base = cm_view.copy() if "cm_view" in locals() and cm_view is not None else company_master.copy()

    if base is None or len(base) == 0 or "SUBS_END" not in base.columns:
        st.info("Kolom SUBS_END tidak tersedia. Upload Rekap Perusahaan dan pastikan ada kolom AKHIR BERLANGGANAN.")
    else:
        # pastikan datetime
        base["SUBS_END"] = pd.to_datetime(base["SUBS_END"], errors="coerce")

        # hitung days to end (negatif berarti sudah lewat)
        base["DAYS_TO_END"] = (base["SUBS_END"] - now).dt.days

        # ambil kolom-kolom yang "cukup lengkap" jika tersedia
        cols_priority = [
            "COMPANY_ID", "COMPANY_NAME", "STATUS",
            "SUBS_START", "SUBS_END", "DAYS_TO_END",
            "USER_ACTIVE", "USER_INACTIVE",
            "EMP_TOTAL", "EMP_ACTIVE", "EMP_ENGAGED_30D%", "HEALTH_SCORE",
            "NPWP_MISS%_AKTIF", "BPJSKES_MISS%_AKTIF", "BPJSTK_MISS%_AKTIF"
        ]
        show_cols = [c for c in cols_priority if c in base.columns]

        # Filter: mau expired (0..30)
        soon = base[
            base["SUBS_END"].notna() &
            (base["DAYS_TO_END"] >= 0) &
            (base["DAYS_TO_END"] <= 30)
        ].copy().sort_values(["DAYS_TO_END", "COMPANY_NAME"], ascending=[True, True])

        # Filter: mendatang expired (31..90)
        upcoming = base[
            base["SUBS_END"].notna() &
            (base["DAYS_TO_END"] >= 31) &
            (base["DAYS_TO_END"] <= 90)
        ].copy().sort_values(["DAYS_TO_END", "COMPANY_NAME"], ascending=[True, True])

        # Optional: sudah expired (biar lengkap, kalau mau)
        expired = base[
            base["SUBS_END"].notna() &
            (base["DAYS_TO_END"] < 0)
        ].copy().sort_values(["DAYS_TO_END", "COMPANY_NAME"], ascending=[True, True])

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 1) Akan Expired (≤ 30 hari)")
            st.caption("Prioritas: segera hubungi untuk renewal / extend kontrak.")
            if len(soon) == 0:
                st.info("Tidak ada perusahaan yang akan expired dalam 30 hari.")
            else:
                st.dataframe(soon[show_cols], use_container_width=True, height=320)
                st.download_button(
                    "⬇️ Download list expiring ≤30D (CSV)",
                    data=soon[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name="renewal_expiring_30d.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with c2:
            st.markdown("#### 2) Mendatang Expired (31–90 hari)")
            st.caption("Pipeline renewal: persiapan penawaran + follow up bertahap.")
            if len(upcoming) == 0:
                st.info("Tidak ada perusahaan yang akan expired dalam 31–90 hari.")
            else:
                st.dataframe(upcoming[show_cols], use_container_width=True, height=320)
                st.download_button(
                    "⬇️ Download list expiring 31–90D (CSV)",
                    data=upcoming[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name="renewal_upcoming_31_90d.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # (Opsional tapi recommended) tampilkan yang sudah lewat
        with st.expander("Lihat perusahaan yang sudah expired (SUBS_END lewat)", expanded=False):
            if len(expired) == 0:
                st.info("Tidak ada perusahaan yang SUBS_END-nya sudah lewat.")
            else:
                st.dataframe(expired[show_cols], use_container_width=True, height=320)


# =============================
# TAB: PORTFOLIO & SCORE
# =============================
with tab_portfolio:
    st.subheader("Portfolio Decision Table + Health Score (0–100)")

    view = cm_score.copy()

    # derived adoption inactive ratio (rekap)
    if "USER_ACTIVE" in view.columns and "USER_INACTIVE" in view.columns:
        ua = pd.to_numeric(view["USER_ACTIVE"], errors="coerce").fillna(0)
        ui = pd.to_numeric(view["USER_INACTIVE"], errors="coerce").fillna(0)
        denom = ua + ui
        view["INACTIVE_USER_RATE%"] = np.where(denom > 0, (ui/denom)*100.0, np.nan).round(1)

    # risk flags
    if "SUBS_END" in view.columns:
        view["SUBS_END_IN_30D"] = np.where(view["SUBS_END"].notna() & (view["SUBS_END"] >= now) & (view["SUBS_END"] <= now + pd.Timedelta(days=30)), "YES", "NO")

    # table controls
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        q = st.text_input("Search perusahaan", "")
    with c2:
        min_score = st.slider("Min Health Score", 0, 100, 0)
    with c3:
        min_emp_active = st.number_input("Min EMP_ACTIVE", 0, 10**9, 0, 1)
    with c4:
        max_inactive_user = st.slider("Max Inactive User Rate%", 0, 100, 100)

    if q.strip():
        view = view[view["COMPANY_NAME"].astype(str).str.contains(q.strip(), case=False, na=False)]
    view = view[view["HEALTH_SCORE"].fillna(0) >= min_score]
    if "EMP_ACTIVE" in view.columns:
        view = view[pd.to_numeric(view["EMP_ACTIVE"], errors="coerce").fillna(0) >= min_emp_active]
    if "INACTIVE_USER_RATE%" in view.columns:
        view = view[view["INACTIVE_USER_RATE%"].fillna(0) <= max_inactive_user]

    # sort default + charts
    if len(view):
        sort_cols = [c for c in ["HEALTH_SCORE", "EMP_ACTIVE", "USER_ACTIVE"] if c in view.columns]
        if not sort_cols:
            sort_cols = [view.columns[0]]
        view = view.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        st.dataframe(view, use_container_width=True, height=520)

        st.markdown('<hr class="hrline"/>', unsafe_allow_html=True)
        topn = st.slider("Top N chart", 5, 30, 10)

        cA, cB = st.columns(2)
        with cA:
            if "COMPANY_NAME" in view.columns and "HEALTH_SCORE" in view.columns:
                t = view[["COMPANY_NAME", "HEALTH_SCORE"]].dropna().sort_values("HEALTH_SCORE").tail(topn)
                fig = px.bar(t, x="HEALTH_SCORE", y="COMPANY_NAME", orientation="h", title=f"Top {topn} — Health Score")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom COMPANY_NAME/HEALTH_SCORE tidak tersedia untuk chart.")

        with cB:
            if "SUBS_END_IN_30D" in view.columns:
                r = view["SUBS_END_IN_30D"].value_counts(dropna=False).rename_axis("FLAG").reset_index(name="COUNT")
                fig = px.pie(r, values="COUNT", names="FLAG", title="Renewal Risk (SUBS_END in 30D)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom SUBS_END tidak tersedia untuk Renewal Risk chart.")
    else:
        st.info("Tidak ada data yang match filter di Portfolio.")

# =============================
# TAB: GROWTH / CHURN / COHORT
# =============================
with tab_growth:
    st.subheader("Growth / Churn / Cohort Analytics")

    # 1) Employee monthly new & churn
    if emp_view is not None and "TANGGAL_DIBUAT" in emp_view.columns and emp_view["TANGGAL_DIBUAT"].notna().any():
        d = emp_view[emp_view["TANGGAL_DIBUAT"].notna()].copy()
        d["MONTH"] = d["TANGGAL_DIBUAT"].dt.to_period("M").astype(str)
        m_new = d.groupby("MONTH").size().reset_index(name="NEW_EMP").sort_values("MONTH")

        if "TANGGAL_NONAKTIF" in emp_view.columns and emp_view["TANGGAL_NONAKTIF"].notna().any():
            c = emp_view[emp_view["TANGGAL_NONAKTIF"].notna()].copy()
            c["MONTH"] = c["TANGGAL_NONAKTIF"].dt.to_period("M").astype(str)
            m_ch = c.groupby("MONTH").size().reset_index(name="CHURN_EMP").sort_values("MONTH")
        else:
            m_ch = pd.DataFrame(columns=["MONTH", "CHURN_EMP"])

        trend = pd.merge(m_new, m_ch, on="MONTH", how="outer").fillna(0)
        trend["NET_EMP"] = trend["NEW_EMP"] - trend["CHURN_EMP"]
        trend = trend.sort_values("MONTH")

        fig = px.line(trend, x="MONTH", y=["NEW_EMP", "CHURN_EMP", "NET_EMP"], markers=True,
                      title="Employee Accounts: New vs Churn vs Net (Monthly)")
        st.plotly_chart(fig, use_container_width=True)

        # Cohort active-rate (employee): created-month vs current active %
        coh = emp_view[emp_view["TANGGAL_DIBUAT"].notna()].copy()
        coh["COHORT"] = coh["TANGGAL_DIBUAT"].dt.to_period("M").astype(str)
        cohort_tbl = coh.groupby("COHORT").agg(
            ACCOUNTS=("COHORT","size"),
            ACTIVE=("_IS_AKTIF","sum"),
        ).reset_index()
        cohort_tbl["ACTIVE_RATE%"] = np.where(cohort_tbl["ACCOUNTS"]>0, cohort_tbl["ACTIVE"]/cohort_tbl["ACCOUNTS"]*100.0, np.nan).round(1)
        cohort_tbl = cohort_tbl.sort_values("COHORT")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cohort_tbl, x="COHORT", y="ACCOUNTS", title="Cohort Size (Employee Accounts by Created Month)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(cohort_tbl, x="COHORT", y="ACTIVE_RATE%", markers=True, title="Cohort Active Rate% (Proxy Retention)")
            fig.update_yaxes(range=[0,100])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload dataset karyawan (dan pastikan TANGGAL_DIBUAT terbaca) untuk Growth/Churn/Cohort di level karyawan.")

    st.markdown('<hr class="hrline"/>', unsafe_allow_html=True)

    # 2) Company growth + subscription churn proxy
    if rekap_df is not None and "CREATED_AT" in rekap_df.columns and rekap_df["CREATED_AT"].notna().any():
        rc = rekap_df.copy()
        rc["MONTH"] = rc["CREATED_AT"].dt.to_period("M").astype(str)
        m = rc.groupby("MONTH").size().reset_index(name="NEW_COMPANIES").sort_values("MONTH")
        fig = px.line(m, x="MONTH", y="NEW_COMPANIES", markers=True, title="New Companies (Monthly) — Rekap Perusahaan")
        st.plotly_chart(fig, use_container_width=True)

        if "SUBS_END" in rekap_df.columns and rekap_df["SUBS_END"].notna().any():
            se = rekap_df[rekap_df["SUBS_END"].notna()].copy()
            se["MONTH"] = se["SUBS_END"].dt.to_period("M").astype(str)
            mm = se.groupby("MONTH").size().reset_index(name="SUBS_END_COUNT").sort_values("MONTH")
            fig = px.bar(mm, x="MONTH", y="SUBS_END_COUNT", title="Subscription End (Monthly) — Proxy Churn")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload rekap perusahaan untuk Growth/Subscription analytics di level perusahaan.")


# =============================
# TAB: EWA / PPOB
# =============================
with tab_ewa:
    render_ewa_page(ewa_file, company_master=company_master)

# =============================
# TAB: ENGAGEMENT
# =============================
with tab_engage:
    st.subheader("Engagement / Usage Analytics (Login Terakhir)")

    if emp_view is None or "LOGIN_TERAKHIR" not in emp_view.columns or not emp_view["LOGIN_TERAKHIR"].notna().any():
        st.info("LOGIN_TERAKHIR tidak tersedia / kosong. Upload dataset karyawan atau cek kolom Unnamed: 39.")
    else:
        e = emp_view[emp_view["LOGIN_TERAKHIR"].notna()].copy()
        e["DAYS_SINCE_LOGIN"] = (now - e["LOGIN_TERAKHIR"]).dt.days

        # Distribution bins
        bins = pd.cut(
            e["DAYS_SINCE_LOGIN"],
            bins=[-1, 7, 30, 90, 365, 999999],
            labels=["0–7D", "8–30D", "31–90D", "91–365D", ">365D"]
        )
        dist = bins.value_counts().sort_index().reset_index()
        dist.columns = ["BUCKET", "COUNT"]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(dist, x="BUCKET", y="COUNT", title="Login Recency Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Engagement by company
        if "PERUSAHAAN" in e.columns:
            last30 = now - pd.Timedelta(days=30)
            g = e.groupby("PERUSAHAAN")["LOGIN_TERAKHIR"].apply(lambda s: (s >= last30).mean() * 100.0).reset_index(name="ENGAGED_30D%")
            g["ENGAGED_30D%"] = g["ENGAGED_30D%"].round(1)
            g = g.sort_values("ENGAGED_30D%", ascending=False)

            topn = st.slider("Top N engagement company", 5, 30, 10)
            fig = px.bar(g.head(topn).sort_values("ENGAGED_30D%"),
                         x="ENGAGED_30D%", y="PERUSAHAAN", orientation="h",
                         title=f"Top {topn} Company — Engagement 30D% (Employee Login)")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

# =============================
# TAB: HR INSIGHTS
# =============================
with tab_hr:
    st.subheader("HR Insights (Gender / Dept / Job / Tenure)")

    if emp_view is None:
        st.info("Upload dataset karyawan untuk HR Insights.")
    else:
        act = emp_view[emp_view["_IS_AKTIF"]].copy()  # HR chart fokus aktif (sesuai instruksi awal)
        st.caption("Chart HR default memakai karyawan aktif (lebih relevan untuk komposisi saat ini).")

        col_gender = emp_meta.get("col_gender")
        col_dept = emp_meta.get("col_dept")
        col_job = emp_meta.get("col_job")
        col_bank = emp_meta.get("col_bank")
        col_name = emp_meta.get("col_name")

        c1, c2 = st.columns(2)

        # Gender pie
        with c1:
            if col_gender and col_gender in act.columns and len(act):
                g = act[col_gender].fillna("UNKNOWN").astype(str).value_counts()
                fig = px.pie(values=g.values, names=g.index, title="Komposisi Gender (Aktif)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom gender tidak ditemukan / data aktif kosong.")

        # Tenure distribution
        with c2:
            if "MASA_KERJA_BULAN_CALC" in act.columns and act["MASA_KERJA_BULAN_CALC"].notna().any():
                t = act["MASA_KERJA_BULAN_CALC"].clip(lower=0)
                fig = px.histogram(t, nbins=30, title="Distribusi Masa Kerja (bulan) — Aktif")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tenure (MASA_KERJA_BULAN_CALC) tidak tersedia.")

        # Top Departemen
        if col_dept and col_dept in act.columns and len(act):
            top = act[col_dept].fillna("UNKNOWN").astype(str).value_counts().head(10)
            dfp = top.reset_index(name="COUNT")
            dfp.columns = ["DEPARTEMEN", "COUNT"]  # FIX: avoid 'index' plotly error
            fig = px.bar(dfp, x="COUNT", y="DEPARTEMEN", orientation="h", title="Top 10 Departemen (Aktif)")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # Top Jabatan
        if col_job and col_job in act.columns and len(act):
            topj = act[col_job].fillna("UNKNOWN").astype(str).value_counts().head(10)
            dfp = topj.reset_index(name="COUNT")
            dfp.columns = ["JABATAN", "COUNT"]
            fig = px.bar(dfp, x="COUNT", y="JABATAN", orientation="h", title="Top 10 Jabatan (Aktif)")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<hr class="hrline"/>', unsafe_allow_html=True)
        st.markdown("### Audit khusus (Aktif): NPWP/BPJS kosong + Tenure lama tapi BANK kosong")

        # Missing rates
        npwp_pct = bpjs_pct = bank_pct = None
        if "NPWP" in act.columns:
            npwp_pct = float(is_missing_text(act["NPWP"]).mean() * 100.0)
        if "BPJS_KES" in act.columns:
            bpjs_pct = float(is_missing_text(act["BPJS_KES"]).mean() * 100.0)
        if col_bank and col_bank in act.columns:
            bank_pct = float(is_missing_text(act[col_bank]).mean() * 100.0)

        a1, a2, a3 = st.columns(3)
        with a1: kpi_card("NPWP Missing % (Aktif)", format_pct(npwp_pct), "Compliance", pill="AUDIT")
        with a2: kpi_card("BPJS KES Missing % (Aktif)", format_pct(bpjs_pct), "Compliance", pill="AUDIT")
        with a3: kpi_card("BANK Missing % (Aktif)", format_pct(bank_pct), "Payroll readiness", pill="AUDIT")

        # Top 5 longest tenure with bank missing
        if col_bank and col_bank in act.columns and "MASA_KERJA_BULAN_CALC" in act.columns:
            worst = act[is_missing_text(act[col_bank])].sort_values("MASA_KERJA_BULAN_CALC", ascending=False).head(5)
            show_cols = []
            for c in ["PERUSAHAAN", col_name, col_dept, col_job, col_bank, "MASA_KERJA_BULAN_CALC"]:
                if c and c in worst.columns:
                    show_cols.append(c)
            if len(worst):
                st.dataframe(worst[show_cols], use_container_width=True)
            else:
                st.info("Tidak ada karyawan aktif dengan BANK kosong.")
        else:
            st.info("Butuh kolom BANK dan tenure untuk menampilkan audit ini.")

# =============================
# TAB: DATA QUALITY
# =============================
with tab_quality:
    st.subheader("Data Quality & Consistency (Decision-Grade)")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Employee Dataset Quality")
        if emp_df is None:
            st.info("Upload dataset karyawan.")
        else:
            # key coverage
            coverage = []
            for col in ["PERUSAHAAN", "TANGGAL_DIBUAT", "LOGIN_TERAKHIR", "TANGGAL_NONAKTIF", "STATUS", "NPWP", "BPJS_KES", "NO_BPJSTK"]:
                if col in emp_df.columns:
                    coverage.append([col, float(emp_df[col].notna().mean()*100.0)])
            cov = pd.DataFrame(coverage, columns=["COLUMN", "NOT_NULL_%"]).sort_values("NOT_NULL_%")
            st.dataframe(cov, use_container_width=True, height=320)

    with c2:
        st.markdown("#### Rekap Dataset Quality")
        if rekap_df is None:
            st.info("Upload rekap perusahaan.")
        else:
            coverage = []
            for col in ["COMPANY_ID", "COMPANY_NAME", "USER_ACTIVE", "USER_INACTIVE", "CREATED_AT", "SUBS_START", "SUBS_END", "STATUS"]:
                if col in rekap_df.columns:
                    coverage.append([col, float(rekap_df[col].notna().mean()*100.0)])
            cov = pd.DataFrame(coverage, columns=["COLUMN", "NOT_NULL_%"]).sort_values("NOT_NULL_%")
            st.dataframe(cov, use_container_width=True, height=320)

    st.markdown('<hr class="hrline"/>', unsafe_allow_html=True)
    st.markdown("### Mismatch perusahaan (rekap vs employee)")
    if rekap_df is not None and emp_df is not None and "COMPANY_NAME" in rekap_df.columns and "PERUSAHAAN" in emp_df.columns:
        rekap_keys = set(company_key(rekap_df["COMPANY_NAME"]).dropna().unique().tolist())
        emp_keys = set(company_key(emp_df["PERUSAHAAN"]).dropna().unique().tolist())
        only_rekap = sorted(list(rekap_keys - emp_keys))
        only_emp = sorted(list(emp_keys - rekap_keys))

        x1, x2 = st.columns(2)
        with x1:
            st.markdown("#### Ada di Rekap tapi tidak ada di Employee (sample)")
            st.dataframe(pd.DataFrame({"company_key": only_rekap[:80]}), use_container_width=True, height=280)
        with x2:
            st.markdown("#### Ada di Employee tapi tidak ada di Rekap (sample)")
            st.dataframe(pd.DataFrame({"company_key": only_emp[:80]}), use_container_width=True, height=280)

        st.caption("Jika mismatch tinggi, next step: kita buat 'Alias Mapping Table' agar nama perusahaan bisa disamakan otomatis.")
    else:
        st.info("Upload kedua dataset untuk melihat mismatch check.")

# =============================
# TAB: DEBUG & EXPORT
# =============================
with tab_debug:
    st.subheader("Debug & Export")

    st.markdown("#### Status Loader")
    st.write({
        "employee_loaded": emp_df is not None,
        "rekap_loaded": rekap_df is not None,
        "company_master_rows": int(company_master.shape[0]) if company_master is not None else 0,
        "employee_rows": int(emp_df.shape[0]) if emp_df is not None else 0,
        "rekap_rows": int(rekap_df.shape[0]) if rekap_df is not None else 0,
    })

    with st.expander("Show columns — Employee", expanded=False):
        if emp_df is not None:
            st.write(emp_df.columns.tolist())
    with st.expander("Show columns — Rekap", expanded=False):
        if rekap_df is not None:
            st.write(rekap_df.columns.tolist())

    st.markdown('<hr class="hrline"/>', unsafe_allow_html=True)
    st.markdown("#### Download (Cleaned)")
    colA, colB, colC = st.columns(3)
    with colA:
        if emp_df is not None:
            st.download_button(
                "⬇️ Download cleaned employee (CSV)",
                data=emp_df.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_employee.csv",
                mime="text/csv",
                use_container_width=True
            )
    with colB:
        if rekap_df is not None:
            st.download_button(
                "⬇️ Download cleaned rekap (CSV)",
                data=rekap_df.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_rekap_perusahaan.csv",
                mime="text/csv",
                use_container_width=True
            )
    with colC:
        if cm_score is not None and len(cm_score):
            st.download_button(
                "⬇️ Download company master + score (CSV)",
                data=cm_score.to_csv(index=False).encode("utf-8"),
                file_name="company_master_score.csv",
                mime="text/csv",
                use_container_width=True
            )

st.caption("✅ Dashboard ini sudah 'scalable': tinggal tambah loader baru untuk jenis file lain (transaksi EWA/PPOB, invoice, presensi, support ticket) untuk memperkaya decision system.")
