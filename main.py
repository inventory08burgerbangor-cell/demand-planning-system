import streamlit as st
import pandas as pd

# forecasting.py, database.py, dan export.py adalah modul utama aplikasi.
# main.py hanya menjadi layer UI/orchestrator; seluruh perhitungan
# forecasting tetap berada di forecasting.py dan penyimpanan/export
# menggunakan modul masing-masing.
from forecasting import (
    run_forecasting,
    parse_period,
    XGBOOST_AVAILABLE,
    XGBOOST_MIN_HISTORY,
    get_recursive_forecasting_status,
    explain_recursive_forecast,
)

from database import (
    init_db,
    save_history,
    load_history,
    load_history_by_id,
    delete_history,
    delete_all_history,
)

from export import (
    export_forecast_excel,
    generate_export_filename,
)


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Demand Planning System",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DATABASE
# =========================================================

init_db()


# =========================================================
# CONSTANT
# =========================================================

BULAN = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}

MENU_OPTIONS = [
    "📊 Dashboard",
    "📦 Data OUT",
    "⚙️ Setting",
    "✅ Validasi",
    "🔮 Forecast",
    "🕘 History",
]

MENU_QUERY_MAP = {
    "dashboard": "📊 Dashboard",
    "data-out": "📦 Data OUT",
    "setting": "⚙️ Setting",
    "validasi": "✅ Validasi",
    "forecast": "🔮 Forecast",
    "history": "🕘 History",
}

MENU_REVERSE_MAP = {
    value: key
    for key, value in MENU_QUERY_MAP.items()
}


# =========================================================
# DEFAULT SUMMARY
# =========================================================

DEFAULT_SUMMARY = {
    "bbb": {
        "wape_ma": None,
        "wape_wma": None,
        "wape_xgboost": None,
        "accuracy_ma": None,
        "accuracy_wma": None,
        "accuracy_xgboost": None,
        "total_actual_ma": 0.0,
        "total_actual_wma": 0.0,
        "total_actual_xgboost": 0.0,
        "total_error_ma": 0.0,
        "total_error_wma": 0.0,
        "total_error_xgboost": 0.0,
        # Alias legacy untuk kompatibilitas history lama.
        "wape": None,
        "accuracy": None,
        "total_actual": 0.0,
        "total_error": 0.0,
        "best_method": None,
    },
    "bbt": {
        "wape_ma": None,
        "wape_wma": None,
        "wape_xgboost": None,
        "accuracy_ma": None,
        "accuracy_wma": None,
        "accuracy_xgboost": None,
        "total_actual_ma": 0.0,
        "total_actual_wma": 0.0,
        "total_actual_xgboost": 0.0,
        "total_error_ma": 0.0,
        "total_error_wma": 0.0,
        "total_error_xgboost": 0.0,
        "wape": None,
        "accuracy": None,
        "total_actual": 0.0,
        "total_error": 0.0,
        "best_method": None,
    },
}


# =========================================================
# SESSION STATE
# =========================================================

def reset_forecast_session():

    st.session_state.forecast_bbb = pd.DataFrame()

    st.session_state.forecast_bbt = pd.DataFrame()

    st.session_state.forecast_summary = {
        "bbb": DEFAULT_SUMMARY["bbb"].copy(),
        "bbt": DEFAULT_SUMMARY["bbt"].copy(),
    }

    st.session_state.forecast_loaded = False

    st.session_state.loaded_history_id = None

    st.session_state.loaded_forecast_info = None

    st.session_state.last_forecast_period = None

    st.session_state.last_forecast_history_months = None


if "forecast_setting" not in st.session_state:

    st.session_state.forecast_setting = {
        "month": 9,
        "year": 2026,
        # DEFAULT REVISI: gunakan 8 bulan histori.
        # Jika data tersedia kurang dari 8 bulan, validasi akan
        # memberitahu jumlah histori yang benar-benar tersedia.
        "history_months": 8,
        # CATATAN REVISI: True = gunakan seluruh histori sebelum forecast.
        "use_all_history": False,
    }

# Kompatibilitas session state dari versi sebelum mode semua histori.
if "use_all_history" not in st.session_state.forecast_setting:

    st.session_state.forecast_setting[
        "use_all_history"
    ] = False


if "data_out" not in st.session_state:

    st.session_state.data_out = pd.DataFrame()


if "data_valid" not in st.session_state:

    st.session_state.data_valid = False


if "forecast_bbb" not in st.session_state:

    st.session_state.forecast_bbb = pd.DataFrame()


if "forecast_bbt" not in st.session_state:

    st.session_state.forecast_bbt = pd.DataFrame()


if "forecast_summary" not in st.session_state:

    st.session_state.forecast_summary = {
        "bbb": DEFAULT_SUMMARY["bbb"].copy(),
        "bbt": DEFAULT_SUMMARY["bbt"].copy(),
    }


if "forecast_loaded" not in st.session_state:

    st.session_state.forecast_loaded = False


if "loaded_history_id" not in st.session_state:

    st.session_state.loaded_history_id = None


if "loaded_forecast_info" not in st.session_state:

    st.session_state.loaded_forecast_info = None


if "show_help" not in st.session_state:

    st.session_state.show_help = False


if "help_menu" not in st.session_state:

    st.session_state.help_menu = "📊 Dashboard"


if "help_page" not in st.session_state:

    st.session_state.help_page = 1


if "confirm_delete_history" not in st.session_state:

    st.session_state.confirm_delete_history = None


if "confirm_delete_all" not in st.session_state:

    st.session_state.confirm_delete_all = False


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .main-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 0px;
        animation: titleFade 0.45s ease-out;
    }

    .sub-title {
        color: #777;
        font-size: 14px;
        margin-top: 0px;
        margin-bottom: 20px;
        animation: subtitleFade 0.55s ease-out;
    }


    /* =====================================================
       SECTION
       ===================================================== */

    .section-title {
        font-size: 21px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 12px;
    }


    /* =====================================================
       METRIC
       ===================================================== */

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
        margin-bottom: 10px;
        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    }

    .metric-label {
        font-size: 13px;
        font-weight: 700;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        margin-top: 4px;
    }


    /* =====================================================
       STREAM
       ===================================================== */

    .stream-title {
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 8px;
    }


    /* =====================================================
       LOADED BOX
       ===================================================== */

    .loaded-box {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid rgba(0,128,0,0.25);
        background: rgba(0,128,0,0.06);
        margin-bottom: 18px;
        animation: contentSlideIn 0.45s ease-out;
    }


    /* =====================================================
       XGBOOST INFO
       ===================================================== */

    .method-box {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
        margin: 10px 0 15px 0;
    }


    /* =====================================================
       HELP PAGE
       ===================================================== */

    .help-page-indicator {
        text-align: center;
        font-size: 12px;
        color: #777;
        margin-top: 5px;
        margin-bottom: 10px;
    }

    .help-page-title {
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 12px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        color: #888;
        font-size: 12px;
        margin-top: 50px;
        padding-top: 15px;
        border-top: 1px solid rgba(128,128,128,0.2);
    }


    /* =====================================================
       HELP DIALOG
       ===================================================== */

    div[data-testid="stDialog"] {
        animation: helpModalIn 0.30s ease-out;
    }

    @keyframes helpModalIn {
        from {
            opacity: 0;
            transform: translateY(-18px) scale(0.97);
        }

        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    div[data-testid="stDialog"] > div {
        border-radius: 18px;
    }

    .help-dialog-content {
        animation: helpContentFade 0.45s ease-out;
    }

    @keyframes helpContentFade {
        from {
            opacity: 0;
            transform: translateY(8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        animation: sidebarSlide 0.35s ease-out;
    }

    @keyframes sidebarSlide {
        from {
            opacity: 0;
            transform: translateX(-15px);
        }

        to {
            opacity: 1;
            transform: translateX(0);
        }
    }


    /* =====================================================
       SIDEBAR MENU ANIMATION
       ===================================================== */

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 10px;
        padding: 5px 8px;
        transition:
            transform 0.20s ease,
            background 0.20s ease,
            box-shadow 0.20s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        transform: translateX(5px);
        background: rgba(128,128,128,0.10);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(128,128,128,0.16);
        box-shadow:
            inset 3px 0 0 currentColor;
        transform: translateX(4px);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        font-weight: 800;
    }


    /* =====================================================
       BUTTON ANIMATION
       ===================================================== */

    .stButton > button {
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 14px rgba(0,0,0,0.10);
    }

    .stButton > button:active {
        transform: scale(0.97);
    }


    /* =====================================================
       PAGE CONTENT ANIMATION
       ===================================================== */

    [data-testid="stAppViewContainer"] .main .block-container {
        animation: pageFadeSlide 0.38s ease-out;
    }

    @keyframes pageFadeSlide {
        from {
            opacity: 0;
            transform: translateY(8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes titleFade {
        from {
            opacity: 0;
            transform: translateX(-8px);
        }

        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes subtitleFade {
        from {
            opacity: 0;
            transform: translateX(-5px);
        }

        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes contentSlideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FORMAT NUMBER
# =========================================================

def format_number(value):

    if value is None:
        return "-"

    try:

        if pd.isna(value):
            return "-"

    except Exception:

        return "-"

    try:

        value = float(value)

    except Exception:

        return "-"

    text = f"{value:,.2f}"

    text = text.replace(",", "TEMP")
    text = text.replace(".", ",")
    text = text.replace("TEMP", ".")

    return text


def format_percent(value):

    if value is None:
        return "-"

    try:

        if pd.isna(value):
            return "-"

    except Exception:

        return "-"

    try:

        value = float(value)

    except Exception:

        return "-"

    return (
        f"{value:.2f}"
        .replace(".", ",")
        + "%"
    )


# =========================================================
# PERIOD
# =========================================================

def forecast_period_text():

    month = st.session_state.forecast_setting["month"]

    year = st.session_state.forecast_setting["year"]

    return f"{BULAN[month]} {year}"


def get_forecast_date():

    return pd.Timestamp(
        year=st.session_state.forecast_setting["year"],
        month=st.session_state.forecast_setting["month"],
        day=1,
    )


# =========================================================
# HISTORY DETECTION
# =========================================================

def get_available_history(df):

    if df is None or df.empty:
        return []

    if "Bulan" not in df.columns:
        return []

    forecast_date = get_forecast_date()

    temp = df.copy()

    temp["_periode"] = temp["Bulan"].apply(
        lambda x: parse_period(
            x,
            default_year=forecast_date.year,
        )
    )

    temp = temp[
        temp["_periode"].notna()
    ].copy()

    temp = temp[
        temp["_periode"] < forecast_date
    ].copy()

    if temp.empty:
        return []

    months = (
        temp["_periode"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return months


# =========================================================
# CHECK FORECAST STATUS
# =========================================================

def get_history_mode(setting):

    """Mengembalikan True jika setting menggunakan seluruh histori."""

    if not isinstance(setting, dict):
        return False

    return bool(
        setting.get(
            "use_all_history",
            False,
        )
    )


def get_effective_history_months(
    setting,
    available_history=None,
):

    """Menentukan jumlah histori efektif yang digunakan."""

    if get_history_mode(setting):

        if available_history is not None:
            return len(available_history)

        return None

    try:
        value = int(
            setting.get(
                "history_months",
                8,
            )
        )
    except Exception:
        value = 8

    return max(1, value)


def get_forecast_history_parameter(setting):

    """Parameter history_months yang dikirim ke forecasting.py."""

    if get_history_mode(setting):
        return None

    return get_effective_history_months(setting)


def forecast_matches_current_setting():

    if not st.session_state.forecast_loaded:
        return False

    current_period = forecast_period_text()

    setting = st.session_state.forecast_setting

    available_history = get_available_history(
        st.session_state.data_out
    )

    current_history = get_effective_history_months(
        setting,
        available_history,
    )

    last_period = (
        st.session_state.last_forecast_period
    )

    last_history = (
        st.session_state.last_forecast_history_months
    )

    return (
        current_period == last_period
        and current_history == last_history
    )


# =========================================================
# DISPLAY FORECAST TABLE
# =========================================================

def display_forecast_table(
    df,
    limit=5,
):
    """Menampilkan Forecast dan WAPE untuk semua metode."""
    if df is None or df.empty:
        st.info(
            "Belum ada hasil forecast."
        )
        return

    show = df.head(limit).copy()

    for col in [
        "Forecast MA",
        "Forecast WMA",
        "Forecast XGBoost",
    ]:
        if col in show.columns:
            show[col] = (
                pd.to_numeric(
                    show[col],
                    errors="coerce",
                )
                .apply(format_number)
            )

    for col in [
        "WAPE MA",
        "WAPE WMA",
        "WAPE XGBoost",
    ]:
        if col in show.columns:
            show[col] = (
                pd.to_numeric(
                    show[col],
                    errors="coerce",
                )
                .apply(format_percent)
            )

    columns = [
        "Nama Barang",
        "Satuan",
        "Forecast MA",
        "WAPE MA",
        "Forecast WMA",
        "WAPE WMA",
        "Forecast XGBoost",
        "WAPE XGBoost",
        "Histori",
    ]

    columns = [
        col
        for col in columns
        if col in show.columns
    ]

    show = show[columns]

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# NORMALIZE LOADED DATA
# =========================================================

def normalize_loaded_dataframe(data):

    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):

        df = data.copy()

    elif isinstance(data, list):

        df = pd.DataFrame(data)

    else:

        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # ---------------------------------------------------------
    # Kompatibilitas history lama
    # ---------------------------------------------------------
    # History sebelum revisi 3-metode hanya memiliki:
    #   Forecast + WAPE (dan pada versi lama dapat memiliki kolom metode)
    # Agar history lama tetap bisa dibuka di Dashboard, nilai
    # Forecast/WAPE lama dipetakan sebagai kolom MA. Kolom WMA
    # dan XGBoost tetap kosong karena data historis tersebut
    # memang tidak menyimpan hasil kedua metode itu.
    if "Forecast MA" not in df.columns and "Forecast" in df.columns:
        df["Forecast MA"] = pd.to_numeric(
            df["Forecast"],
            errors="coerce",
        )

    if "WAPE MA" not in df.columns and "WAPE" in df.columns:
        df["WAPE MA"] = pd.to_numeric(
            df["WAPE"],
            errors="coerce",
        )

    for col in [
        "Histori",
        "WAPE MA",
        "WAPE WMA",
        "WAPE XGBoost",
        "Forecast MA",
        "Forecast WMA",
        "Forecast XGBoost",
        # Kolom legacy tetap dinormalisasi untuk history lama.
        "WAPE",
        "Forecast",
        "Accuracy",
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Pastikan urutan kolom Dashboard konsisten walaupun history
    # lama/baru mempunyai susunan kolom yang berbeda.
    preferred_columns = [
        "Nama Barang",
        "Satuan",
        "Forecast MA",
        "WAPE MA",
        "Forecast WMA",
        "WAPE WMA",
        "Forecast XGBoost",
        "WAPE XGBoost",
        "Histori",
    ]

    existing_preferred = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    remaining_columns = [
        col
        for col in df.columns
        if col not in existing_preferred
        and col not in [
            "Best Method",
            "Forecast",
            "WAPE",
            "Accuracy",
        ]
    ]

    df = df[
        existing_preferred + remaining_columns
    ]

    return df


# =========================================================
# NORMALIZE LOADED SUMMARY
# =========================================================

def normalize_loaded_summary(summary):
    """
    Menyamakan summary history lama dan history 3-metode.

    History lama hanya menyimpan WAPE/Accuracy generik. Nilai
    tersebut diperlakukan sebagai alias MA untuk kompatibilitas.
    WMA dan XGBoost tidak ditebak dari data lama dan tetap None.
    """
    if not isinstance(summary, dict):
        summary = {}

    normalized = {}

    for stream in ["bbb", "bbt"]:
        source = summary.get(stream, {})
        if not isinstance(source, dict):
            source = {}

        target = DEFAULT_SUMMARY[stream].copy()

        # Salin nilai yang memang tersedia pada history baru.
        for key in target:
            if key in source:
                target[key] = source[key]

        # History lama: WAPE/Accuracy generik dipetakan ke MA.
        if target.get("wape_ma") is None:
            legacy_wape = source.get("wape")
            if legacy_wape is not None:
                target["wape_ma"] = legacy_wape

        if target.get("accuracy_ma") is None:
            legacy_accuracy = source.get("accuracy")
            if legacy_accuracy is not None:
                target["accuracy_ma"] = legacy_accuracy

        if target.get("total_actual_ma", 0.0) == 0.0:
            legacy_actual = source.get("total_actual")
            if legacy_actual is not None:
                try:
                    target["total_actual_ma"] = float(legacy_actual)
                except Exception:
                    pass

        if target.get("total_error_ma", 0.0) == 0.0:
            legacy_error = source.get("total_error")
            if legacy_error is not None:
                try:
                    target["total_error_ma"] = float(legacy_error)
                except Exception:
                    pass

        # Alias tetap disediakan untuk kompatibilitas kode/history lama.
        target["wape"] = target.get("wape_ma")
        target["accuracy"] = target.get("accuracy_ma")
        target["total_actual"] = target.get("total_actual_ma", 0.0)
        target["total_error"] = target.get("total_error_ma", 0.0)

        # Tidak ada lagi pemilihan metode.
        target["best_method"] = None

        normalized[stream] = target

    return normalized


# =========================================================
# LOAD FORECAST INTO SESSION
# =========================================================

def load_forecast_to_session(history_id):

    record = load_history_by_id(
        history_id
    )

    if record is None:

        return False, "Data history tidak ditemukan."

    df_bbb = normalize_loaded_dataframe(
        record.get("forecast_bbb")
    )

    df_bbt = normalize_loaded_dataframe(
        record.get("forecast_bbt")
    )

    summary = normalize_loaded_summary(
        record.get("summary", {})
    )

    st.session_state.forecast_bbb = df_bbb

    st.session_state.forecast_bbt = df_bbt

    st.session_state.forecast_summary = summary

    st.session_state.loaded_history_id = history_id

    st.session_state.forecast_loaded = True

    st.session_state.loaded_forecast_info = {
        "id": record["id"],
        "nama_user": record["nama_user"],
        "periode_forecast": record["periode_forecast"],
        "history_months": record["history_months"],
        "created_at": record["created_at"],
    }

    forecast_date = parse_period(
        record["periode_forecast"]
    )

    if forecast_date is not None and not pd.isna(forecast_date):

        st.session_state.forecast_setting = {
            "month": int(
                forecast_date.month
            ),
            "year": int(
                forecast_date.year
            ),
            "history_months": int(
                record["history_months"]
                or 3
            ),
            # Database lama tidak menyimpan mode semua histori.
            # Nilai yang tersimpan adalah jumlah histori aktual.
            "use_all_history": False,
        }

        st.session_state.last_forecast_period = (
            record["periode_forecast"]
        )

        st.session_state.last_forecast_history_months = (
            int(
                record["history_months"]
                or 3
            )
        )

    return True, "Forecast berhasil dimuat."


# =========================================================
# HELP CONTENT
# =========================================================

HELP_CONTENT = {

    "📊 Dashboard": [

        {
            "title": "Dashboard — Fungsi Utama",
            "content": """
### Fungsi Dashboard

Dashboard merupakan halaman ringkasan dari hasil forecasting yang sudah dibuat atau dimuat dari History.

Dashboard digunakan untuk melihat:

- Performance forecast BBB
- Performance forecast BBT
- WAPE MA, WMA, XGBoost
- Forecast Accuracy setiap metode
- Hasil forecast masing-masing item
- Informasi forecast yang sedang dimuat
- Export hasil forecast ke Excel

Dashboard **tidak melakukan perhitungan forecasting baru**.

Perhitungan forecasting dilakukan pada menu:

**🔮 Forecast**

---

### BBB dan BBT

BBB dan BBT selalu dihitung secara **terpisah**.

Artinya:

**OUT BBB → Forecast BBB**

dan

**OUT BBT → Forecast BBT**

Data BBB dan BBT tidak digabungkan menjadi satu demand.
            """,
        },

        {
            "title": "Dashboard — Dari Mana WAPE Berasal?",
            "content": """
### Dari Mana WAPE Berasal?

WAPE yang tampil di Dashboard berasal dari proses:

**Backtesting histori**

Bukan dari bulan forecast yang akan datang.

Kenapa?

Karena ketika sistem melakukan forecast untuk bulan yang akan datang, nilai actual bulan tersebut belum tersedia.

Jadi sistem menggunakan data histori yang sudah memiliki:

- Actual OUT
- Forecast hasil backtesting

Kemudian keduanya dibandingkan untuk mengetahui error.

---

### Apa itu Backtesting?

Backtesting adalah proses seolah-olah sistem sedang melakukan forecast pada periode yang sudah lewat.

Contoh histori:

- Juni
- Juli
- Agustus

Sistem melakukan simulasi forecast pada histori tersebut, kemudian membandingkan hasil forecast dengan actual yang memang sudah diketahui.

Contoh:

**Actual Juni = 1.000**

**Forecast Backtest Juni = 900**

Maka error absolut:

**|1.000 − 900| = 100**

Proses yang sama dilakukan pada periode backtesting lainnya.

Dari seluruh error tersebut kemudian dihitung WAPE.

---

### Penting

Jadi WAPE menunjukkan:

**Seberapa besar error metode forecast ketika diuji menggunakan histori yang actual-nya sudah diketahui.**

WAPE bukan error forecast masa depan.
            """,
        },

        {
            "title": "Dashboard — Contoh WAPE & Accuracy",
            "content": """
### Contoh Perhitungan WAPE

Misalnya hasil backtesting untuk satu item:

| Bulan | Actual | Forecast Backtest | Error Absolut |
|---|---:|---:|---:|
| Juni | 1.000 | 900 | 100 |
| Juli | 1.200 | 1.100 | 100 |
| Agustus | 800 | 900 | 100 |

### Langkah 1 — Total Actual

1.000 + 1.200 + 800

**= 3.000**

### Langkah 2 — Total Error Absolut

100 + 100 + 100

**= 300**

### Langkah 3 — Hitung WAPE

**WAPE = Total Error Absolut ÷ Total Actual × 100%**

**WAPE = 300 ÷ 3.000 × 100%**

**WAPE = 10%**

### Langkah 4 — Forecast Accuracy

**Forecast Accuracy = 100% − WAPE**

**Forecast Accuracy = 100% − 10%**

**Forecast Accuracy = 90%**

---

### Kesimpulan

Pada contoh tersebut:

**WAPE = 10%**

**Forecast Accuracy = 90%**

Semakin kecil WAPE → semakin baik.

Semakin besar Forecast Accuracy → semakin baik.

---

### BBB dan BBT

Perhitungan dilakukan secara terpisah:

**WAPE BBB → berdasarkan histori OUT BBB**

**WAPE BBT → berdasarkan histori OUT BBT**

Keduanya tidak digabungkan.
            """,
        },

        {
            "title": "Dashboard — Forecast & Export",
            "content": """
### Hasil Forecast

Tabel forecast menampilkan hasil setiap metode secara terpisah:

- Nama Barang
- Satuan
- Forecast MA
- WAPE MA
- Forecast WMA
- WAPE WMA
- Forecast XGBoost
- WAPE XGBoost
- Histori

Tidak ada pemilihan metode otomatis; semua hasil metode ditampilkan.

### Export Excel

Hasil forecast dapat diekspor menjadi Excel.

Excel terdiri dari:

1. **Forecast Bulanan**
2. **Detail BBB**
3. **Detail BBT**

Dengan demikian hasil forecast dapat digunakan kembali untuk kebutuhan reporting dan operasional.

---

### Catatan

Dashboard hanya menampilkan hasil yang sudah dihitung.

Untuk membuat forecast baru, gunakan menu:

**🔮 Forecast**
            """,
        },

    ],

    "📦 Data OUT": [

        {
            "title": "Data OUT — Fungsi",
            "content": """
### Fungsi Data OUT

Menu Data OUT digunakan untuk memasukkan histori pemakaian barang ke dalam sistem.

Data inilah yang menjadi dasar forecasting.

Format Excel wajib:

- **Bulan**
- **Nama Barang**
- **Satuan**
- **OUT BBB**
- **OUT BBT**

Pastikan nama kolom sesuai format tersebut.

---

### Alur Data

Secara sederhana:

**Excel OUT**

↓

**Data OUT**

↓

**Validasi**

↓

**Forecast**

Data OUT tidak langsung menjadi forecast.

Data harus melewati proses forecasting terlebih dahulu.
            """,
        },

        {
            "title": "Data OUT — BBB & BBT",
            "content": """
### BBB dan BBT

OUT BBB dan OUT BBT diproses secara terpisah.

Contoh:

| Bulan | Item | OUT BBB | OUT BBT |
|---|---|---:|---:|
| Juni | Burger Bun | 3.000 | 2.000 |
| Juli | Burger Bun | 3.500 | 2.300 |

Sistem tidak menjumlahkan:

**3.000 + 2.000**

sebagai satu demand.

BBB tetap memiliki forecast sendiri.

BBT tetap memiliki forecast sendiri.

---

### Kenapa Dipisahkan?

Karena pola demand BBB dan BBT dapat berbeda.

Contohnya:

- BBB dapat mengalami kenaikan OUT.
- BBT dapat mengalami penurunan OUT.

Jika digabung, pola tersebut akan tercampur dan hasil forecast bisa menjadi kurang representatif.

Karena itu sistem mempertahankan dua stream:

**BBB**

dan

**BBT**
            """,
        },

        {
            "title": "Data OUT — Persiapan Data",
            "content": """
### Sebelum Upload

Pastikan:

- Nama barang konsisten
- Satuan konsisten
- Bulan dapat dibaca sistem
- OUT BBB berupa angka
- OUT BBT berupa angka
- Tidak ada nilai negatif
- Tidak ada data kosong
- Histori tersedia sebelum periode forecast

Jika terdapat masalah, gunakan menu:

**✅ Validasi**

untuk mengetahui bagian mana yang perlu diperbaiki.

---

### Histori

Jumlah histori yang dapat digunakan bersifat fleksibel.

Contoh:

**3 bulan**

atau

**6 bulan**

atau

**12 bulan**

tergantung data yang tersedia dan setting yang dipilih.
            """,
        },

    ],

    "⚙️ Setting": [

        {
            "title": "Setting — Periode Forecast",
            "content": """
### Fungsi Setting

Setting digunakan untuk menentukan periode forecast dan jumlah histori yang akan digunakan.

### Bulan Forecast

Menentukan bulan yang ingin diprediksi.

Contoh:

**September 2026**

berarti sistem akan membuat forecast untuk September 2026.

### Tahun Forecast

Menentukan tahun periode forecast.

---

### Contoh

Jika setting:

**Bulan = September**

**Tahun = 2026**

maka:

**Periode Forecast = September 2026**
            """,
        },

        {
            "title": "Setting — Periode Histori",
            "content": """
### Periode Histori

Menentukan berapa bulan histori yang digunakan untuk forecasting.

Contoh:

**3 bulan**

berarti sistem mengambil 3 periode histori terakhir sebelum bulan forecast.

Misalnya forecast September 2026:

- Juni 2026
- Juli 2026
- Agustus 2026

---

### Jika Memilih 6 Bulan

Sistem akan mengambil 6 periode terakhir sebelum September 2026.

Contohnya:

- Maret
- April
- Mei
- Juni
- Juli
- Agustus

---

### Catatan

Histori harus tersedia sebelum periode forecast.

Jika setting membutuhkan 6 bulan tetapi hanya tersedia 4 bulan, forecast tidak dapat dijalankan.
            """,
        },

    ],

    "✅ Validasi": [

        {
            "title": "Validasi — Pemeriksaan Data",
            "content": """
### Fungsi Validasi

Validasi memastikan data OUT siap digunakan untuk forecasting.

Sistem memeriksa:

1. Kolom wajib
2. Data kosong
3. Nilai OUT
4. Nilai negatif
5. Data duplikat
6. Format periode
7. Ketersediaan histori

---

### Status

**OK**

Data memenuhi pemeriksaan.

**WARNING**

Ada kondisi yang perlu diperhatikan tetapi tidak selalu menghentikan proses.

**ERROR**

Data harus diperbaiki sebelum forecasting.

---

### Tujuan

Validasi membantu memastikan data yang masuk ke proses forecasting memiliki struktur yang benar dan dapat diproses dengan aman.
            """,
        },

        {
            "title": "Validasi — Data Duplikat",
            "content": """
### Data Duplikat

Duplikat dicek berdasarkan:

**Bulan + Nama Barang + Satuan**

Contoh:

| Bulan | Nama Barang | Satuan |
|---|---|---|
| Juni | Burger Bun | Pack |
| Juni | Burger Bun | Pack |

Jika terdapat baris seperti ini, sistem memberikan WARNING.

Data tersebut tetap dapat diagregasi berdasarkan item dan periode ketika forecasting dilakukan.

---

### Kenapa Bukan ERROR?

Karena beberapa sumber data dapat memiliki lebih dari satu baris untuk item dan periode yang sama.

Sistem dapat melakukan agregasi ketika proses forecasting.

Namun user tetap perlu mengetahui bahwa data tersebut memiliki duplikasi.
            """,
        },

    ],

    "🔮 Forecast": [

        {
            "title": "Forecast — Alur Perhitungan",
            "content": """
### Alur Forecasting

Secara sederhana prosesnya:

**Data OUT**

↓

**Ambil Histori**

↓

**Backtesting per metode**

↓

**MA / WMA / XGBoost**

↓

**Hitung WAPE masing-masing metode**

↓

**Forecast masing-masing metode**

---

### BBB dan BBT

Proses tersebut dilakukan secara terpisah.

**OUT BBB → Forecast BBB**

**OUT BBT → Forecast BBT**

Jadi hasil performa BBB tidak dicampur dengan BBT.
            """,
        },

        {
            "title": "Forecast — MA & WMA",
            "content": """
### MA — Moving Average

MA menggunakan rata-rata beberapa periode terakhir.

Contoh:

MA 2:

**Forecast = (Actual bulan 1 + Actual bulan 2) ÷ 2**

Jika:

Bulan 1 = 1.000

Bulan 2 = 1.200

Maka:

**MA 2 = (1.000 + 1.200) ÷ 2**

**= 1.100**

MA cocok digunakan ketika demand relatif stabil.

---

### WMA — Weighted Moving Average

WMA memberikan bobot lebih besar kepada periode yang lebih baru.

Artinya data terbaru memiliki pengaruh lebih besar terhadap forecast.

WMA berguna ketika kondisi demand terbaru dianggap lebih relevan dibandingkan periode yang lebih lama.
            """,
        },

        {
            "title": "Forecast — XGBoost",
            "content": """
### XGBoost

XGBoost adalah metode machine learning yang dapat mempelajari pola historis untuk menghasilkan forecast.

Dalam sistem ini XGBoost dihitung **secara independen** dari MA dan WMA.

XGBoost dievaluasi menggunakan:

**Backtesting**

dan

**WAPE**

Contoh hasil:

| Metode | WAPE |
|---|---:|
| MA | 12,40% |
| WMA | 9,80% |
| XGBoost | 7,50% |

Ketiga hasil tetap ditampilkan. WAPE digunakan untuk melihat performa masing-masing metode, bukan untuk memilih satu metode secara otomatis.
            """,
        },

        {
            "title": "Forecast — Kenapa XGBoost Tidak Dipakai?",
            "content": f"""
### Kenapa XGBoost Bisa Tidak Dipakai?

XGBoost membutuhkan data yang cukup untuk belajar pola.

Jika histori terlalu sedikit, model machine learning dapat menjadi kurang stabil karena jumlah data training dan backtesting terbatas.

Karena itu sistem menggunakan batas minimum:

**Minimal histori XGBoost = {XGBOOST_MIN_HISTORY} bulan**

---

### Contoh

Jika histori tersedia:

**3 bulan**

→ XGBoost dapat digunakan dan dievaluasi secara independen melalui backtesting.

Jika histori tersedia:

**6 bulan**

→ XGBoost dapat digunakan dan dievaluasi secara independen melalui backtesting.

Jika histori tersedia:

**{XGBOOST_MIN_HISTORY} bulan atau lebih**

→ XGBoost dapat digunakan dan dievaluasi secara independen, selama package XGBoost tersedia.

---

### Kenapa Ada Batas?

Tujuannya bukan untuk mengatakan XGBoost jelek.

Tujuannya agar sistem tidak memaksakan machine learning ketika data histori terlalu sedikit.

Untuk histori terbatas, metode statistik seperti:

**MA**

dan

**WMA**

lebih sesuai untuk dijadikan kandidat.

---

### Jika Package Tidak Tersedia

Jika package XGBoost belum ter-install, XGBoost juga tidak akan digunakan.

Ini bukan error pada proses forecasting.

Sistem akan tetap menggunakan metode forecasting yang tersedia.
            """,
        },

        {
            "title": "Forecast — Tiga Metode Secara Independen",
            "content": """
### Tiga Metode Forecast

Sistem menghitung tiga metode secara **independen**:

1. **MA**
2. **WMA**
3. **XGBoost** (jika tersedia dan memenuhi syarat histori)

Setiap metode menghasilkan:

- Forecast sendiri
- WAPE sendiri
- Forecast Accuracy sendiri

### Tidak Ada Pemilihan Metode Otomatis

Sistem **tidak memilih satu metode secara otomatis**; MA, WMA, dan XGBoost ditampilkan bersamaan.

Contoh:

| Metode | Forecast | WAPE |
|---|---:|---:|
| MA | 150 | 12% |
| WMA | 155 | 9% |
| XGBoost | 153 | 7% |

Ketiga hasil tetap ditampilkan agar user dapat membandingkan performa masing-masing metode.

### Recursive Forecast

Jika target lebih jauh dari actual terakhir, proses recursive dilakukan **secara terpisah untuk setiap metode**.

Contoh actual sampai Agustus dan target Oktober:

- MA: forecast September → forecast Oktober
- WMA: forecast September → forecast Oktober
- XGBoost: forecast September → forecast Oktober, jika tersedia

Actual intermediate selalu meng-override forecast recursive.

Forecast recursive tidak digunakan sebagai actual untuk menghitung WAPE.
            """,
        },

        {
            "title": "Forecast — Dari Mana WAPE Berasal?",
            "content": """
### WAPE Berasal dari Backtesting

WAPE setiap metode berasal dari hasil pengujian metode tersebut terhadap histori actual.

Sistem tidak menghitung WAPE dari forecast masa depan.

MA memiliki WAPE sendiri, WMA memiliki WAPE sendiri, dan XGBoost memiliki WAPE sendiri jika tersedia.

Alurnya:

**Histori Actual**

↓

**Simulasikan Forecast**

↓

**Forecast Backtest**

↓

**Bandingkan dengan Actual**

↓

**Hitung Error Absolut**

↓

**Hitung WAPE**

---

### Contoh Sederhana

Misalnya:

| Periode | Actual | Forecast Backtest | Error |
|---|---:|---:|---:|
| Juni | 1.000 | 900 | 100 |
| Juli | 1.200 | 1.100 | 100 |
| Agustus | 800 | 900 | 100 |

Total Actual:

**3.000**

Total Error:

**300**

Maka:

**WAPE = 300 ÷ 3.000 × 100%**

**= 10%**

---

### Apa Artinya?

Metode tersebut menghasilkan error agregat sebesar:

**10% terhadap total actual pada data backtesting.**

Kemudian metode lain juga diuji secara independen.

Hasil WAPE setiap metode tetap ditampilkan dan tidak digunakan untuk memilih satu metode secara otomatis.
            """,
        },

        {
            "title": "Forecast — WAPE & Accuracy",
            "content": """
### WAPE

WAPE adalah Weighted Absolute Percentage Error.

Rumus yang digunakan sistem:

**WAPE = Σ |Actual − Forecast Backtest| ÷ Σ |Actual| × 100%**

Keterangan:

**Actual**

= nilai OUT yang benar-benar terjadi pada histori.

**Forecast Backtest**

= hasil forecast simulasi untuk periode histori tersebut.

**Error Absolut**

= |Actual − Forecast Backtest|

---

### Forecast Accuracy

Rumus:

**Forecast Accuracy = 100% − WAPE**

Contoh:

WAPE = **10%**

maka:

**Forecast Accuracy = 90%**

---

### Prinsip Penilaian

**WAPE semakin kecil → semakin baik**

**Forecast Accuracy semakin besar → semakin baik**

---

### BBB dan BBT

Perhitungan dilakukan secara terpisah.

**WAPE BBB**

dihitung dari actual dan forecast backtest BBB.

**WAPE BBT**

dihitung dari actual dan forecast backtest BBT.

Tidak digabungkan menjadi satu nilai.

---

### Catatan Penting

WAPE adalah ukuran performa berdasarkan histori.

Untuk bulan forecast yang belum terjadi, actual belum tersedia sehingga error aktual bulan tersebut belum bisa dihitung.
            """,
        },


        {
            "title": "Forecast — Recursive Forecasting (Bertahap)",
            "content": """
### Recursive Forecasting

Sistem menggunakan **recursive forecasting** jika periode target lebih jauh dari actual terakhir yang tersedia. Forecast dihitung **bulan demi bulan** dan dilakukan secara terpisah untuk MA, WMA, dan XGBoost.

### Contoh: Actual sampai Agustus, Target Oktober

Jika actual tersedia Januari sampai Agustus dan target adalah Oktober:

- **MA:** forecast September → forecast Oktober
- **WMA:** forecast September → forecast Oktober
- **XGBoost:** forecast September → forecast Oktober, jika tersedia

Jadi sistem tidak melompati September.

---

### Jika Actual Intermediate Sudah Tersedia

Actual selalu lebih diprioritaskan daripada forecast sementara.

Contoh:

**Januari–Agustus Actual + September Actual → Forecast Oktober**

Jika September sudah memiliki actual, sistem menggunakan actual September.

---

### Contoh Target November

Jika actual terakhir Agustus dan target November:

**Forecast September → Forecast Oktober → Forecast November**

Namun bila actual Oktober tersedia:

**Forecast September → Actual Oktober → Forecast November**

Mekanisme ini diterapkan secara independen untuk setiap metode.

---

### Apakah Forecast Recursive Masuk ke WAPE?

**Tidak.**

WAPE setiap metode tetap dihitung dari **backtesting terhadap actual historis**.

Jadi:

- **WAPE MA** = performa MA pada backtesting histori.
- **WAPE WMA** = performa WMA pada backtesting histori.
- **WAPE XGBoost** = performa XGBoost pada backtesting histori jika tersedia.
- Forecast recursive hanya digunakan untuk mencapai target masa depan.

---

### Ringkasan

1. **MA, WMA, dan XGBoost dihitung secara terpisah.**
2. **Tidak ada pemilihan metode otomatis.**
3. **Target jauh dihitung bertahap per bulan.**
4. **Actual intermediate selalu lebih diprioritaskan.**
5. **Forecast recursive tidak digunakan sebagai actual untuk WAPE.**
            """,
        },
    ],
    "🕘 History": [

        {
            "title": "History — Fungsi",
            "content": """
### Fungsi History

History menyimpan hasil forecasting yang sudah dibuat.

Data yang disimpan antara lain:

- Nama user
- Periode forecast
- Jumlah histori
- Forecast BBB
- Forecast BBT
- Summary performance
- Tanggal penyimpanan

History digunakan agar hasil forecast dapat dibuka kembali tanpa melakukan perhitungan ulang.
            """,
        },

        {
            "title": "History — Load",
            "content": """
### Load History

User dapat memilih hasil forecast dari daftar History.

Setelah memilih:

**📂 Load History**

sistem akan:

1. Membaca data dari database.
2. Memuat forecast BBB.
3. Memuat forecast BBT.
4. Memuat WAPE.
5. Memuat Forecast Accuracy.
6. Memuat performa WAPE dan Forecast Accuracy setiap metode.
7. Mengembalikan setting forecast.
8. Mengarahkan user ke Dashboard.

Dengan demikian hasil lama dapat langsung direview kembali.

---

### Keuntungan

Forecast yang sudah disimpan tidak perlu dihitung ulang hanya untuk melihat hasilnya.
            """,
        },

        {
            "title": "History — Delete",
            "content": """
### Delete History

History dapat dihapus satu per satu.

Tersedia juga opsi:

**Hapus Semua History**

Gunakan fungsi ini dengan hati-hati karena seluruh hasil forecasting yang tersimpan akan dihapus dari database aplikasi.

---

### Catatan

History tidak menghapus Data OUT yang sedang berada di session aplikasi.

History dan Data OUT merupakan dua bagian penyimpanan yang berbeda.
            """,
        },

    ],
}


# =========================================================
# COMPLETE HELP CONTENT
# =========================================================
#
# HELP_CONTENT di atas tetap menjadi sumber bantuan per-menu.
# HELP_ALL_PAGES hanya membuat versi gabungan untuk tombol
# Bantuan Lengkap di sidebar.
#
# Setiap halaman diberi nama menu supaya pengguna selalu tahu
# bagian aplikasi yang sedang dijelaskan.

HELP_ALL_PAGES = [
    {
        "title": "📖 Bantuan Lengkap — Panduan Aplikasi",
        "content": """
### Selamat Datang di Demand Planning System

Halaman **Bantuan Lengkap** berisi seluruh keterangan dari setiap menu aplikasi.

Urutan panduan:

1. 📊 Dashboard
2. 📦 Data OUT
3. ⚙️ Setting
4. ✅ Validasi
5. 🔮 Forecast
6. 🕘 History

### Cara Membaca Bantuan

Gunakan tombol **Next →** untuk membaca bagian berikutnya dan **← Back** untuk kembali.

Contoh-contoh di halaman berikut menggunakan alur sederhana agar fungsi aplikasi mudah dipahami.

### Alur Singkat Penggunaan

**Data OUT → Setting → Validasi → Forecast → Dashboard / History**

Data OUT menjadi sumber histori. Setting menentukan target forecast dan periode histori. Validasi memastikan data layak digunakan. Forecast melakukan perhitungan. Dashboard menampilkan hasil, sedangkan History menyimpan dan memuat kembali hasil forecast.
        """,
    },
]

for _menu_name in MENU_OPTIONS:
    for _page in HELP_CONTENT.get(_menu_name, []):
        HELP_ALL_PAGES.append(
            {
                "title": f"{_menu_name} — {_page.get('title', 'Bantuan')}",
                "content": _page.get("content", ""),
            }
        )

HELP_CONTENT["__ALL__"] = HELP_ALL_PAGES


def open_help(menu_name, page=1):
    """Membuka bantuan dengan konteks menu tertentu."""

    st.session_state.help_menu = menu_name
    st.session_state.help_page = int(page)
    st.session_state.show_help = True
    st.rerun()


def render_context_help(menu_name, key_suffix):
    """Tombol bantuan yang hanya membuka dokumentasi menu aktif."""

    if st.button(
        f"❓ Bantuan {menu_name}",
        use_container_width=False,
        key=key_suffix,
        help=f"Buka bantuan khusus untuk {menu_name}.",
    ):
        open_help(menu_name, 1)



# =========================================================
# HELP DIALOG
# =========================================================

@st.dialog("📖 Bantuan")
def help_dialog():

    menu_name = st.session_state.get(
        "help_menu",
        "📊 Dashboard",
    )

    pages = HELP_CONTENT.get(
        menu_name,
        HELP_CONTENT["📊 Dashboard"],
    )

    total_pages = len(pages)

    current_page = int(
        st.session_state.get(
            "help_page",
            1,
        )
    )

    current_page = max(
        1,
        min(
            current_page,
            total_pages,
        ),
    )

    st.session_state.help_page = current_page

    page_data = pages[
        current_page - 1
    ]

    st.markdown(
        f"""
        <div class="help-dialog-content">
            <div class="help-page-title">
                {page_data["title"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        page_data["content"]
    )

    st.divider()

    st.markdown(
        f"""
        <div class="help-page-indicator">
            Halaman {current_page} dari {total_pages}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1, 1, 1]
    )

    with col1:

        if current_page > 1:

            if st.button(
                "← Back",
                use_container_width=True,
                key=f"help_back_{current_page}",
            ):

                st.session_state.help_page = (
                    current_page - 1
                )

                st.rerun()

    with col2:

        if st.button(
            "✕ Tutup",
            use_container_width=True,
            key=f"help_close_{current_page}",
        ):

            st.session_state.show_help = False

            st.rerun()

    with col3:

        if current_page < total_pages:

            if st.button(
                "Next →",
                type="primary",
                use_container_width=True,
                key=f"help_next_{current_page}",
            ):

                st.session_state.help_page = (
                    current_page + 1
                )

                st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## 🍔 Demand Planning"
)


# ---------------------------------------------------------
# INITIAL MENU DARI QUERY PARAMETER
# ---------------------------------------------------------

query_menu = st.query_params.get(
    "menu",
    "dashboard",
)

initial_menu = MENU_QUERY_MAP.get(
    query_menu,
    "📊 Dashboard",
)

initial_index = MENU_OPTIONS.index(
    initial_menu
)


# Simpan menu sebelumnya supaya dialog bantuan tidak pernah
# terbuka otomatis hanya karena pengguna berpindah menu.
_previous_menu = st.session_state.get(
    "_previous_sidebar_menu",
    initial_menu,
)


menu = st.sidebar.radio(
    "MENU",
    MENU_OPTIONS,
    index=initial_index,
    key="sidebar_menu",
)


# Jika pengguna berpindah menu, bantuan yang sedang terbuka
# langsung ditutup. Bantuan hanya boleh muncul setelah pengguna
# menekan tombol bantuan secara eksplisit.
if menu != _previous_menu:
    st.session_state.show_help = False
    st.session_state.help_page = 1


st.session_state._previous_sidebar_menu = menu


# Update URL sesuai menu yang dipilih

current_query_menu = MENU_REVERSE_MAP.get(
    menu,
    "dashboard",
)

if st.query_params.get("menu") != current_query_menu:

    st.query_params["menu"] = current_query_menu


st.sidebar.divider()


if st.sidebar.button(
    "📖 Bantuan Lengkap",
    use_container_width=True,
    key="sidebar_help_button",
    help="Buka panduan lengkap untuk seluruh menu aplikasi.",
):

    open_help("__ALL__", 1)


st.sidebar.divider()


st.sidebar.caption(
    "Main Warehouse Batu Ceper"
)


# =========================================================
# HEADER
# =========================================================

col_title, col_help = st.columns(
    [8, 1]
)


with col_title:

    st.markdown(
        '<div class="main-title">🍔 DEMAND PLANNING SYSTEM</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">Main Warehouse Batu Ceper</div>',
        unsafe_allow_html=True,
    )


with col_help:

    if st.button(
        "❓",
        help=f"Bantuan khusus {menu}",
        key="header_help_button",
    ):

        open_help(menu, 1)


# =========================================================
# SHOW HELP DIALOG
# =========================================================

if st.session_state.show_help:

    help_dialog()


# =========================================================
# DASHBOARD
# =========================================================

if menu == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Dashboard</div>',
        unsafe_allow_html=True,
    )

    render_context_help(
        "📊 Dashboard",
        "context_help_dashboard",
    )

    # -----------------------------------------------------
    # LOAD FORECAST
    # -----------------------------------------------------

    st.markdown(
        "### Load Forecasting"
    )

    histories = load_history()

    if histories:

        history_options = {}

        for row in histories:

            label = (
                f"Forecast "
                f"{row['periode_forecast']} "
                f"— "
                f"{row['nama_user']} "
                f"(ID {row['id']})"
            )

            history_options[label] = row["id"]

        selected_label = st.selectbox(
            "Pilih Forecast",
            list(history_options.keys()),
            key="dashboard_history_select",
        )

        selected_id = history_options[
            selected_label
        ]

        if st.button(
            "LOAD",
            type="primary",
            key="dashboard_load_button",
        ):

            success, message = (
                load_forecast_to_session(
                    selected_id
                )
            )

            if success:

                st.success(
                    message
                )

                st.rerun()

            else:

                st.error(
                    message
                )

    else:

        st.info(
            "Belum ada forecast yang tersimpan."
        )

    # -----------------------------------------------------
    # LOADED INFORMATION
    # -----------------------------------------------------

    if (
        st.session_state.forecast_loaded
        and st.session_state.loaded_forecast_info
    ):

        info = (
            st.session_state.loaded_forecast_info
        )

        st.markdown(
            f"""
            <div class="loaded-box">
                <b>✓ Forecast sedang dimuat</b><br>
                Forecast {info["periode_forecast"]}
                —
                {info["nama_user"]}
                <br>
                <small>
                    Histori: {info["history_months"]} bulan<br>
                    Disimpan: {info["created_at"]}
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -----------------------------------------------------
    # PERFORMANCE
    # -----------------------------------------------------

    st.markdown(
        "### Performance Forecasting"
    )

    summary = (
        st.session_state.forecast_summary
    )

    summary_bbb = summary.get(
        "bbb",
        {},
    )

    summary_bbt = summary.get(
        "bbt",
        {},
    )

    # Dashboard hanya menampilkan performance jika memang ada
    # hasil forecast yang sudah dimuat. Ini mencegah Dashboard
    # terlihat seolah-olah memiliki nilai performance padahal
    # belum ada forecast pada session.
    has_loaded_forecast = (
        st.session_state.forecast_loaded
        and (
            not st.session_state.forecast_bbb.empty
            or not st.session_state.forecast_bbt.empty
        )
    )

    def render_method_performance(
        stream_label,
        stream_summary,
    ):
        st.markdown(
            f'<div class="stream-title">{stream_label}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        methods = [
            ("MA", "wape_ma", "accuracy_ma"),
            ("WMA", "wape_wma", "accuracy_wma"),
            ("XGBoost", "wape_xgboost", "accuracy_xgboost"),
        ]

        for col, (method_name, wape_key, accuracy_key) in zip(
            [c1, c2, c3],
            methods,
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            WAPE {method_name}
                        </div>
                        <div class="metric-value">
                            {format_percent(
                                stream_summary.get(wape_key)
                            )}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"Forecast Accuracy {method_name}: "
                    f"{format_percent(
                        stream_summary.get(accuracy_key)
                    )}"
                )

    if has_loaded_forecast:

        render_method_performance(
            "PERFORMANCE BBB",
            summary_bbb,
        )

        render_method_performance(
            "PERFORMANCE BBT",
            summary_bbt,
        )

        st.caption(
            "WAPE dan Forecast Accuracy berasal dari backtesting "
            "histori. Setiap metode dihitung secara independen; "
            "tidak ada pemilihan satu metode otomatis."
        )

        # Tabel ringkas agar Dashboard lebih mudah dibandingkan
        # dengan hasil pada Forecast/Excel.
        comparison_df = pd.DataFrame(
            [
                {
                    "Stream": "BBB",
                    "Metode": "MA",
                    "WAPE": summary_bbb.get("wape_ma"),
                    "Forecast Accuracy": summary_bbb.get("accuracy_ma"),
                },
                {
                    "Stream": "BBB",
                    "Metode": "WMA",
                    "WAPE": summary_bbb.get("wape_wma"),
                    "Forecast Accuracy": summary_bbb.get("accuracy_wma"),
                },
                {
                    "Stream": "BBB",
                    "Metode": "XGBoost",
                    "WAPE": summary_bbb.get("wape_xgboost"),
                    "Forecast Accuracy": summary_bbb.get("accuracy_xgboost"),
                },
                {
                    "Stream": "BBT",
                    "Metode": "MA",
                    "WAPE": summary_bbt.get("wape_ma"),
                    "Forecast Accuracy": summary_bbt.get("accuracy_ma"),
                },
                {
                    "Stream": "BBT",
                    "Metode": "WMA",
                    "WAPE": summary_bbt.get("wape_wma"),
                    "Forecast Accuracy": summary_bbt.get("accuracy_wma"),
                },
                {
                    "Stream": "BBT",
                    "Metode": "XGBoost",
                    "WAPE": summary_bbt.get("wape_xgboost"),
                    "Forecast Accuracy": summary_bbt.get("accuracy_xgboost"),
                },
            ]
        )

        comparison_df["WAPE"] = comparison_df["WAPE"].apply(
            format_percent
        )
        comparison_df["Forecast Accuracy"] = comparison_df[
            "Forecast Accuracy"
        ].apply(format_percent)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Belum ada hasil forecast yang dimuat. "
            "Jalankan forecast pada menu **🔮 Forecast** atau "
            "gunakan **LOAD** untuk membuka hasil dari History."
        )

    st.divider()

    # -----------------------------------------------------
    # FORECAST TABLES
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Forecast BBB"
        )

        df_bbb = (
            st.session_state.forecast_bbb
        )

        if df_bbb.empty:

            st.info(
                "Belum ada forecast BBB."
            )

        else:

            display_forecast_table(
                df_bbb,
                limit=min(20, len(df_bbb)),
            )

            if len(df_bbb) > 20:
                st.caption(
                    f"Menampilkan 20 dari "
                    f"{len(df_bbb)} item. "
                    "Buka menu Forecast untuk melihat hasil lengkap."
                )
            else:
                st.caption(
                    f"Menampilkan {len(df_bbb)} item."
                )

    with col2:

        st.markdown(
            "### Forecast BBT"
        )

        df_bbt = (
            st.session_state.forecast_bbt
        )

        if df_bbt.empty:

            st.info(
                "Belum ada forecast BBT."
            )

        else:

            display_forecast_table(
                df_bbt,
                limit=min(20, len(df_bbt)),
            )

            if len(df_bbt) > 20:
                st.caption(
                    f"Menampilkan 20 dari "
                    f"{len(df_bbt)} item. "
                    "Buka menu Forecast untuk melihat hasil lengkap."
                )
            else:
                st.caption(
                    f"Menampilkan {len(df_bbt)} item."
                )

    # -----------------------------------------------------
    # EXPORT EXCEL
    # -----------------------------------------------------

    if (
        not st.session_state.forecast_bbb.empty
        or not st.session_state.forecast_bbt.empty
    ):

        st.divider()

        st.markdown(
            "### Export Forecast"
        )

        if not forecast_matches_current_setting():

            st.warning(
                "Hasil forecast tidak sesuai dengan "
                "setting saat ini. Jalankan forecast "
                "kembali sebelum export."
            )

        else:

            try:

                export_data = export_forecast_excel(
                    forecast_bbb=(
                        st.session_state.forecast_bbb
                    ),
                    forecast_bbt=(
                        st.session_state.forecast_bbt
                    ),
                    periode_forecast=(
                        forecast_period_text()
                    ),
                    nama_user=(
                        (
                            st.session_state
                            .loaded_forecast_info
                            or {}
                        ).get(
                            "nama_user",
                            "Demand Planner",
                        )
                    ),
                )

                filename = generate_export_filename(
                    forecast_period_text()
                )

                st.download_button(
                    label="📥 Export Excel",
                    data=export_data,
                    file_name=filename,
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                    type="primary",
                    key="dashboard_export_button",
                )

                st.caption(
                    "Excel berisi Forecast Bulanan, "
                    "Detail BBB, dan Detail BBT."
                )

            except Exception as e:

                st.error(
                    f"Gagal membuat file Excel: {e}"
                )


# =========================================================
# DATA OUT
# =========================================================

elif menu == "📦 Data OUT":

    st.markdown(
        '<div class="section-title">📦 Data OUT</div>',
        unsafe_allow_html=True,
    )

    render_context_help(
        "📦 Data OUT",
        "context_help_data_out",
    )

    st.info(
        "Upload Excel dengan format: "
        "Bulan, Nama Barang, Satuan, OUT BBB, OUT BBT"
    )

    uploaded_file = st.file_uploader(
        "Upload Excel",
        type=["xlsx", "xls"],
    )

    if uploaded_file is not None:

        try:

            df = pd.read_excel(
                uploaded_file
            )

            required_columns = [
                "Bulan",
                "Nama Barang",
                "Satuan",
                "OUT BBB",
                "OUT BBT",
            ]

            missing = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing:

                st.error(
                    "Kolom tidak ditemukan: "
                    + ", ".join(missing)
                )

                st.session_state.data_out = (
                    pd.DataFrame()
                )

                st.session_state.data_valid = False

                reset_forecast_session()

            else:

                st.session_state.data_out = (
                    df.copy()
                )

                st.session_state.data_valid = False

                reset_forecast_session()

                st.success(
                    "Data OUT berhasil dimuat."
                )

                available_months = (
                    get_available_history(df)
                )

                if available_months:

                    month_text = ", ".join(
                        [
                            f"{BULAN[x.month]} {x.year}"
                            for x in available_months
                        ]
                    )

                    st.info(
                        f"Histori terdeteksi: "
                        f"{month_text}"
                    )

                else:

                    st.warning(
                        "Tidak ditemukan histori "
                        "sebelum periode forecast."
                    )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:

            st.error(
                f"Gagal membaca file: {e}"
            )

    elif not st.session_state.data_out.empty:

        st.dataframe(
            st.session_state.data_out,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# SETTING
# =========================================================

elif menu == "⚙️ Setting":

    st.markdown(
        '<div class="section-title">⚙️ Setting Forecast</div>',
        unsafe_allow_html=True,
    )

    render_context_help(
        "⚙️ Setting",
        "context_help_setting",
    )

    current = (
        st.session_state.forecast_setting
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        month_name = st.selectbox(
            "Bulan Forecast",
            list(BULAN.values()),
            index=current["month"] - 1,
            key="setting_month",
        )

        selected_month = next(
            key
            for key, value in BULAN.items()
            if value == month_name
        )

    with col2:

        selected_year = st.selectbox(
            "Tahun Forecast",
            list(range(2024, 2036)),
            index=(
                current["year"] - 2024
                if 2024 <= current["year"] <= 2035
                else 2
            ),
            key="setting_year",
        )

    with col3:

        # REVISI HISTORI:
        # Maksimum diperbesar agar dataset dengan histori panjang
        # tidak terpaksa dipotong hanya 24 bulan dari sisi UI.
        selected_history = st.number_input(
            "Periode Histori (bulan)",
            min_value=1,
            max_value=120,
            value=int(
                current["history_months"]
            ),
            step=1,
            key="setting_history",
        )

        selected_all_history = st.checkbox(
            "Gunakan semua histori tersedia",
            value=get_history_mode(current),
            key="setting_all_history",
            help=(
                "Jika aktif, sistem menggunakan seluruh histori "
                "yang tersedia sebelum periode forecast."
            ),
        )

        if selected_all_history:

            st.caption(
                "✓ Mode semua histori aktif. Nilai periode bulan "
                "di atas tidak membatasi histori yang digunakan."
            )

    if st.button(
        "💾 Simpan Setting",
        type="primary",
        key="save_setting_button",
    ):

        old_setting = (
            st.session_state.forecast_setting.copy()
        )

        new_setting = {
            "month": selected_month,
            "year": selected_year,
            "history_months": int(
                selected_history
            ),
            "use_all_history": bool(
                selected_all_history
            ),
        }

        setting_changed = (
            old_setting != new_setting
        )

        st.session_state.forecast_setting = (
            new_setting
        )

        if setting_changed:

            reset_forecast_session()

            st.success(
                "Setting berhasil disimpan. "
                "Hasil forecast lama di-reset karena "
                "periode/setting berubah."
            )

        else:

            st.success(
                "Setting berhasil disimpan."
            )

    st.divider()

    setting = (
        st.session_state.forecast_setting
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Setting": [
                    "Bulan Forecast",
                    "Tahun Forecast",
                    "Periode Histori",
                ],
                "Nilai": [
                    BULAN[setting["month"]],
                    setting["year"],
                    (
                        "Semua histori tersedia"
                        if get_history_mode(setting)
                        else setting["history_months"]
                    ),
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "💡 Rekomendasi: aktifkan **Gunakan semua histori tersedia** "
        "untuk mengambil seluruh bulan sebelum periode forecast. "
        "Jika mode manual digunakan dan tersedia 8 bulan histori, "
        "gunakan **Periode Histori = 8**. Forecast tetap hanya "
        "mengambil data sebelum periode forecast. Jika target lebih jauh "
        "dari actual terakhir, forecasting dilakukan secara recursive per "
        "bulan sampai target; actual intermediate selalu meng-override "
        "forecast dan forecast recursive tidak masuk WAPE."
    )


# =========================================================
# VALIDASI
# =========================================================

elif menu == "✅ Validasi":

    st.markdown(
        '<div class="section-title">✅ Validasi Data</div>',
        unsafe_allow_html=True,
    )

    render_context_help(
        "✅ Validasi",
        "context_help_validasi",
    )

    df = st.session_state.data_out

    if df.empty:

        st.warning(
            "Belum ada Data OUT."
        )

        st.session_state.data_valid = False

    else:

        required_columns = [
            "Bulan",
            "Nama Barang",
            "Satuan",
            "OUT BBB",
            "OUT BBT",
        ]

        validation_results = []

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        validation_results.append(
            {
                "Validasi": "Kolom Wajib",
                "Status": (
                    "OK"
                    if not missing
                    else "ERROR"
                ),
                "Keterangan": (
                    "Semua kolom tersedia"
                    if not missing
                    else
                    "Missing: "
                    + ", ".join(missing)
                ),
            }
        )

        # -------------------------------------------------
        # DATA KOSONG
        # -------------------------------------------------

        blank_count = 0

        if not missing:

            blank_mask = (
                df[
                    [
                        "Bulan",
                        "Nama Barang",
                        "Satuan",
                        "OUT BBB",
                        "OUT BBT",
                    ]
                ]
                .isna()
                .any(axis=1)
            )

            blank_count = int(
                blank_mask.sum()
            )

            for col in [
                "Bulan",
                "Nama Barang",
                "Satuan",
            ]:

                blank_count += int(
                    df[col]
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                )

        validation_results.append(
            {
                "Validasi": "Data Kosong",
                "Status": (
                    "OK"
                    if blank_count == 0
                    else "ERROR"
                ),
                "Keterangan": (
                    "Tidak ada data kosong"
                    if blank_count == 0
                    else
                    f"{blank_count} data kosong"
                ),
            }
        )

        # -------------------------------------------------
        # NILAI NUMERIK
        # -------------------------------------------------

        numeric_error = 0

        if not missing:

            for col in [
                "OUT BBB",
                "OUT BBT",
            ]:

                converted = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                numeric_error += int(
                    converted.isna().sum()
                )

        validation_results.append(
            {
                "Validasi": "Nilai OUT",
                "Status": (
                    "OK"
                    if numeric_error == 0
                    else "ERROR"
                ),
                "Keterangan": (
                    "Nilai numerik valid"
                    if numeric_error == 0
                    else
                    f"{numeric_error} nilai tidak valid"
                ),
            }
        )

        # -------------------------------------------------
        # NILAI NEGATIF
        # -------------------------------------------------

        negative_count = 0

        if not missing:

            for col in [
                "OUT BBB",
                "OUT BBT",
            ]:

                numeric = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                negative_count += int(
                    (numeric < 0)
                    .fillna(False)
                    .sum()
                )

        validation_results.append(
            {
                "Validasi": "Nilai Negatif",
                "Status": (
                    "OK"
                    if negative_count == 0
                    else "ERROR"
                ),
                "Keterangan": (
                    "Tidak ada nilai negatif"
                    if negative_count == 0
                    else
                    f"{negative_count} nilai negatif"
                ),
            }
        )

        # -------------------------------------------------
        # DUPLIKAT
        # -------------------------------------------------

        duplicate_count = 0

        if not missing:

            duplicate_count = int(
                df.duplicated(
                    subset=[
                        "Bulan",
                        "Nama Barang",
                        "Satuan",
                    ],
                    keep=False,
                ).sum()
            )

        validation_results.append(
            {
                "Validasi": "Data Duplikat",
                "Status": (
                    "OK"
                    if duplicate_count == 0
                    else "WARNING"
                ),
                "Keterangan": (
                    "Tidak ada duplikat"
                    if duplicate_count == 0
                    else
                    f"{duplicate_count} baris terindikasi duplikat"
                ),
            }
        )

        # -------------------------------------------------
        # PERIODE
        # -------------------------------------------------

        invalid_period_count = 0

        if "Bulan" in df.columns:

            forecast_date = get_forecast_date()

            parsed_periods = df["Bulan"].apply(
                lambda x: parse_period(
                    x,
                    default_year=forecast_date.year,
                )
            )

            invalid_period_count = int(
                parsed_periods.isna().sum()
            )

        validation_results.append(
            {
                "Validasi": "Format Periode",
                "Status": (
                    "OK"
                    if invalid_period_count == 0
                    else "ERROR"
                ),
                "Keterangan": (
                    "Semua periode dapat dibaca"
                    if invalid_period_count == 0
                    else
                    f"{invalid_period_count} periode tidak dapat dibaca"
                ),
            }
        )

        # -------------------------------------------------
        # PERIODE HISTORI
        # -------------------------------------------------

        available_history = (
            get_available_history(df)
        )

        current_setting = (
            st.session_state.forecast_setting
        )

        history_required = get_effective_history_months(
            current_setting,
            available_history,
        )

        if get_history_mode(current_setting):

            if len(available_history) > 0:

                history_status = "OK"

                history_description = (
                    f"Tersedia {len(available_history)} bulan histori "
                    "dan mode semua histori aktif; seluruh histori "
                    "sebelum periode forecast akan digunakan"
                )

            else:

                history_status = "ERROR"

                history_description = (
                    "Tidak ada histori sebelum periode forecast"
                )

        elif len(available_history) >= history_required:

            history_status = "OK"

            history_description = (
                f"Tersedia {len(available_history)} bulan "
                f"histori, membutuhkan {history_required} bulan"
            )

        elif len(available_history) > 0:

            history_status = "ERROR"

            history_description = (
                f"Hanya tersedia {len(available_history)} bulan "
                f"histori, membutuhkan {history_required} bulan"
            )

        else:

            history_status = "ERROR"

            history_description = (
                "Tidak ada histori sebelum periode forecast"
            )

        validation_results.append(
            {
                "Validasi": "Ketersediaan Histori",
                "Status": history_status,
                "Keterangan": history_description,
            }
        )

        validation_df = pd.DataFrame(
            validation_results
        )

        st.dataframe(
            validation_df,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # STATUS DATA
        # -------------------------------------------------

        critical_statuses = [
            row["Status"]
            for row in validation_results
            if row["Validasi"] != "Data Duplikat"
        ]

        critical_ok = all(
            status == "OK"
            for status in critical_statuses
        )

        if critical_ok:

            st.session_state.data_valid = True

            st.success(
                "Data siap digunakan untuk forecasting."
            )

        else:

            st.session_state.data_valid = False

            st.error(
                "Masih terdapat masalah pada data. "
                "Perbaiki validasi ERROR sebelum forecasting."
            )

        if duplicate_count > 0:

            st.warning(
                "Terdapat data duplikat. "
                "Duplikat akan tetap diagregasi berdasarkan "
                "Bulan + Nama Barang + Satuan saat forecasting."
            )


# =========================================================
# FORECAST
# =========================================================

elif menu == "🔮 Forecast":

    st.markdown(
        '<div class="section-title">🔮 Forecasting</div>',
        unsafe_allow_html=True,
    )

    render_context_help(
        "🔮 Forecast",
        "context_help_forecast",
    )

    df = st.session_state.data_out

    setting = (
        st.session_state.forecast_setting
    )

    period_text = forecast_period_text()

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "### Periode Forecast"
        )

        st.info(
            f"**{period_text}**"
        )

    with c2:

        st.markdown(
            "### Histori Digunakan"
        )

        effective_history = get_effective_history_months(
            setting,
            get_available_history(df),
        )

        if get_history_mode(setting):

            history_label = (
                f"Semua histori tersedia ({effective_history} bulan)"
            )

        else:

            history_label = (
                f"{effective_history} bulan"
            )

        st.info(
            f"**{history_label}**"
        )

        st.caption(
            "Histori diambil dari periode yang tersedia sebelum "
            "bulan forecast. Mode semua histori menggunakan seluruh "
            "bulan yang tersedia; mode manual mengikuti setting."
        )

    st.divider()

    if df.empty:

        st.warning(
            "Belum ada Data OUT."
        )

    else:

        available_history = (
            get_available_history(df)
        )

        if available_history:

            effective_history = get_effective_history_months(
                setting,
                available_history,
            )

            selected_history_months = (
                available_history
                if get_history_mode(setting)
                else available_history[-effective_history:]
            )

            history_text = ", ".join(
                [
                    f"{BULAN[x.month]} {x.year}"
                    for x in selected_history_months
                ]
            )

            effective_history = get_effective_history_months(
                setting,
                available_history,
            )

            if get_history_mode(setting):

                st.info(
                    "Histori yang akan digunakan "
                    "(semua histori tersedia): "
                    f"**{history_text}**"
                )

            elif len(available_history) >= effective_history:

                st.info(
                    "Histori yang akan digunakan: "
                    f"**{history_text}**"
                )

                # Catatan: forecast untuk bulan berikutnya akan
                # otomatis memakai histori sampai bulan terakhir
                # sebelum forecast. Jadi, misalnya forecast November,
                # data Oktober ikut masuk selama Oktober tersedia.

            else:

                st.error(
                    "Histori yang tersedia hanya: "
                    f"**{history_text}**. "
                    f"Setting membutuhkan "
                    f"**{effective_history} bulan**."
                )

        else:

            st.error(
                "Histori sebelum periode forecast "
                "belum ditemukan."
            )

        # -------------------------------------------------
        # XGBOOST INFORMATION
        # -------------------------------------------------

        history_count = len(
            available_history
        )

        if not XGBOOST_AVAILABLE:

            st.warning(
                "⚠️ **XGBoost belum tersedia.** "
                "Package XGBoost belum ter-install sehingga "
                "sistem hanya menggunakan metode forecasting "
                "yang tersedia seperti MA/WMA."
            )

        elif history_count < XGBOOST_MIN_HISTORY:

            st.warning(
                "⚠️ **XGBoost tidak digunakan untuk histori ini.** "
                f"Histori tersedia {history_count} bulan, "
                f"sedangkan XGBoost membutuhkan minimal "
                f"{XGBOOST_MIN_HISTORY} bulan histori agar "
                "training dan backtesting lebih layak. "
                "Sistem tetap menggunakan metode statistik "
                "yang sesuai dengan jumlah histori."
            )

        else:

            st.success(
                "🤖 **XGBoost tersedia.** "
                f"Histori tersedia {history_count} bulan "
                f"dan sudah memenuhi minimum "
                f"{XGBOOST_MIN_HISTORY} bulan. "
                "XGBoost dihitung dan dievaluasi secara independen "
                "bersama MA/WMA melalui backtesting."
            )

        # -------------------------------------------------
        # RECURSIVE FORECASTING INFORMATION
        # -------------------------------------------------
        try:
            recursive_status = get_recursive_forecasting_status()
        except Exception:
            recursive_status = {
                "enabled": True,
                "mode": "recursive",
                "actual_override": True,
                "wape_uses_recursive_forecast": False,
            }

        if recursive_status.get("enabled", False):
            st.success(
                "🔁 **Recursive forecasting aktif.** "
                "Jika target lebih dari satu bulan setelah actual terakhir, "
                "forecast dihitung bertahap sampai bulan target. "
                "Actual intermediate selalu meng-override forecast. "
                "Forecast recursive tidak dimasukkan ke perhitungan WAPE."
            )
        else:
            st.warning(
                "Recursive forecasting tidak aktif pada module forecasting."
            )

        st.caption(
            "Contoh: actual terakhir Agustus dan target Oktober → "
            "sistem menghitung September terlebih dahulu, lalu Oktober. "
            "Jika September sudah memiliki actual, actual September digunakan "
            "sebagai input untuk Oktober."
        )

        # -------------------------------------------------
        # VALIDATION STATUS
        # -------------------------------------------------

        if not st.session_state.data_valid:

            st.warning(
                "⚠️ Data belum divalidasi. "
                "Buka menu **Validasi** dan pastikan "
                "tidak ada ERROR sebelum menjalankan forecast."
            )

        # -------------------------------------------------
        # RUN FORECAST
        # -------------------------------------------------

        if st.button(
            "🚀 Jalankan Forecast",
            type="primary",
            use_container_width=True,
            key="run_forecast_button",
        ):

            if not st.session_state.data_valid:

                st.error(
                    "Forecast tidak dapat dijalankan. "
                    "Silakan selesaikan Validasi Data terlebih dahulu."
                )

            elif not available_history:

                st.error(
                    "Tidak ada histori yang tersedia "
                    "sebelum periode forecast."
                )

            elif (
                not get_history_mode(setting)
                and len(available_history) < get_effective_history_months(
                    setting,
                    available_history,
                )
            ):

                effective_history = get_effective_history_months(
                    setting,
                    available_history,
                )

                st.error(
                    f"Histori tersedia hanya "
                    f"{len(available_history)} bulan, "
                    f"sedangkan setting membutuhkan "
                    f"{effective_history} bulan."
                )

            else:

                try:

                    with st.spinner(
                        "Sedang menghitung forecasting..."
                    ):

                        (
                            df_bbb,
                            df_bbt,
                            summary,
                        ) = run_forecasting(
                            df=df,
                            forecast_period=period_text,
                            history_months=get_forecast_history_parameter(
                                setting
                            ),
                        )

                    st.session_state.forecast_bbb = (
                        df_bbb
                    )

                    st.session_state.forecast_bbt = (
                        df_bbt
                    )

                    st.session_state.forecast_summary = (
                        summary
                    )

                    st.session_state.forecast_loaded = (
                        True
                    )

                    st.session_state.loaded_history_id = (
                        None
                    )

                    st.session_state.loaded_forecast_info = (
                        None
                    )

                    st.session_state.last_forecast_period = (
                        period_text
                    )

                    st.session_state.last_forecast_history_months = (
                        get_effective_history_months(
                            setting,
                            available_history,
                        )
                    )

                    st.success(
                        "Forecast berhasil dihitung."
                    )

                    # -------------------------------------------------
                    # RECURSIVE STEP PREVIEW
                    # -------------------------------------------------
                    # Preview bersifat informatif dan tidak mengubah hasil utama.
                    try:
                        preview_item = None
                        preview_column = None

                        if not df_bbb.empty and "Nama Barang" in df_bbb.columns:
                            preview_item = str(df_bbb.iloc[0]["Nama Barang"])
                            preview_column = "OUT BBB"
                        elif not df_bbt.empty and "Nama Barang" in df_bbt.columns:
                            preview_item = str(df_bbt.iloc[0]["Nama Barang"])
                            preview_column = "OUT BBT"

                        if preview_item and preview_column:
                            recursive_detail = explain_recursive_forecast(
                                df=df,
                                item_name=preview_item,
                                value_column=preview_column,
                                forecast_period=period_text,
                                history_months=get_forecast_history_parameter(setting),
                            )

                            detail_df = recursive_detail.get(
                                "details",
                                pd.DataFrame(),
                            )

                            if not detail_df.empty:
                                st.markdown("#### 🔁 Contoh Langkah Recursive")
                                st.caption(
                                    f"Contoh item: **{preview_item}** ({preview_column}). "
                                    "Tabel menunjukkan langkah bulan demi bulan sampai target. "
                                    "Setiap metode diproses secara independen."
                                )
                                st.dataframe(
                                    detail_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.caption(
                                    "Sumber **Actual** berarti nilai aktual tersedia dan dipakai langsung. "
                                    "Sumber **Forecast Recursive** berarti nilai dihitung terlebih dahulu "
                                    "dan dipakai sebagai input untuk langkah berikutnya. Nilai recursive "
                                    "ini tidak digunakan sebagai actual untuk menghitung WAPE."
                                )
                    except Exception:
                        # Preview tidak boleh menggagalkan forecast utama.
                        pass

                except Exception as e:

                    st.error(
                        f"Forecast gagal dijalankan: {e}"
                    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    df_bbb = (
        st.session_state.forecast_bbb
    )

    df_bbt = (
        st.session_state.forecast_bbt
    )

    if (
        not df_bbb.empty
        or not df_bbt.empty
    ):

        st.divider()

        st.markdown(
            "### Hasil Forecast"
        )

        # -------------------------------------------------
        # CHECK CURRENT SETTING
        # -------------------------------------------------

        if not forecast_matches_current_setting():

            st.warning(
                "Hasil forecast yang tampil bukan hasil "
                "dari setting periode saat ini. "
                "Jalankan forecast kembali."
            )

        tab1, tab2 = st.tabs(
            [
                "📦 BBB",
                "📦 BBT",
            ]
        )

        with tab1:
            if df_bbb.empty:
                st.info(
                    "Tidak ada hasil forecast BBB."
                )
            else:
                display_df = df_bbb.copy()

                for col in [
                    "Forecast MA",
                    "Forecast WMA",
                    "Forecast XGBoost",
                ]:
                    if col in display_df.columns:
                        display_df[col] = (
                            pd.to_numeric(
                                display_df[col],
                                errors="coerce",
                            )
                            .apply(format_number)
                        )

                for col in [
                    "WAPE MA",
                    "WAPE WMA",
                    "WAPE XGBoost",
                ]:
                    if col in display_df.columns:
                        display_df[col] = (
                            pd.to_numeric(
                                display_df[col],
                                errors="coerce",
                            )
                            .apply(format_percent)
                        )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

        with tab2:
            if df_bbt.empty:
                st.info(
                    "Tidak ada hasil forecast BBT."
                )
            else:
                display_df = df_bbt.copy()

                for col in [
                    "Forecast MA",
                    "Forecast WMA",
                    "Forecast XGBoost",
                ]:
                    if col in display_df.columns:
                        display_df[col] = (
                            pd.to_numeric(
                                display_df[col],
                                errors="coerce",
                            )
                            .apply(format_number)
                        )

                for col in [
                    "WAPE MA",
                    "WAPE WMA",
                    "WAPE XGBoost",
                ]:
                    if col in display_df.columns:
                        display_df[col] = (
                            pd.to_numeric(
                                display_df[col],
                                errors="coerce",
                            )
                            .apply(format_percent)
                        )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

        # -------------------------------------------------
        # METHOD PERFORMANCE SUMMARY
        # -------------------------------------------------

        summary = (
            st.session_state.forecast_summary
        )

        summary_bbb = summary.get(
            "bbb",
            {},
        )

        summary_bbt = summary.get(
            "bbt",
            {},
        )

        st.divider()

        st.markdown(
            "### Performance per Metode"
        )

        st.caption(
            "Setiap metode memiliki Forecast dan WAPE sendiri. "
            "Tidak ada pemilihan metode otomatis."
        )

        performance_df = pd.DataFrame(
            [
                {
                    "Stream": "BBB",
                    "Metode": "MA",
                    "WAPE": summary_bbb.get("wape_ma"),
                    "Forecast Accuracy": summary_bbb.get("accuracy_ma"),
                },
                {
                    "Stream": "BBB",
                    "Metode": "WMA",
                    "WAPE": summary_bbb.get("wape_wma"),
                    "Forecast Accuracy": summary_bbb.get("accuracy_wma"),
                },
                {
                    "Stream": "BBB",
                    "Metode": "XGBoost",
                    "WAPE": summary_bbb.get("wape_xgboost"),
                    "Forecast Accuracy": summary_bbb.get("accuracy_xgboost"),
                },
                {
                    "Stream": "BBT",
                    "Metode": "MA",
                    "WAPE": summary_bbt.get("wape_ma"),
                    "Forecast Accuracy": summary_bbt.get("accuracy_ma"),
                },
                {
                    "Stream": "BBT",
                    "Metode": "WMA",
                    "WAPE": summary_bbt.get("wape_wma"),
                    "Forecast Accuracy": summary_bbt.get("accuracy_wma"),
                },
                {
                    "Stream": "BBT",
                    "Metode": "XGBoost",
                    "WAPE": summary_bbt.get("wape_xgboost"),
                    "Forecast Accuracy": summary_bbt.get("accuracy_xgboost"),
                },
            ]
        )

        performance_df["WAPE"] = performance_df["WAPE"].apply(
            format_percent
        )
        performance_df["Forecast Accuracy"] = performance_df[
            "Forecast Accuracy"
        ].apply(format_percent)

        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # SAVE FORECAST
        # -------------------------------------------------

        st.divider()

        st.markdown(
            "### Simpan Forecast"
        )

        c1, c2 = st.columns(2)

        with c1:

            nama_user = st.text_input(
                "Nama User",
                placeholder="Masukkan nama user",
                key="nama_user_forecast",
            )

        with c2:

            st.text_input(
                "Periode Forecast",
                value=period_text,
                disabled=True,
                key="periode_forecast_display",
            )

        if st.button(
            "💾 Simpan Forecast",
            type="primary",
            key="save_forecast_button",
        ):

            if not nama_user.strip():

                st.warning(
                    "Nama User wajib diisi."
                )

            elif not forecast_matches_current_setting():

                st.error(
                    "Hasil forecast tidak sesuai dengan "
                    "setting periode saat ini. "
                    "Silakan jalankan forecast kembali."
                )

            else:

                try:

                    history_id = save_history(
                        nama_user=nama_user.strip(),
                        periode_forecast=period_text,
                        # Database menyimpan jumlah histori aktual.
                        history_months=get_effective_history_months(
                            setting,
                            get_available_history(df),
                        ),
                        forecast_bbb=df_bbb,
                        forecast_bbt=df_bbt,
                        summary=(
                            st.session_state
                            .forecast_summary
                        ),
                    )

                    st.session_state.loaded_history_id = (
                        history_id
                    )

                    st.session_state.forecast_loaded = (
                        True
                    )

                    st.session_state.loaded_forecast_info = {
                        "id": history_id,
                        "nama_user": nama_user.strip(),
                        "periode_forecast": period_text,
                        "history_months": get_effective_history_months(
                            setting,
                            get_available_history(df),
                        ),
                        "created_at": "Baru saja",
                    }

                    st.session_state.last_forecast_period = (
                        period_text
                    )

                    st.session_state.last_forecast_history_months = (
                        get_effective_history_months(
                            setting,
                            get_available_history(df),
                        )
                    )

                    st.success(
                        f"Forecast {period_text} "
                        "berhasil disimpan."
                    )

                except Exception as e:

                    st.error(
                        f"Gagal menyimpan forecast: {e}"
                    )


# =========================================================
# HISTORY
# =========================================================

elif menu == "🕘 History":

    st.markdown(
        '<div class="section-title">🕘 History Forecast</div>',
        unsafe_allow_html=True,
    )

    render_context_help(
        "🕘 History",
        "context_help_history",
    )

    histories = load_history()

    if not histories:

        st.info(
            "Belum ada history forecasting."
        )

    else:

        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------

        history_data = []

        for row in histories:

            history_data.append(
                {
                    "ID": row["id"],
                    "Nama User": row["nama_user"],
                    "Periode Forecast": row[
                        "periode_forecast"
                    ],
                    "Histori": row[
                        "history_months"
                    ],
                    "Tanggal Simpan": row[
                        "created_at"
                    ],
                    "Status": row[
                        "status"
                    ],
                }
            )

        history_df = pd.DataFrame(
            history_data
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # -------------------------------------------------
        # SELECT
        # -------------------------------------------------

        history_ids = [
            row["id"]
            for row in histories
        ]

        selected_id = st.selectbox(
            "Pilih History",
            history_ids,
            key="history_select",
        )

        selected_row = next(
            (
                row
                for row in histories
                if row["id"] == selected_id
            ),
            None,
        )

        if selected_row:

            st.info(
                f"Forecast "
                f"{selected_row['periode_forecast']} "
                f"— "
                f"{selected_row['nama_user']}"
            )

            c1, c2 = st.columns(2)

            # -------------------------------------------------
            # LOAD
            # -------------------------------------------------

            with c1:

                if st.button(
                    "📂 Load History",
                    use_container_width=True,
                    key="history_load_button",
                ):

                    success, message = (
                        load_forecast_to_session(
                            selected_id
                        )
                    )

                    if success:

                        st.query_params["menu"] = (
                            "dashboard"
                        )

                        st.success(
                            message
                        )

                        st.rerun()

                    else:

                        st.error(
                            message
                        )

            # -------------------------------------------------
            # DELETE
            # -------------------------------------------------

            with c2:

                if st.button(
                    "🗑️ Hapus History",
                    use_container_width=True,
                    key="history_delete_button",
                ):

                    st.session_state[
                        "confirm_delete_history"
                    ] = selected_id

                    st.rerun()

            # -------------------------------------------------
            # DELETE CONFIRMATION
            # -------------------------------------------------

            if (
                st.session_state.get(
                    "confirm_delete_history"
                )
                == selected_id
            ):

                st.warning(
                    "Apakah Anda yakin ingin "
                    "menghapus data forecasting ini?"
                )

                c1, c2 = st.columns(2)

                with c1:

                    if st.button(
                        "Batal",
                        use_container_width=True,
                        key="cancel_delete_history",
                    ):

                        st.session_state[
                            "confirm_delete_history"
                        ] = None

                        st.rerun()

                with c2:

                    if st.button(
                        "Hapus",
                        type="primary",
                        use_container_width=True,
                        key="confirm_delete_history_button",
                    ):

                        delete_history(
                            selected_id
                        )

                        st.session_state[
                            "confirm_delete_history"
                        ] = None

                        if (
                            st.session_state
                            .loaded_history_id
                            == selected_id
                        ):

                            reset_forecast_session()

                        st.success(
                            "History berhasil dihapus."
                        )

                        st.rerun()

        st.divider()

        # -------------------------------------------------
        # DELETE ALL
        # -------------------------------------------------

        if st.button(
            "🗑️ Hapus Semua History",
            key="delete_all_history_button",
        ):

            st.session_state[
                "confirm_delete_all"
            ] = True

            st.rerun()

        if st.session_state.get(
            "confirm_delete_all",
            False,
        ):

            st.warning(
                "Apakah Anda yakin ingin "
                "menghapus SEMUA history forecasting?"
            )

            c1, c2 = st.columns(2)

            with c1:

                if st.button(
                    "Batal",
                    key="cancel_delete_all",
                    use_container_width=True,
                ):

                    st.session_state[
                        "confirm_delete_all"
                    ] = False

                    st.rerun()

            with c2:

                if st.button(
                    "Hapus Semua",
                    key="confirm_delete_all_button",
                    type="primary",
                    use_container_width=True,
                ):

                    delete_all_history()

                    st.session_state[
                        "confirm_delete_all"
                    ] = False

                    reset_forecast_session()

                    st.success(
                        "Semua history berhasil dihapus."
                    )

                    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        © Demand Planner Main Warehouse Batu Ceper
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DOKUMENTASI IMPLEMENTASI BANTUAN
# =========================================================
#
# LEVEL 1 — BANTUAN LENGKAP SIDEBAR
#
# Tombol "📖 Bantuan Lengkap" membuka HELP_CONTENT["__ALL__"].
# Halaman gabungan dibuat dari seluruh halaman HELP_CONTENT yang asli,
# sehingga perubahan isi bantuan per-menu otomatis ikut tampil di sini.
#
# LEVEL 2 — BANTUAN KHUSUS MENU
#
# Setiap menu memiliki tombol bantuan kontekstual yang memanggil
# render_context_help(). Tombol tersebut tidak membuka seluruh panduan,
# melainkan langsung membuka halaman pertama dokumentasi menu aktif.
#
# HEADER
#
# Tombol ❓ di header juga bersifat kontekstual. Jika pengguna sedang
# berada di menu Forecast, tombol tersebut membuka bantuan Forecast.
#
# SIDEBAR VS HEADER
#
# Sidebar cocok untuk pengguna yang ingin belajar aplikasi dari awal.
# Header dan tombol di dalam menu cocok untuk pengguna yang sedang berada
# pada satu menu dan hanya membutuhkan penjelasan bagian tersebut.
#
# CONTOH ALUR PENGGUNAAN
#
# 1. Buka Data OUT dan upload file histori.
# 2. Gunakan tombol ❓ Bantuan Data OUT jika format kolom belum jelas.
# 3. Buka Setting untuk memilih bulan/tahun target dan periode histori.
# 4. Buka Validasi dan pastikan tidak ada ERROR yang menghalangi forecast.
# 5. Buka Forecast dan jalankan perhitungan MA, WMA, dan XGBoost secara terpisah.
# 6. Buka Dashboard untuk melihat WAPE dan Accuracy setiap metode.
# 7. Simpan hasil ke History jika diperlukan.
# 8. Gunakan Bantuan Lengkap dari sidebar jika ingin membaca seluruh
#    dokumentasi dan contoh yang tersedia.
#
# CONTOH RECURSIVE FORECASTING
#
# Jika actual tersedia sampai Agustus dan target forecast adalah Oktober,
# setiap metode melakukan forecast September terlebih dahulu, lalu memakai
# September sebagai histori sementara untuk menghitung Oktober.
# Jika actual September tersedia, actual September yang dipakai.
# Forecast recursive bukan actual dan tidak dimasukkan sebagai actual pada
# perhitungan WAPE/backtesting. Tidak ada pemilihan metode otomatis.
#
# CATATAN PEMELIHARAAN
#
# Jangan menghapus HELP_CONTENT hanya untuk memperpendek file. Struktur
# bantuan lama sengaja dipertahankan karena menjadi sumber dokumentasi
# masing-masing menu. HELP_ALL_PAGES adalah lapisan tambahan di atasnya.
#
# =========================================================
# END DOCUMENTATION
# =========================================================
