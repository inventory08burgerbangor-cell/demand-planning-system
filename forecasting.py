# forecasting.py

import re
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================================================
# OPTIONAL XGBOOST
# =========================================================

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True

except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================

# Minimum histori untuk XGBoost
XGBOOST_MIN_HISTORY = 3

# Minimum jumlah training row XGBoost
XGBOOST_MIN_TRAIN_SIZE = 1

RANDOM_STATE = 42


RESULT_COLUMNS = [
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


# =========================================================
# MONTH MAPPING
# =========================================================

MONTH_MAP = {
    "januari": 1,
    "jan": 1,
    "january": 1,

    "februari": 2,
    "feb": 2,
    "february": 2,

    "maret": 3,
    "mar": 3,
    "march": 3,

    "april": 4,
    "apr": 4,

    "mei": 5,
    "may": 5,

    "juni": 6,
    "jun": 6,
    "june": 6,

    "juli": 7,
    "jul": 7,
    "july": 7,

    "agustus": 8,
    "agu": 8,
    "ags": 8,
    "aug": 8,
    "august": 8,

    "september": 9,
    "sep": 9,
    "sept": 9,

    "oktober": 10,
    "okt": 10,
    "oct": 10,
    "october": 10,

    "november": 11,
    "nov": 11,

    "desember": 12,
    "des": 12,
    "dec": 12,
    "december": 12,
}


MONTH_NAME_ID = {
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


# =========================================================
# BASIC HELPERS
# =========================================================

def _clean_text(value) -> str:
    """
    Membersihkan text agar aman untuk parsing.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def _safe_float(value, default=0.0) -> float:
    """
    Konversi nilai menjadi float dengan aman.
    """

    try:
        if pd.isna(value):
            return float(default)

        result = float(value)

        if not np.isfinite(result):
            return float(default)

        return result

    except Exception:
        return float(default)


def _normalize_history_months(history_months):
    """
    Normalisasi setting histori.

    Aturan:

        None / kosong / <= 0
            -> None
            -> gunakan SEMUA histori tersedia

        > 0
            -> gunakan maksimal sejumlah bulan tersebut

    Contoh:

        history_months = 3
            -> maksimal 3 bulan

        history_months = 8
            -> maksimal 8 bulan

        history_months = None
            -> semua histori tersedia
    """

    if history_months is None:
        return None

    try:
        if isinstance(history_months, str):
            text = history_months.strip()

            if text == "":
                return None

            # Support setting seperti:
            # "all", "semua", "all history"
            if text.lower() in {
                "all",
                "semua",
                "semua histori",
                "all history",
            }:
                return None

        value = int(float(history_months))

        if value <= 0:
            return None

        return value

    except Exception:
        return None


# =========================================================
# PERIOD PARSER
# =========================================================

def parse_period(period, default_year=None):
    """
    Mengubah berbagai format periode menjadi pandas Timestamp
    dengan tanggal selalu tanggal 1.

    Format yang didukung:

        Juni 2026
        September 2026
        2026-06
        2026/06
        06/2026
        2026-06-01
        Timestamp
        datetime
        Juni + default_year
        202606
    """

    if period is None:
        return None

    # -----------------------------------------------------
    # Timestamp / datetime
    # -----------------------------------------------------

    if isinstance(period, pd.Timestamp):

        return pd.Timestamp(
            year=period.year,
            month=period.month,
            day=1,
        )

    if hasattr(period, "year") and hasattr(period, "month"):

        try:
            return pd.Timestamp(
                year=int(period.year),
                month=int(period.month),
                day=1,
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Numeric
    # -----------------------------------------------------

    if isinstance(
        period,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):

        try:
            numeric_value = float(period)

            # Format YYYYMM
            if numeric_value.is_integer():
                numeric_text = str(
                    int(numeric_value)
                )

                if re.fullmatch(
                    r"\d{6}",
                    numeric_text,
                ):

                    year = int(
                        numeric_text[:4]
                    )

                    month = int(
                        numeric_text[4:]
                    )

                    if (
                        1900 <= year <= 2100
                        and 1 <= month <= 12
                    ):

                        return pd.Timestamp(
                            year=year,
                            month=month,
                            day=1,
                        )

            # Coba sebagai tanggal pandas
            parsed = pd.to_datetime(
                period,
                errors="coerce",
            )

            if pd.notna(parsed):

                return pd.Timestamp(
                    year=parsed.year,
                    month=parsed.month,
                    day=1,
                )

        except Exception:
            pass

    text = _clean_text(period)

    if not text:
        return None

    normalized = (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("_", " ")
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    tokens = normalized.split()

    found_month = None
    found_year = None

    # -----------------------------------------------------
    # Month name + year
    # -----------------------------------------------------

    for token in tokens:

        if token in MONTH_MAP:
            found_month = MONTH_MAP[token]

        if (
            token.isdigit()
            and len(token) == 4
        ):
            found_year = int(token)

    if found_month is not None:

        if found_year is None:

            if default_year is None:
                return None

            found_year = int(default_year)

        try:
            return pd.Timestamp(
                year=found_year,
                month=found_month,
                day=1,
            )

        except Exception:
            return None

    # -----------------------------------------------------
    # YYYYMM
    # -----------------------------------------------------

    if re.fullmatch(
        r"\d{6}",
        normalized,
    ):

        year = int(
            normalized[:4]
        )

        month = int(
            normalized[4:]
        )

        if (
            1900 <= year <= 2100
            and 1 <= month <= 12
        ):

            try:
                return pd.Timestamp(
                    year=year,
                    month=month,
                    day=1,
                )

            except Exception:
                return None

    # -----------------------------------------------------
    # YYYY MM / MM YYYY
    # -----------------------------------------------------

    if len(tokens) == 2:

        try:

            first = int(tokens[0])
            second = int(tokens[1])

            if (
                1900 <= first <= 2100
                and 1 <= second <= 12
            ):

                return pd.Timestamp(
                    year=first,
                    month=second,
                    day=1,
                )

            if (
                1 <= first <= 12
                and 1900 <= second <= 2100
            ):

                return pd.Timestamp(
                    year=second,
                    month=first,
                    day=1,
                )

        except Exception:
            pass

    # -----------------------------------------------------
    # General pandas parsing
    # -----------------------------------------------------

    try:

        parsed = pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=True,
        )

        if pd.notna(parsed):

            return pd.Timestamp(
                year=parsed.year,
                month=parsed.month,
                day=1,
            )

    except Exception:
        pass

    # -----------------------------------------------------
    # Month number + default year
    # -----------------------------------------------------

    if default_year is not None:

        try:

            month_number = int(text)

            if 1 <= month_number <= 12:

                return pd.Timestamp(
                    year=int(default_year),
                    month=month_number,
                    day=1,
                )

        except Exception:
            pass

    return None


# =========================================================
# WAPE
# =========================================================

def calculate_wape(actual, forecast) -> float:
    """
    WAPE:

        SUM |Actual - Forecast|
        ----------------------- x 100%
             SUM |Actual|

    Jika tidak ada data valid:
        NaN

    Jika total actual = 0:
        error = 0 -> 0%
        error > 0 -> 100%
    """

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    forecast_array = np.asarray(
        forecast,
        dtype=float,
    )

    if (
        actual_array.size == 0
        or forecast_array.size == 0
    ):
        return np.nan

    if actual_array.shape != forecast_array.shape:

        try:
            actual_array, forecast_array = (
                np.broadcast_arrays(
                    actual_array,
                    forecast_array,
                )
            )

        except Exception:
            return np.nan

    mask = (
        np.isfinite(actual_array)
        & np.isfinite(forecast_array)
    )

    actual_array = actual_array[mask]
    forecast_array = forecast_array[mask]

    if len(actual_array) == 0:
        return np.nan

    total_actual = np.sum(
        np.abs(actual_array)
    )

    total_error = np.sum(
        np.abs(
            actual_array
            - forecast_array
        )
    )

    if total_actual == 0:

        if total_error == 0:
            return 0.0

        return 100.0

    return float(
        total_error
        / total_actual
        * 100.0
    )


def calculate_accuracy(wape: float) -> float:
    """
    Accuracy = 100% - WAPE.
    """

    try:

        if not np.isfinite(
            float(wape)
        ):
            return np.nan

        return max(
            0.0,
            100.0 - float(wape),
        )

    except Exception:
        return np.nan


# =========================================================
# MOVING AVERAGE
# =========================================================

def moving_average(
    values,
    window: int,
) -> float:
    """
    Simple Moving Average.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return np.nan

    if window <= 0:
        return np.nan

    if len(values) < window:

        return float(
            np.mean(values)
        )

    return float(
        np.mean(
            values[-window:]
        )
    )


# =========================================================
# WEIGHTED MOVING AVERAGE
# =========================================================

def weighted_moving_average(
    values,
    window: int,
) -> float:
    """
    Weighted Moving Average.

    Data terbaru mendapatkan bobot terbesar.

    Window 3:

        oldest = 1
        middle = 2
        latest = 3
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return np.nan

    if window <= 0:
        return np.nan

    if len(values) < window:
        window = len(values)

    if window <= 0:
        return np.nan

    recent = values[-window:]

    weights = np.arange(
        1,
        window + 1,
        dtype=float,
    )

    return float(
        np.sum(
            recent * weights
        )
        / np.sum(weights)
    )


# =========================================================
# XGBOOST FEATURE ENGINEERING
# =========================================================

def create_xgboost_training_data(values):
    """
    Membuat feature training XGBoost.

    Feature:

        lag_1
        lag_2
        rolling_3
        trend

    Target:

        nilai bulan berikutnya.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if len(values) < 3:
        return None, None

    X = []
    y = []

    for i in range(
        2,
        len(values),
    ):

        lag_1 = values[i - 1]

        lag_2 = values[i - 2]

        start_index = max(
            0,
            i - 3,
        )

        rolling_3 = np.mean(
            values[start_index:i]
        )

        trend = (
            values[i - 1]
            - values[i - 2]
        )

        X.append(
            [
                lag_1,
                lag_2,
                rolling_3,
                trend,
            ]
        )

        y.append(
            values[i]
        )

    if len(X) == 0:
        return None, None

    return (
        np.asarray(
            X,
            dtype=float,
        ),
        np.asarray(
            y,
            dtype=float,
        ),
    )


def create_future_xgboost_features(values):
    """
    Membuat feature untuk prediksi
    periode berikutnya.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if len(values) < 2:
        return None

    lag_1 = values[-1]

    lag_2 = values[-2]

    rolling_3 = np.mean(
        values[-3:]
    )

    trend = (
        values[-1]
        - values[-2]
    )

    return np.asarray(
        [
            [
                lag_1,
                lag_2,
                rolling_3,
                trend,
            ]
        ],
        dtype=float,
    )


# =========================================================
# XGBOOST MODEL
# =========================================================

def build_xgboost_model():
    """
    Membuat model XGBoost dengan parameter
    konservatif untuk histori pendek.
    """

    if not XGBOOST_AVAILABLE:
        return None

    return XGBRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


# =========================================================
# XGBOOST FORECAST
# =========================================================

def xgboost_forecast(values) -> float:
    """
    Forecast 1 periode ke depan
    menggunakan XGBoost.
    """

    if not XGBOOST_AVAILABLE:
        return np.nan

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if len(values) < XGBOOST_MIN_HISTORY:
        return np.nan

    X, y = create_xgboost_training_data(
        values
    )

    if X is None or y is None:
        return np.nan

    if len(X) < XGBOOST_MIN_TRAIN_SIZE:
        return np.nan

    future_features = (
        create_future_xgboost_features(
            values
        )
    )

    if future_features is None:
        return np.nan

    try:

        model = build_xgboost_model()

        if model is None:
            return np.nan

        model.fit(
            X,
            y,
        )

        prediction = model.predict(
            future_features
        )[0]

        prediction = max(
            0.0,
            float(prediction),
        )

        if not np.isfinite(
            prediction
        ):
            return np.nan

        return prediction

    except Exception:
        return np.nan


# =========================================================
# XGBOOST BACKTEST
# =========================================================

def backtest_xgboost_details(values):
    """
    Walk-forward backtesting XGBoost
    tanpa data leakage.

    Actual bulan test tidak dimasukkan
    ke training.
    """

    if not XGBOOST_AVAILABLE:

        return {
            "actual": [],
            "forecast": [],
            "wape": np.nan,
        }

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    history_count = len(values)

    if history_count < XGBOOST_MIN_HISTORY:

        return {
            "actual": [],
            "forecast": [],
            "wape": np.nan,
        }

    actual_list = []
    forecast_list = []

    # -----------------------------------------------------
    # Walk-forward
    # -----------------------------------------------------

    for i in range(
        2,
        history_count,
    ):

        train_values = values[:i]

        actual_value = values[i]

        if len(train_values) < 2:
            continue

        try:

            # -------------------------------------------------
            # Histori training hanya 2 bulan
            # -------------------------------------------------

            if len(train_values) == 2:

                X_train = np.asarray(
                    [
                        [
                            train_values[0],
                            0.0,
                            train_values[0],
                            0.0,
                        ]
                    ],
                    dtype=float,
                )

                y_train = np.asarray(
                    [
                        train_values[1]
                    ],
                    dtype=float,
                )

            else:

                X_train, y_train = (
                    create_xgboost_training_data(
                        train_values
                    )
                )

            if (
                X_train is None
                or y_train is None
            ):
                continue

            if len(X_train) < (
                XGBOOST_MIN_TRAIN_SIZE
            ):
                continue

            model = build_xgboost_model()

            if model is None:
                continue

            model.fit(
                X_train,
                y_train,
            )

            future_features = (
                create_future_xgboost_features(
                    train_values
                )
            )

            if future_features is None:
                continue

            prediction = model.predict(
                future_features
            )[0]

            prediction = max(
                0.0,
                float(prediction),
            )

            if not np.isfinite(
                prediction
            ):
                continue

            actual_list.append(
                float(actual_value)
            )

            forecast_list.append(
                prediction
            )

        except Exception:
            continue

    if not actual_list:

        return {
            "actual": [],
            "forecast": [],
            "wape": np.nan,
        }

    wape = calculate_wape(
        actual_list,
        forecast_list,
    )

    return {
        "actual": actual_list,
        "forecast": forecast_list,
        "wape": wape,
    }


def backtest_xgboost(values) -> float:
    """
    Versi sederhana yang hanya
    mengembalikan WAPE.
    """

    result = backtest_xgboost_details(
        values
    )

    return result["wape"]


# =========================================================
# GENERIC METHOD FORECAST
# =========================================================

def _default_moving_window(values, preferred=3):
    """Menentukan window default untuk MA/WMA.

    Sistem sekarang hanya menampilkan tiga metode utama:
    MA, WMA, dan XGBoost. MA/WMA menggunakan window 3
    jika histori minimal 3 bulan tersedia. Jika histori
    lebih pendek, window mengikuti jumlah histori yang ada.
    """

    try:
        count = len(values)
    except Exception:
        count = 0

    if count <= 0:
        return 0

    return max(1, min(int(preferred), int(count)))


def _normalize_method_name(method: str) -> str:
    """Normalisasi nama metode menjadi MA, WMA, atau XGBoost."""

    method_clean = (
        str(method)
        .strip()
        .upper()
        .replace(" ", "")
    )

    if method_clean in {"XGBOOST", "XGB"}:
        return "XGBoost"

    if method_clean.startswith("WMA"):
        return "WMA"

    if method_clean.startswith("MA"):
        return "MA"

    return str(method).strip()


def _method_window(method: str, values) -> int:
    """Mengambil window MA/WMA; default MA/WMA adalah 3."""

    method_clean = str(method).strip().upper().replace(" ", "")

    for prefix in ("WMA", "MA"):
        if method_clean.startswith(prefix):
            suffix = method_clean.replace(prefix, "", 1)
            if suffix.isdigit():
                try:
                    return max(1, int(suffix))
                except Exception:
                    pass
            return _default_moving_window(values, preferred=3)

    return _default_moving_window(values, preferred=3)


def forecast_with_method(
    values,
    method: str,
) -> float:
    """
    Forecast satu periode menggunakan satu dari tiga metode utama:

        MA
        WMA
        XGBoost

    MA dan WMA menggunakan window default 3 bulan. Alias lama
    seperti MA3/WMA3 tetap didukung agar kompatibel dengan kode
    lain yang mungkin masih memanggil fungsi ini.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    method_clean = (
        str(method)
        .strip()
        .upper()
        .replace(" ", "")
    )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    if method_clean in {
        "XGBOOST",
        "XGB",
    }:
        return xgboost_forecast(values)

    # -----------------------------------------------------
    # WMA
    # -----------------------------------------------------

    if method_clean.startswith("WMA"):
        window = _method_window(method_clean, values)

        if window <= 0:
            return np.nan

        return weighted_moving_average(
            values,
            window,
        )

    # -----------------------------------------------------
    # MA
    # -----------------------------------------------------

    if method_clean.startswith("MA"):
        window = _method_window(method_clean, values)

        if window <= 0:
            return np.nan

        return moving_average(
            values,
            window,
        )

    return np.nan


# =========================================================
# GENERIC BACKTEST DETAILS
# =========================================================

def backtest_method_details(
    values,
    method: str,
):
    """
    Walk-forward backtesting untuk satu metode tertentu.

    WAPE dihitung TERPISAH untuk setiap metode.
    Forecast recursive intermediate tidak pernah dimasukkan
    ke sini; hanya pasangan actual-vs-backtest forecast historis.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    method_clean = (
        str(method)
        .strip()
        .upper()
        .replace(" ", "")
    )

    # XGBoost memiliki backtest walk-forward sendiri.
    if method_clean in {"XGBOOST", "XGB"}:
        return backtest_xgboost_details(values)

    if method_clean.startswith("WMA"):
        prefix = "WMA"
    elif method_clean.startswith("MA"):
        prefix = "MA"
    else:
        return {
            "actual": [],
            "forecast": [],
            "wape": np.nan,
        }

    window = _method_window(method_clean, values)

    if window <= 0:
        return {
            "actual": [],
            "forecast": [],
            "wape": np.nan,
        }

    actual_list = []
    forecast_list = []

    # -----------------------------------------------------
    # Walk-forward backtest
    # -----------------------------------------------------

    for i in range(
        window,
        len(values),
    ):

        train_values = values[:i]
        actual_value = values[i]

        try:
            if prefix == "WMA":
                prediction = weighted_moving_average(
                    train_values,
                    window,
                )
            else:
                prediction = moving_average(
                    train_values,
                    window,
                )

            if not np.isfinite(prediction):
                continue

            prediction = max(
                0.0,
                float(prediction),
            )

            actual_list.append(float(actual_value))
            forecast_list.append(prediction)

        except Exception:
            continue

    if not actual_list:
        return {
            "actual": [],
            "forecast": [],
            "wape": np.nan,
        }

    wape = calculate_wape(
        actual_list,
        forecast_list,
    )

    return {
        "actual": actual_list,
        "forecast": forecast_list,
        "wape": wape,
    }


def backtest_method(
    values,
    method: str,
) -> float:
    """
    Mengembalikan WAPE dari suatu method.
    """

    result = backtest_method_details(
        values,
        method,
    )

    return result["wape"]


# =========================================================
# AVAILABLE METHODS
# =========================================================

def get_available_methods(
    history_count: int,
) -> List[str]:
    """
    Mengembalikan tepat tiga metode forecast utama.

    Tidak ada lagi pemilihan ``Best Method``.
    Metode yang ditampilkan adalah:

        1. MA
        2. WMA
        3. XGBoost

    XGBoost hanya eligible jika package tersedia dan histori
    memenuhi minimum. MA/WMA memerlukan minimal 2 titik untuk
    backtest, sementara proses forecast keseluruhan tetap
    mensyaratkan minimal 3 bulan histori seperti sebelumnya.
    """

    try:
        history_count = int(history_count)
    except Exception:
        return []

    if history_count < 1:
        return []

    methods = [
        "MA",
        "WMA",
    ]

    if (
        XGBOOST_AVAILABLE
        and history_count >= XGBOOST_MIN_HISTORY
    ):
        methods.append("XGBoost")

    return methods


def evaluate_all_methods(values):
    """
    Menghitung forecast dan WAPE secara TERPISAH untuk MA, WMA,
    dan XGBoost. Tidak ada ranking dan tidak ada Best Method.

    Return:
        {
            "MA": {"forecast", "wape", "actual", "backtest_forecast"},
            "WMA": {...},
            "XGBoost": {...},
        }
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    values = np.maximum(values, 0.0)

    method_names = ["MA", "WMA", "XGBoost"]
    results = {}

    for method in method_names:
        if method == "XGBoost" and not XGBOOST_AVAILABLE:
            results[method] = {
                "forecast": np.nan,
                "wape": np.nan,
                "actual": [],
                "backtest_forecast": [],
                "available": False,
            }
            continue

        details = backtest_method_details(
            values,
            method,
        )

        forecast = forecast_with_method(
            values,
            method,
        )

        if not np.isfinite(forecast):
            forecast = np.nan
        else:
            forecast = max(0.0, float(forecast))

        results[method] = {
            "forecast": forecast,
            "wape": details["wape"],
            "actual": details["actual"],
            "backtest_forecast": details["forecast"],
            "available": True,
        }

    return results


def auto_best_method(values):
    """
    Kompatibilitas API lama.

    Fungsi ini tidak lagi memilih Best Method. Nilai ``method``
    dikembalikan sebagai None, sedangkan detail ketiga metode
    tersedia di ``methods``. Kode baru sebaiknya menggunakan
    ``evaluate_all_methods()`` secara langsung.
    """

    methods = evaluate_all_methods(values)

    return {
        "method": None,
        "wape": np.nan,
        "forecast": np.nan,
        "actual": [],
        "backtest_forecast": [],
        "methods": methods,
    }


# =========================================================
# PREPARE HISTORY
# =========================================================

def prepare_item_history(
    df,
    item_name,
    value_column,
    forecast_date,
    history_months=None,
):
    """
    Menyiapkan histori satu item
    untuk satu stream:

        OUT BBB
        atau
        OUT BBT

    Perubahan penting:

        history_months=None
            -> SEMUA histori sebelum forecast

        history_months=8
            -> maksimal 8 bulan terakhir

        history_months=3
            -> maksimal 3 bulan terakhir

    Duplicate periode akan di-aggregate.
    """

    empty_columns = [
        "_periode",
        "value",
        "Satuan",
    ]

    if df is None or df.empty:

        return pd.DataFrame(
            columns=empty_columns
        )

    required_columns = [
        "Bulan",
        "Nama Barang",
        value_column,
    ]

    for column in required_columns:

        if column not in df.columns:

            return pd.DataFrame(
                columns=empty_columns
            )

    work = df.copy()

    # -----------------------------------------------------
    # Filter item
    # -----------------------------------------------------

    work["Nama Barang"] = (
        work["Nama Barang"]
        .astype(str)
        .str.strip()
    )

    item_mask = (
        work["Nama Barang"]
        == str(item_name).strip()
    )

    work = work.loc[
        item_mask
    ].copy()

    if work.empty:

        return pd.DataFrame(
            columns=empty_columns
        )

    # -----------------------------------------------------
    # Parse periode
    # -----------------------------------------------------

    work["_periode"] = (
        work["Bulan"].apply(
            lambda x: parse_period(
                x,
                default_year=forecast_date.year,
            )
        )
    )

    work = work[
        work["_periode"].notna()
    ].copy()

    # -----------------------------------------------------
    # Hanya histori sebelum forecast
    # -----------------------------------------------------

    work = work[
        work["_periode"]
        < forecast_date
    ].copy()

    if work.empty:

        return pd.DataFrame(
            columns=empty_columns
        )

    # -----------------------------------------------------
    # Numeric
    # -----------------------------------------------------

    work["value"] = pd.to_numeric(
        work[value_column],
        errors="coerce",
    ).fillna(0.0)

    # Negative OUT dianggap 0
    work["value"] = work[
        "value"
    ].clip(
        lower=0
    )

    # -----------------------------------------------------
    # Satuan
    # -----------------------------------------------------

    if "Satuan" in work.columns:

        work["Satuan"] = (
            work["Satuan"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        work["Satuan"] = ""

    # -----------------------------------------------------
    # Aggregate per bulan
    # -----------------------------------------------------

    grouped = (
        work.groupby(
            "_periode",
            as_index=False,
        )
        .agg(
            value=("value", "sum"),
            Satuan=("Satuan", "first"),
        )
        .sort_values(
            "_periode"
        )
    )

    if grouped.empty:
        return grouped

    # -----------------------------------------------------
    # Normalisasi setting histori
    # -----------------------------------------------------

    history_months = _normalize_history_months(
        history_months
    )

    # -----------------------------------------------------
    # Ambil histori terakhir
    #
    # PENTING:
    #
    # Kalau history_months = None,
    # JANGAN dipotong.
    #
    # Jadi semua histori tersedia dipakai.
    # -----------------------------------------------------

    if history_months is not None:

        grouped = grouped.tail(
            history_months
        )

    return grouped.reset_index(
        drop=True
    )


# =========================================================
# RECURSIVE FORECAST HELPERS
# =========================================================

def _prepare_item_actual_history(
    df,
    item_name,
    value_column,
    forecast_date,
):
    """
    Mengambil seluruh histori ACTUAL satu item
    sebelum forecast_date.

    Berbeda dengan prepare_item_history():

        prepare_item_history()
            -> mengikuti history_months

        _prepare_item_actual_history()
            -> mengambil seluruh actual yang tersedia

    Fungsi ini sengaja tidak dipakai untuk memilih
    Best Method. Tujuannya hanya untuk mengetahui apakah
    bulan-bulan intermediate memiliki ACTUAL sehingga
    actual tersebut dapat meng-override forecast recursive.
    """

    empty_columns = [
        "_periode",
        "value",
        "Satuan",
    ]

    if df is None or df.empty:
        return pd.DataFrame(
            columns=empty_columns
        )

    required_columns = [
        "Bulan",
        "Nama Barang",
        value_column,
    ]

    for column in required_columns:
        if column not in df.columns:
            return pd.DataFrame(
                columns=empty_columns
            )

    work = df.copy()

    work["Nama Barang"] = (
        work["Nama Barang"]
        .astype(str)
        .str.strip()
    )

    work = work.loc[
        work["Nama Barang"]
        == str(item_name).strip()
    ].copy()

    if work.empty:
        return pd.DataFrame(
            columns=empty_columns
        )

    work["_periode"] = work["Bulan"].apply(
        lambda x: parse_period(
            x,
            default_year=forecast_date.year,
        )
    )

    work = work[
        work["_periode"].notna()
    ].copy()

    work = work[
        work["_periode"] < forecast_date
    ].copy()

    if work.empty:
        return pd.DataFrame(
            columns=empty_columns
        )

    work["value"] = pd.to_numeric(
        work[value_column],
        errors="coerce",
    ).fillna(0.0)

    work["value"] = work["value"].clip(
        lower=0
    )

    if "Satuan" in work.columns:
        work["Satuan"] = (
            work["Satuan"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        work["Satuan"] = ""

    grouped = (
        work.groupby(
            "_periode",
            as_index=False,
        )
        .agg(
            value=("value", "sum"),
            Satuan=("Satuan", "first"),
        )
        .sort_values("_periode")
        .reset_index(drop=True)
    )

    return grouped


def _next_period(period):
    """
    Menghasilkan periode bulan berikutnya.
    """

    parsed = parse_period(period)

    if parsed is None:
        return None

    return (
        parsed
        + pd.DateOffset(months=1)
    ).normalize()


def _build_actual_map(
    actual_history,
):
    """
    Membuat map:

        Timestamp -> actual value

    dari histori actual.

    Map ini dipakai agar actual intermediate selalu
    meng-override forecast recursive.
    """

    actual_map = {}

    if (
        actual_history is None
        or actual_history.empty
    ):
        return actual_map

    for _, row in actual_history.iterrows():

        period = row.get(
            "_periode"
        )

        value = row.get(
            "value"
        )

        period = parse_period(
            period
        )

        if period is None:
            continue

        actual_map[period] = max(
            0.0,
            _safe_float(
                value,
                default=0.0,
            ),
        )

    return actual_map


def recursive_forecast_to_target(
    values,
    start_period,
    target_period,
    method,
    actual_map=None,
    history_months=None,
):
    """
    Forecast recursive sampai target untuk SATU metode.

    Contoh actual Jan-Ags -> target Okt:
        metode X
        Ags -> forecast Sep -> forecast Okt

    Jika actual September tersedia, actual tersebut otomatis
    meng-override forecast September. Forecast intermediate
    hanya menjadi input recursive dan tidak masuk WAPE.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    values = np.maximum(values, 0.0)

    start_period = parse_period(start_period)
    target_period = parse_period(target_period)

    if start_period is None or target_period is None:
        return {
            "forecast": np.nan,
            "steps": [],
            "history_values": values,
        }

    if target_period <= start_period:
        return {
            "forecast": (
                float(values[-1])
                if len(values) > 0
                else np.nan
            ),
            "steps": [],
            "history_values": values,
        }

    if actual_map is None:
        actual_map = {}

    history_limit = _normalize_history_months(history_months)
    buffer_values = list(values.astype(float))
    current_period = start_period
    steps = []

    while current_period < target_period:
        next_period = _next_period(current_period)

        if next_period is None:
            break

        if next_period in actual_map:
            next_value = max(
                0.0,
                _safe_float(
                    actual_map[next_period],
                    default=0.0,
                ),
            )
            source = "actual"
        else:
            prediction = forecast_with_method(
                np.asarray(buffer_values, dtype=float),
                method,
            )

            if not np.isfinite(prediction):
                return {
                    "forecast": np.nan,
                    "steps": steps,
                    "history_values": np.asarray(
                        buffer_values,
                        dtype=float,
                    ),
                }

            next_value = max(0.0, float(prediction))
            source = "forecast"

        buffer_values.append(next_value)

        if (
            history_limit is not None
            and history_limit > 0
            and len(buffer_values) > history_limit
        ):
            buffer_values = buffer_values[-history_limit:]

        steps.append(
            {
                "period": next_period,
                "value": next_value,
                "source": source,
                "method": _normalize_method_name(method),
            }
        )

        current_period = next_period

    if not steps:
        forecast_value = (
            float(buffer_values[-1])
            if buffer_values
            else np.nan
        )
    else:
        forecast_value = float(steps[-1]["value"])

    return {
        "forecast": forecast_value,
        "steps": steps,
        "history_values": np.asarray(
            buffer_values,
            dtype=float,
        ),
    }


def _forecast_item_recursive(
    df,
    item_name,
    value_column,
    forecast_date,
    history_months=None,
):
    """
    Menghasilkan forecast untuk tiga metode secara terpisah.

    Tidak ada Best Method. Setiap metode memiliki:
        - forecast sendiri
        - WAPE sendiri
        - backtest sendiri
        - recursive steps sendiri
    """

    history = prepare_item_history(
        df=df,
        item_name=item_name,
        value_column=value_column,
        forecast_date=forecast_date,
        history_months=history_months,
    )

    history_count = len(history)
    satuan = ""

    if (
        not history.empty
        and "Satuan" in history.columns
    ):
        satuan_values = (
            history["Satuan"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if not satuan_values.empty:
            satuan = satuan_values.iloc[-1]

    empty_methods = {
        "MA": {
            "forecast": np.nan,
            "wape": np.nan,
            "actual": [],
            "backtest_forecast": [],
            "recursive_steps": [],
            "available": False,
        },
        "WMA": {
            "forecast": np.nan,
            "wape": np.nan,
            "actual": [],
            "backtest_forecast": [],
            "recursive_steps": [],
            "available": False,
        },
        "XGBoost": {
            "forecast": np.nan,
            "wape": np.nan,
            "actual": [],
            "backtest_forecast": [],
            "recursive_steps": [],
            "available": XGBOOST_AVAILABLE,
        },
    }

    if history_count < 3:
        return {
            "history": history,
            "history_count": history_count,
            "satuan": satuan,
            "methods": empty_methods,
            "forecast": np.nan,
            "recursive_steps": [],
        }

    values = history["value"].astype(float).to_numpy()
    methods = evaluate_all_methods(values)

    start_period = parse_period(history["_periode"].iloc[-1])

    actual_history = _prepare_item_actual_history(
        df=df,
        item_name=item_name,
        value_column=value_column,
        forecast_date=forecast_date,
    )

    actual_map = _build_actual_map(actual_history)

    for method in ("MA", "WMA", "XGBoost"):
        if method not in methods:
            continue

        if method == "XGBoost" and not XGBOOST_AVAILABLE:
            continue

        recursive_result = recursive_forecast_to_target(
            values=values,
            start_period=start_period,
            target_period=forecast_date,
            method=method,
            actual_map=actual_map,
            history_months=history_months,
        )

        methods[method]["forecast"] = recursive_result["forecast"]
        methods[method]["recursive_steps"] = recursive_result["steps"]
        methods[method]["available"] = True

    return {
        "history": history,
        "history_count": history_count,
        "satuan": satuan,
        "methods": methods,
        "forecast": np.nan,
        "recursive_steps": [],
    }


# =========================================================
# FORECAST ONE STREAM
# =========================================================

def forecast_stream(
    df,
    value_column,
    forecast_date,
    history_months=None,
):
    """
    Forecast seluruh item untuk satu stream (BBB/BBT).

    Output sekarang menampilkan tiga metode secara terpisah:

        Forecast MA + WAPE MA
        Forecast WMA + WAPE WMA
        Forecast XGBoost + WAPE XGBoost

    Tidak ada lagi kolom Best Method.
    """

    empty_summary = {
        "wape": np.nan,
        "accuracy": np.nan,
        "total_actual": 0.0,
        "total_error": 0.0,
        "wape_ma": np.nan,
        "wape_wma": np.nan,
        "wape_xgboost": np.nan,
        "accuracy_ma": np.nan,
        "accuracy_wma": np.nan,
        "accuracy_xgboost": np.nan,
        "total_actual_ma": 0.0,
        "total_actual_wma": 0.0,
        "total_actual_xgboost": 0.0,
        "total_error_ma": 0.0,
        "total_error_wma": 0.0,
        "total_error_xgboost": 0.0,
        "best_method": None,
    }

    if df is None or df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS), empty_summary

    if "Nama Barang" not in df.columns:
        return pd.DataFrame(columns=RESULT_COLUMNS), empty_summary

    history_months = _normalize_history_months(history_months)

    items = (
        df["Nama Barang"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    items = sorted(
        [item for item in items.unique() if item]
    )

    result_rows = []

    # Aggregate WAPE dihitung TERPISAH untuk masing-masing metode.
    aggregate = {
        "MA": {"actual": [], "forecast": []},
        "WMA": {"actual": [], "forecast": []},
        "XGBoost": {"actual": [], "forecast": []},
    }

    for item_name in items:
        item_result = _forecast_item_recursive(
            df=df,
            item_name=item_name,
            value_column=value_column,
            forecast_date=forecast_date,
            history_months=history_months,
        )

        history = item_result["history"]
        history_count = item_result["history_count"]
        satuan = item_result["satuan"]
        methods = item_result["methods"]

        row = {
            "Nama Barang": item_name,
            "Satuan": satuan,
            "Forecast MA": np.nan,
            "WAPE MA": np.nan,
            "Forecast WMA": np.nan,
            "WAPE WMA": np.nan,
            "Forecast XGBoost": np.nan,
            "WAPE XGBoost": np.nan,
            "Histori": history_count,
        }

        if history_count < 3:
            result_rows.append(row)
            continue

        for method, forecast_col, wape_col in (
            ("MA", "Forecast MA", "WAPE MA"),
            ("WMA", "Forecast WMA", "WAPE WMA"),
            ("XGBoost", "Forecast XGBoost", "WAPE XGBoost"),
        ):
            details = methods.get(method, {})
            forecast = details.get("forecast", np.nan)
            wape = details.get("wape", np.nan)

            row[forecast_col] = (
                round(float(forecast), 2)
                if np.isfinite(forecast)
                else np.nan
            )

            row[wape_col] = (
                round(float(wape), 3)
                if np.isfinite(wape)
                else np.nan
            )

            # Hanya data actual-vs-backtest yang dikumpulkan.
            # Recursive forecast tidak pernah menjadi pasangan WAPE.
            if (
                details.get("actual")
                and details.get("backtest_forecast")
            ):
                aggregate[method]["actual"].extend(
                    details["actual"]
                )
                aggregate[method]["forecast"].extend(
                    details["backtest_forecast"]
                )

        result_rows.append(row)

    result_df = pd.DataFrame(
        result_rows,
        columns=RESULT_COLUMNS,
    )

    # -----------------------------------------------------
    # Aggregate WAPE masing-masing metode
    # -----------------------------------------------------

    method_summary = {}

    for method in ("MA", "WMA", "XGBoost"):
        actual_values = aggregate[method]["actual"]
        forecast_values = aggregate[method]["forecast"]

        if actual_values and forecast_values:
            actual_array = np.asarray(
                actual_values,
                dtype=float,
            )
            forecast_array = np.asarray(
                forecast_values,
                dtype=float,
            )

            total_actual = float(
                np.sum(np.abs(actual_array))
            )
            total_error = float(
                np.sum(
                    np.abs(
                        actual_array
                        - forecast_array
                    )
                )
            )

            method_wape = calculate_wape(
                actual_array,
                forecast_array,
            )
        else:
            total_actual = 0.0
            total_error = 0.0
            method_wape = np.nan

        method_accuracy = (
            calculate_accuracy(method_wape)
            if np.isfinite(method_wape)
            else np.nan
        )

        method_summary[method] = {
            "wape": (
                round(float(method_wape), 3)
                if np.isfinite(method_wape)
                else np.nan
            ),
            "accuracy": (
                round(float(method_accuracy), 3)
                if np.isfinite(method_accuracy)
                else np.nan
            ),
            "total_actual": round(total_actual, 2),
            "total_error": round(total_error, 2),
        }

    # ``wape``/``accuracy`` lama dipertahankan sebagai alias MA
    # agar kode UI lama tidak langsung error, tetapi tidak lagi
    # dianggap sebagai Best Method.
    summary = {
        "wape": method_summary["MA"]["wape"],
        "accuracy": method_summary["MA"]["accuracy"],
        "total_actual": method_summary["MA"]["total_actual"],
        "total_error": method_summary["MA"]["total_error"],
        "wape_ma": method_summary["MA"]["wape"],
        "wape_wma": method_summary["WMA"]["wape"],
        "wape_xgboost": method_summary["XGBoost"]["wape"],
        "accuracy_ma": method_summary["MA"]["accuracy"],
        "accuracy_wma": method_summary["WMA"]["accuracy"],
        "accuracy_xgboost": method_summary["XGBoost"]["accuracy"],
        "total_actual_ma": method_summary["MA"]["total_actual"],
        "total_actual_wma": method_summary["WMA"]["total_actual"],
        "total_actual_xgboost": method_summary["XGBoost"]["total_actual"],
        "total_error_ma": method_summary["MA"]["total_error"],
        "total_error_wma": method_summary["WMA"]["total_error"],
        "total_error_xgboost": method_summary["XGBoost"]["total_error"],
        "best_method": None,
        "methods": method_summary,
    }

    return result_df, summary


# =========================================================
# RUN FORECASTING
# =========================================================

def run_forecasting(
    df,
    forecast_period,
    history_months=None,
):
    """
    Fungsi utama yang dipanggil oleh main.py.

    Contoh:

        run_forecasting(
            df=df,
            forecast_period="September 2026",
            history_months=None,
        )

    Jika history_months=None:
        -> SEMUA histori tersedia dipakai.

    Jika history_months=8:
        -> maksimal 8 bulan terakhir dipakai.

    Jika history_months=3:
        -> maksimal 3 bulan terakhir dipakai.

    Forecast ke bulan yang lebih jauh akan dilakukan
    secara recursive.

    Return:

        df_bbb,
        df_bbt,
        summary
    """

    if df is None or df.empty:

        empty_df = pd.DataFrame(
            columns=RESULT_COLUMNS
        )

        summary = {
            "bbb": {
                "wape": np.nan,
                "accuracy": np.nan,
                "total_actual": 0.0,
                "total_error": 0.0,
                "wape_ma": np.nan,
                "wape_wma": np.nan,
                "wape_xgboost": np.nan,
                "accuracy_ma": np.nan,
                "accuracy_wma": np.nan,
                "accuracy_xgboost": np.nan,
                "total_actual_ma": 0.0,
                "total_actual_wma": 0.0,
                "total_actual_xgboost": 0.0,
                "total_error_ma": 0.0,
                "total_error_wma": 0.0,
                "total_error_xgboost": 0.0,
                "best_method": None,
            },
            "bbt": {
                "wape": np.nan,
                "accuracy": np.nan,
                "total_actual": 0.0,
                "total_error": 0.0,
                "wape_ma": np.nan,
                "wape_wma": np.nan,
                "wape_xgboost": np.nan,
                "accuracy_ma": np.nan,
                "accuracy_wma": np.nan,
                "accuracy_xgboost": np.nan,
                "total_actual_ma": 0.0,
                "total_actual_wma": 0.0,
                "total_actual_xgboost": 0.0,
                "total_error_ma": 0.0,
                "total_error_wma": 0.0,
                "total_error_xgboost": 0.0,
                "best_method": None,
            },
        }

        return (
            empty_df,
            empty_df.copy(),
            summary,
        )

    # -----------------------------------------------------
    # Forecast date
    # -----------------------------------------------------

    forecast_date = parse_period(
        forecast_period
    )

    if forecast_date is None:

        raise ValueError(
            "Periode forecast tidak dapat dibaca. "
            "Gunakan format seperti 'September 2026'."
        )

    # -----------------------------------------------------
    # Normalize history months
    #
    # None = semua histori
    # -----------------------------------------------------

    history_months = _normalize_history_months(
        history_months
    )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "Bulan",
        "Nama Barang",
        "OUT BBB",
        "OUT BBT",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Kolom Data OUT belum lengkap: "
            + ", ".join(
                missing_columns
            )
        )

    # -----------------------------------------------------
    # BBB
    # -----------------------------------------------------

    df_bbb, summary_bbb = (
        forecast_stream(
            df=df,
            value_column="OUT BBB",
            forecast_date=forecast_date,
            history_months=history_months,
        )
    )

    # -----------------------------------------------------
    # BBT
    # -----------------------------------------------------

    df_bbt, summary_bbt = (
        forecast_stream(
            df=df,
            value_column="OUT BBT",
            forecast_date=forecast_date,
            history_months=history_months,
        )
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {
        "bbb": summary_bbb,
        "bbt": summary_bbt,
    }

    return (
        df_bbb,
        df_bbt,
        summary,
    )


# =========================================================
# XGBOOST STATUS
# =========================================================

def get_xgboost_status(
    history_count: int,
) -> Dict:
    """
    Helper untuk mengetahui status XGBoost.
    """

    try:

        history_count = int(
            history_count
        )

    except Exception:

        history_count = 0

    if not XGBOOST_AVAILABLE:

        return {
            "available": False,
            "eligible": False,
            "history_count": history_count,
            "minimum_history": (
                XGBOOST_MIN_HISTORY
            ),
            "message": (
                "XGBoost belum tersedia "
                "karena package belum ter-install."
            ),
        }

    if history_count < XGBOOST_MIN_HISTORY:

        return {
            "available": True,
            "eligible": False,
            "history_count": history_count,
            "minimum_history": (
                XGBOOST_MIN_HISTORY
            ),
            "message": (
                f"Histori {history_count} bulan "
                f"belum memenuhi minimum XGBoost "
                f"{XGBOOST_MIN_HISTORY} bulan."
            ),
        }

    return {
        "available": True,
        "eligible": True,
        "history_count": history_count,
        "minimum_history": (
            XGBOOST_MIN_HISTORY
        ),
        "message": (
            f"Histori {history_count} bulan "
            f"memenuhi minimum XGBoost "
            f"{XGBOOST_MIN_HISTORY} bulan."
        ),
    }


def get_recursive_forecasting_status() -> Dict:
    """
    Helper untuk mengetahui status fitur recursive
    forecasting.

    Fitur ini selalu tersedia karena recursive forecasting
    menggunakan method yang sudah tersedia di module ini.
    """

    return {
        "available": True,
        "enabled": True,
        "message": (
            "Recursive forecasting aktif. "
            "Jika periode target lebih dari satu bulan "
            "setelah histori actual terakhir, forecast "
            "dihitung bertahap sampai target. Actual "
            "intermediate selalu meng-override forecast."
        ),
    }


def explain_recursive_forecast() -> str:
    """
    Penjelasan singkat yang dapat ditampilkan oleh UI.
    """

    return (
        "Recursive forecasting bekerja bertahap untuk setiap metode "
        "secara terpisah: MA, WMA, dan XGBoost. Contoh: jika actual "
        "tersedia sampai Agustus dan target Oktober, masing-masing "
        "metode forecast September lalu menggunakan hasil tersebut "
        "untuk forecast Oktober. Jika actual September tersedia, "
        "actual September digunakan dan meng-override forecast "
        "September. WAPE dihitung terpisah untuk MA, WMA, dan XGBoost "
        "berdasarkan backtesting actual historis; forecast recursive "
        "intermediate tidak digunakan untuk menghitung WAPE. Tidak ada "
        "lagi pemilihan Best Method."
    )


# =========================================================
# MODULE TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "FORECASTING MODULE"
    )

    print(
        "=" * 60
    )

    print(
        "XGBoost available :",
        XGBOOST_AVAILABLE,
    )

    print(
        "XGBoost min history :",
        XGBOOST_MIN_HISTORY,
    )

    print(
        "Recursive forecasting :",
        get_recursive_forecasting_status()[
            "enabled"
        ],
    )

    print(
        "Forecast methods      : MA, WMA, XGBoost"
    )

    print(
        "Best Method selection : disabled"
    )

    print()

    for count in [
        1,
        2,
        3,
        4,
        6,
        8,
        12,
    ]:

        print(
            f"Histori {count} bulan:"
        )

        print(
            get_available_methods(
                count
            )
        )

        print()

# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
# dengan UI lama tidak langsung gagal saat module dimuat.
# ---------------------------------------------------------
# CATATAN ARSITEKTUR: THREE-METHOD FORECASTING
# ---------------------------------------------------------
# Forecast utama tidak lagi memilih satu metode terbaik.
# MA, WMA, dan XGBoost dihitung secara independen.
# Setiap metode memiliki forecast dan WAPE masing-masing.
# WAPE berasal dari walk-forward backtest pada actual.
# Forecast recursive intermediate tidak masuk perhitungan WAPE.
# Actual intermediate selalu meng-override forecast recursive.
# Target multi-bulan dihitung bertahap untuk setiap metode.
# MA dan WMA memakai window default 3 bulan.
# Alias MA3 dan WMA3 tetap didukung untuk kompatibilitas.
# XGBoost hanya aktif jika package tersedia dan histori cukup.
# Tidak ada ranking Best Method di engine forecast baru.
# Summary tetap menyediakan field legacy agar integrasi bertahap
