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
    "Forecast",
    "Best Method",
    "WAPE",
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

def forecast_with_method(
    values,
    method: str,
) -> float:
    """
    Forecast menggunakan method tertentu.

    Supported:

        MA2
        MA3
        ...
        WMA2
        WMA3
        ...
        XGBoost
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
    )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    if method_clean in {
        "XGBOOST",
        "XG BOOST",
        "XGB",
    }:

        return xgboost_forecast(
            values
        )

    # -----------------------------------------------------
    # WMA
    # -----------------------------------------------------

    if method_clean.startswith(
        "WMA"
    ):

        try:

            window = int(
                method_clean.replace(
                    "WMA",
                    "",
                )
            )

            if window <= 0:
                return np.nan

            return weighted_moving_average(
                values,
                window,
            )

        except Exception:
            return np.nan

    # -----------------------------------------------------
    # MA
    # -----------------------------------------------------

    if method_clean.startswith(
        "MA"
    ):

        try:

            window = int(
                method_clean.replace(
                    "MA",
                    "",
                )
            )

            if window <= 0:
                return np.nan

            return moving_average(
                values,
                window,
            )

        except Exception:
            return np.nan

    return np.nan


# =========================================================
# GENERIC BACKTEST DETAILS
# =========================================================

def backtest_method_details(
    values,
    method: str,
):
    """
    Backtesting method statistik.

    Untuk MA/WMA:

        training -> test bulan berikutnya

    Actual bulan test tidak digunakan
    untuk menghitung forecast.
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
    )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    if method_clean in {
        "XGBOOST",
        "XG BOOST",
        "XGB",
    }:

        return backtest_xgboost_details(
            values
        )

    # -----------------------------------------------------
    # Parse window
    # -----------------------------------------------------

    try:

        if method_clean.startswith(
            "WMA"
        ):

            window = int(
                method_clean.replace(
                    "WMA",
                    "",
                )
            )

        elif method_clean.startswith(
            "MA"
        ):

            window = int(
                method_clean.replace(
                    "MA",
                    "",
                )
            )

        else:

            return {
                "actual": [],
                "forecast": [],
                "wape": np.nan,
            }

    except Exception:

        return {
            "actual": [],
            "forecast": [],
            "wape": np.nan,
        }

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

            if method_clean.startswith(
                "WMA"
            ):

                prediction = (
                    weighted_moving_average(
                        train_values,
                        window,
                    )
                )

            else:

                prediction = (
                    moving_average(
                        train_values,
                        window,
                    )
                )

            if not np.isfinite(
                prediction
            ):
                continue

            prediction = max(
                0.0,
                float(prediction),
            )

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
    Menentukan method berdasarkan jumlah histori.

    1-2 bulan:
        tidak cukup

    3 bulan:
        MA2
        WMA2
        XGBoost

    4 bulan:
        MA2 - MA3
        WMA2 - WMA3
        XGBoost

    8 bulan:
        MA2 - MA7
        WMA2 - WMA7
        XGBoost

    12 bulan:
        MA2 - MA11
        WMA2 - WMA11
        XGBoost

    Semakin panjang histori,
    semakin banyak window MA/WMA yang
    bisa dibandingkan.
    """

    try:

        history_count = int(
            history_count
        )

    except Exception:
        return []

    if history_count < 3:
        return []

    methods = []

    # -----------------------------------------------------
    # MA
    # -----------------------------------------------------

    for window in range(
        2,
        history_count,
    ):

        methods.append(
            f"MA{window}"
        )

    # -----------------------------------------------------
    # WMA
    # -----------------------------------------------------

    for window in range(
        2,
        history_count,
    ):

        methods.append(
            f"WMA{window}"
        )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    if (
        XGBOOST_AVAILABLE
        and history_count >= XGBOOST_MIN_HISTORY
    ):

        methods.append(
            "XGBoost"
        )

    return methods


# =========================================================
# AUTO BEST METHOD
# =========================================================

def auto_best_method(values):
    """
    Memilih method dengan WAPE
    backtesting terendah.

    Setelah method terbaik ditemukan,
    forecast dibuat menggunakan SELURUH
    histori yang diberikan.
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

    history_count = len(values)

    methods = get_available_methods(
        history_count
    )

    if not methods:

        return {
            "method": None,
            "wape": np.nan,
            "forecast": np.nan,
            "actual": [],
            "backtest_forecast": [],
        }

    best_method = None
    best_wape = np.inf
    best_details = None

    # -----------------------------------------------------
    # Backtesting semua candidate
    # -----------------------------------------------------

    for method in methods:

        details = (
            backtest_method_details(
                values,
                method,
            )
        )

        wape = details["wape"]

        if not np.isfinite(
            wape
        ):
            continue

        if wape < best_wape:

            best_wape = wape
            best_method = method
            best_details = details

    # -----------------------------------------------------
    # Tidak ada method valid
    # -----------------------------------------------------

    if (
        best_method is None
        or best_details is None
    ):

        return {
            "method": None,
            "wape": np.nan,
            "forecast": np.nan,
            "actual": [],
            "backtest_forecast": [],
        }

    # -----------------------------------------------------
    # Forecast menggunakan seluruh histori
    # -----------------------------------------------------

    forecast = forecast_with_method(
        values,
        best_method,
    )

    if not np.isfinite(
        forecast
    ):

        forecast = np.nan

    elif forecast < 0:

        forecast = 0.0

    return {
        "method": best_method,
        "wape": float(best_wape),
        "forecast": forecast,
        "actual": best_details[
            "actual"
        ],
        "backtest_forecast": (
            best_details[
                "forecast"
            ]
        ),
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
# FORECAST ONE STREAM
# =========================================================

def forecast_stream(
    df,
    value_column,
    forecast_date,
    history_months=None,
):
    """
    Forecast seluruh item untuk satu stream.

    value_column:

        OUT BBB
        OUT BBT

    history_months:

        None
            -> semua histori tersedia

        3
            -> maksimal 3 bulan terakhir

        8
            -> maksimal 8 bulan terakhir
    """

    empty_summary = {
        "wape": np.nan,
        "accuracy": np.nan,
        "total_actual": 0.0,
        "total_error": 0.0,
        "best_method": None,
    }

    if df is None or df.empty:

        return (
            pd.DataFrame(
                columns=RESULT_COLUMNS
            ),
            empty_summary,
        )

    if "Nama Barang" not in df.columns:

        return (
            pd.DataFrame(
                columns=RESULT_COLUMNS
            ),
            empty_summary,
        )

    # -----------------------------------------------------
    # Normalisasi history setting
    # -----------------------------------------------------

    history_months = _normalize_history_months(
        history_months
    )

    # -----------------------------------------------------
    # Unique item
    # -----------------------------------------------------

    items = (
        df["Nama Barang"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    items = sorted(
        [
            item
            for item in items.unique()
            if item
        ]
    )

    result_rows = []

    aggregate_actual = []
    aggregate_forecast = []

    selected_methods = []

    # -----------------------------------------------------
    # Per item
    # -----------------------------------------------------

    for item_name in items:

        history = prepare_item_history(
            df=df,
            item_name=item_name,
            value_column=value_column,
            forecast_date=forecast_date,
            history_months=history_months,
        )

        history_count = len(
            history
        )

        # -------------------------------------------------
        # Satuan
        # -------------------------------------------------

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

                satuan = (
                    satuan_values.iloc[-1]
                )

        # -------------------------------------------------
        # Minimum 3 bulan
        # -------------------------------------------------

        if history_count < 3:

            result_rows.append(
                {
                    "Nama Barang": item_name,
                    "Satuan": satuan,
                    "Forecast": np.nan,
                    "Best Method": (
                        "Histori Tidak Cukup"
                    ),
                    "WAPE": np.nan,
                    "Histori": history_count,
                }
            )

            continue

        values = (
            history["value"]
            .astype(float)
            .to_numpy()
        )

        # -------------------------------------------------
        # Auto Best Method
        # -------------------------------------------------

        best = auto_best_method(
            values
        )

        method = best["method"]
        wape = best["wape"]
        forecast = best["forecast"]

        if method is not None:

            selected_methods.append(
                method
            )

        # -------------------------------------------------
        # Aggregate backtest
        # -------------------------------------------------

        if (
            best["actual"]
            and best["backtest_forecast"]
        ):

            aggregate_actual.extend(
                best["actual"]
            )

            aggregate_forecast.extend(
                best["backtest_forecast"]
            )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        result_rows.append(
            {
                "Nama Barang": item_name,
                "Satuan": satuan,

                "Forecast": (
                    round(
                        float(forecast),
                        2,
                    )
                    if np.isfinite(
                        forecast
                    )
                    else np.nan
                ),

                "Best Method": (
                    method
                    if method is not None
                    else "Tidak Tersedia"
                ),

                "WAPE": (
                    round(
                        float(wape),
                        3,
                    )
                    if np.isfinite(
                        wape
                    )
                    else np.nan
                ),

                "Histori": history_count,
            }
        )

    # -----------------------------------------------------
    # DataFrame result
    # -----------------------------------------------------

    result_df = pd.DataFrame(
        result_rows,
        columns=RESULT_COLUMNS,
    )

    # -----------------------------------------------------
    # Aggregate WAPE
    # -----------------------------------------------------

    if (
        aggregate_actual
        and aggregate_forecast
    ):

        aggregate_actual_array = np.asarray(
            aggregate_actual,
            dtype=float,
        )

        aggregate_forecast_array = np.asarray(
            aggregate_forecast,
            dtype=float,
        )

        total_actual = float(
            np.sum(
                np.abs(
                    aggregate_actual_array
                )
            )
        )

        total_error = float(
            np.sum(
                np.abs(
                    aggregate_actual_array
                    - aggregate_forecast_array
                )
            )
        )

        if total_actual == 0:

            if total_error == 0:
                stream_wape = 0.0
            else:
                stream_wape = 100.0

        else:

            stream_wape = (
                total_error
                / total_actual
                * 100.0
            )

    else:

        total_actual = 0.0
        total_error = 0.0
        stream_wape = np.nan

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    stream_accuracy = (
        calculate_accuracy(
            stream_wape
        )
        if np.isfinite(
            stream_wape
        )
        else np.nan
    )

    # -----------------------------------------------------
    # Most common best method
    # -----------------------------------------------------

    if selected_methods:

        method_counts = (
            pd.Series(
                selected_methods
            )
            .value_counts()
        )

        stream_best_method = (
            method_counts.index[0]
        )

    else:

        stream_best_method = None

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {
        "wape": (
            round(
                float(stream_wape),
                3,
            )
            if np.isfinite(
                stream_wape
            )
            else np.nan
        ),

        "accuracy": (
            round(
                float(stream_accuracy),
                3,
            )
            if np.isfinite(
                stream_accuracy
            )
            else np.nan
        ),

        "total_actual": round(
            total_actual,
            2,
        ),

        "total_error": round(
            total_error,
            2,
        ),

        "best_method": (
            stream_best_method
        ),
    }

    return (
        result_df,
        summary,
    )


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
                "best_method": None,
            },

            "bbt": {
                "wape": np.nan,
                "accuracy": np.nan,
                "total_actual": 0.0,
                "total_error": 0.0,
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