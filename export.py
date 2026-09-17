import pandas as pd
from io import BytesIO

from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


# =========================================================
# CONSTANT
# =========================================================

FORECAST_COLUMNS = [
    "Nama Barang",
    "Satuan",
    "Forecast MA",
    "WAPE MA",
    "Forecast WMA",
    "WAPE WMA",
    "Forecast XGBoost",
    "WAPE XGBoost",
]


EXPORT_COLUMNS = [
    "Nama Barang",
    "Satuan",
    "Forecast MA",
    "WAPE MA (%)",
    "Accuracy MA (%)",
    "Forecast WMA",
    "WAPE WMA (%)",
    "Accuracy WMA (%)",
    "Forecast XGBoost",
    "WAPE XGBoost (%)",
    "Accuracy XGBoost (%)",
]


# Kolom tambahan yang dapat dipakai oleh forecasting recursive.
# Tidak diwajibkan agar tetap kompatibel dengan dataframe lama.
OPTIONAL_RECURSIVE_COLUMNS = [
    "Periode",
    "Forecast Period",
    "Target Period",
    "Sumber",
    "Source",
    "Method",
    "Recursive",
]


# =========================================================
# HELPER
# =========================================================

def _safe_dataframe(data):
    """
    Mengubah input menjadi DataFrame dengan aman.

    Bisa menerima:
    - pandas DataFrame
    - list of dict
    - list
    - None
    """

    if data is None:

        return pd.DataFrame()

    if isinstance(
        data,
        pd.DataFrame,
    ):

        return data.copy()

    try:

        return pd.DataFrame(
            data
        )

    except Exception:

        return pd.DataFrame()


def _safe_text(value):
    """
    Mengubah nilai menjadi text yang aman
    untuk metadata atau nama file.
    """

    if value is None:

        return ""

    try:

        return str(value).strip()

    except Exception:

        return ""


def _numeric_series(
    series,
    decimals=2,
):
    """
    Konversi Series ke numeric dengan aman.

    Nilai yang tidak valid menjadi NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    ).round(
        decimals
    )


def _ordered_columns(
    df,
    preferred_columns,
):
    """
    Menempatkan kolom prioritas di depan,
    lalu mempertahankan kolom lainnya.
    """

    if df.empty:

        return df

    existing_columns = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    other_columns = [
        col
        for col in df.columns
        if col not in existing_columns
    ]

    return df[
        existing_columns
        +
        other_columns
    ]


# =========================================================
# FORMAT FORECAST DATAFRAME
# =========================================================

def _format_forecast_dataframe(
    df
):
    """
    Menyiapkan DataFrame forecast
    agar rapi ketika masuk Excel.

    Tetap mempertahankan kolom tambahan yang mungkin
    berasal dari proses recursive forecasting.
    """

    df = _safe_dataframe(
        df
    )

    if df.empty:

        return df

    result = df.copy()

    # Kolom Histori hanya dibutuhkan internal untuk perhitungan.
    # Jangan tampilkan di file Excel hasil forecast.
    if "Histori" in result.columns:
        result = result.drop(columns=["Histori"])

    # -----------------------------------------------------
    # Kolom utama
    # -----------------------------------------------------

    preferred_columns = [
        "Nama Barang",
        "Satuan",
        "Forecast MA",
        "WAPE MA",
        "Accuracy MA",
        "Forecast WMA",
        "WAPE WMA",
        "Accuracy WMA",
        "Forecast XGBoost",
        "WAPE XGBoost",
        "Accuracy XGBoost",
    ]

    result = _ordered_columns(
        result,
        preferred_columns,
    )

    # -----------------------------------------------------
    # Rename
    # -----------------------------------------------------

    rename_map = {
        "WAPE MA":
            "WAPE MA (%)",

        "Accuracy MA":
            "Accuracy MA (%)",

        "WAPE WMA":
            "WAPE WMA (%)",

        "Accuracy WMA":
            "Accuracy WMA (%)",

        "WAPE XGBoost":
            "WAPE XGBoost (%)",

        "Accuracy XGBoost":
            "Accuracy XGBoost (%)",
    }

    result = result.rename(
        columns=rename_map
    )

    # -----------------------------------------------------
    # Numeric Forecast - all methods
    # -----------------------------------------------------

    forecast_columns = [
        "Forecast MA",
        "Forecast WMA",
        "Forecast XGBoost",
    ]

    for column in forecast_columns:
        if column in result.columns:
            result[column] = _numeric_series(
                result[column],
                decimals=2,
            )

    # -----------------------------------------------------
    # Numeric WAPE - all methods
    # -----------------------------------------------------

    wape_columns = [
        "WAPE MA (%)",
        "WAPE WMA (%)",
        "WAPE XGBoost (%)",
    ]

    for column in wape_columns:
        if column in result.columns:
            result[column] = _numeric_series(
                result[column],
                decimals=2,
            )

    # -----------------------------------------------------
    # Accuracy - all methods
    # -----------------------------------------------------

    accuracy_pairs = [
        ("Accuracy MA (%)", "WAPE MA (%)"),
        ("Accuracy WMA (%)", "WAPE WMA (%)"),
        ("Accuracy XGBoost (%)", "WAPE XGBoost (%)"),
    ]

    for accuracy_column, wape_column in accuracy_pairs:
        if accuracy_column in result.columns:
            result[accuracy_column] = _numeric_series(
                result[accuracy_column],
                decimals=2,
            )
        elif wape_column in result.columns:
            result[accuracy_column] = (
                100.0
                - pd.to_numeric(
                    result[wape_column],
                    errors="coerce",
                )
            ).round(2)

    # -----------------------------------------------------
    # Normalize optional recursive flag
    # -----------------------------------------------------

    if "Recursive" in result.columns:

        result[
            "Recursive"
        ] = result[
            "Recursive"
        ].map(
            lambda value:
                "Ya"
                if str(value).strip().lower()
                in {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "ya",
                }
                else (
                    "Tidak"
                    if str(value).strip().lower()
                    in {
                        "false",
                        "0",
                        "no",
                        "n",
                        "tidak",
                    }
                    else value
                )
        )

    return result


# =========================================================
# EMPTY EXPORT DATAFRAME
# =========================================================

def _empty_export_dataframe():

    return pd.DataFrame(
        columns=EXPORT_COLUMNS
    )


# =========================================================
# EMPTY RECURSIVE DETAIL
# =========================================================

def _empty_recursive_dataframe():

    return pd.DataFrame(
        columns=[
            "Nama Barang",
            "Periode",
            "Nilai",
            "Sumber",
            "Method",
        ]
    )


# =========================================================
# FORMAT RECURSIVE DETAIL
# =========================================================

def _format_recursive_dataframe(
    data
):
    """
    Menyiapkan detail recursive forecasting.

    Fungsi ini opsional dan tidak mengubah format
    forecast utama.
    """

    df = _safe_dataframe(
        data
    )

    if df.empty:

        return _empty_recursive_dataframe()

    result = df.copy()

    # -----------------------------------------------------
    # Rename nilai internal jika tersedia
    # -----------------------------------------------------

    rename_map = {}

    if (
        "Nilai"
        in result.columns
    ):

        rename_map[
            "Nilai"
        ] = "Nilai Forecast / Actual"

    if (
        "Method"
        in result.columns
    ):

        rename_map[
            "Method"
        ] = "Metode"

    result = result.rename(
        columns=rename_map
    )

    # -----------------------------------------------------
    # Numeric value
    # -----------------------------------------------------

    if (
        "Nilai Forecast / Actual"
        in result.columns
    ):

        result[
            "Nilai Forecast / Actual"
        ] = _numeric_series(
            result[
                "Nilai Forecast / Actual"
            ],
            decimals=2,
        )

    # -----------------------------------------------------
    # Prioritas kolom detail
    # -----------------------------------------------------

    preferred_columns = [
        "Nama Barang",
        "Periode",
        "Nilai Forecast / Actual",
        "Sumber",
        "Metode",
    ]

    result = _ordered_columns(
        result,
        preferred_columns,
    )

    return result


# =========================================================
# STYLE WORKSHEET
# =========================================================

def _format_worksheet(
    worksheet
):

    # -----------------------------------------------------
    # Freeze header
    # -----------------------------------------------------

    worksheet.freeze_panes = "A2"

    # -----------------------------------------------------
    # Filter
    # -----------------------------------------------------

    if (
        worksheet.max_row >= 1
        and worksheet.max_column >= 1
    ):

        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

    # -----------------------------------------------------
    # Header style
    # -----------------------------------------------------

    for cell in worksheet[1]:

        cell.font = Font(
            bold=True
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    # -----------------------------------------------------
    # Vertical alignment
    # -----------------------------------------------------

    for row in worksheet.iter_rows():

        for cell in row:

            cell.alignment = Alignment(
                vertical="center"
            )

    # -----------------------------------------------------
    # Lebar kolom otomatis
    # -----------------------------------------------------

    for column_cells in worksheet.columns:

        max_length = 0

        column_index = (
            column_cells[0].column
        )

        column_letter = (
            get_column_letter(
                column_index
            )
        )

        for cell in column_cells:

            try:

                value_length = len(
                    str(
                        cell.value
                    )
                )

                if (
                    value_length
                    >
                    max_length
                ):

                    max_length = (
                        value_length
                    )

            except Exception:

                pass

        worksheet.column_dimensions[
            column_letter
        ].width = min(
            max(
                max_length + 2,
                10,
            ),
            35,
        )

    # -----------------------------------------------------
    # Format angka berdasarkan header
    # -----------------------------------------------------

    headers = {
        cell.value: cell.column
        for cell in worksheet[1]
    }

    # -----------------------------------------------------
    # WAPE - all methods
    # -----------------------------------------------------

    wape_headers = [
        "WAPE MA (%)",
        "WAPE WMA (%)",
        "WAPE XGBoost (%)",
    ]

    for header in wape_headers:
        if header in headers:
            column_number = headers[header]

            for row in worksheet.iter_rows(
                min_row=2,
                min_col=column_number,
                max_col=column_number,
            ):
                row[0].number_format = "0.00"

    # -----------------------------------------------------
    # Accuracy - all methods
    # -----------------------------------------------------

    accuracy_headers = [
        "Accuracy MA (%)",
        "Accuracy WMA (%)",
        "Accuracy XGBoost (%)",
    ]

    for header in accuracy_headers:
        if header in headers:
            column_number = headers[header]

            for row in worksheet.iter_rows(
                min_row=2,
                min_col=column_number,
                max_col=column_number,
            ):
                row[0].number_format = "0.00"

    # -----------------------------------------------------
    # Forecast - all methods
    # -----------------------------------------------------

    forecast_headers = [
        "Forecast MA",
        "Forecast WMA",
        "Forecast XGBoost",
    ]

    for header in forecast_headers:
        if header in headers:
            column_number = headers[header]

            for row in worksheet.iter_rows(
                min_row=2,
                min_col=column_number,
                max_col=column_number,
            ):
                row[0].number_format = "#,##0.00"

    # -----------------------------------------------------
    # Detail recursive value
    # -----------------------------------------------------

    if (
        "Nilai Forecast / Actual"
        in headers
    ):

        column_number = (
            headers[
                "Nilai Forecast / Actual"
            ]
        )

        for row in worksheet.iter_rows(
            min_row=2,
            min_col=column_number,
            max_col=column_number,
        ):

            row[0].number_format = (
                "#,##0.00"
            )

    # -----------------------------------------------------
    # Tinggi header
    # -----------------------------------------------------

    worksheet.row_dimensions[
        1
    ].height = 24


# =========================================================
# AVERAGE ROW DETAIL SHEET
# =========================================================

def _append_average_row(dataframe):
    """
    Menambahkan satu baris RATA-RATA di bagian paling bawah
    untuk WAPE dan Forecast Accuracy masing-masing metode.

    Rata-rata dihitung hanya dari baris item yang tersedia pada
    sheet tersebut. BBB dan BBT tidak pernah digabung.
    """

    df = _safe_dataframe(dataframe)

    if df.empty:
        return df

    result = df.copy()

    average_row = {column: None for column in result.columns}

    if "Nama Barang" in result.columns:
        average_row["Nama Barang"] = "RATA-RATA"

    if "Satuan" in result.columns:
        average_row["Satuan"] = ""

    # Forecast tidak dirata-ratakan; yang diringkas adalah performa metode.
    performance_columns = [
        "WAPE MA (%)",
        "Accuracy MA (%)",
        "WAPE WMA (%)",
        "Accuracy WMA (%)",
        "WAPE XGBoost (%)",
        "Accuracy XGBoost (%)",
    ]

    for column in performance_columns:
        if column in result.columns:
            values = pd.to_numeric(
                result[column],
                errors="coerce",
            )
            if values.notna().any():
                average_row[column] = round(
                    float(values.mean()),
                    2,
                )

    return pd.concat(
        [result, pd.DataFrame([average_row])],
        ignore_index=True,
    )


# =========================================================
# WRITE DETAIL SHEET
# =========================================================

def _write_dataframe_sheet(
    writer,
    dataframe,
    sheet_name,
    empty_dataframe=None,
):
    """
    Helper penulisan sheet agar seluruh sheet
    konsisten.
    """

    df = _safe_dataframe(
        dataframe
    )

    if df.empty:

        if empty_dataframe is None:

            df = pd.DataFrame()

        else:

            df = _safe_dataframe(
                empty_dataframe
            )

    df.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
    )



# =========================================================
# PERFORMANCE + WAPE EXPLANATION HELPERS
# =========================================================

_METHODS = ("MA", "WMA", "XGBoost")
_COMPARING_COLUMNS = [
    "Nama Barang",
    "Satuan",
    "Forecast MA",
    "WAPE MA (%)",
    "Forecast WMA",
    "WAPE WMA (%)",
    "Forecast XGBoost",
    "WAPE XGBoost (%)",
    "Best Method",
    "Forecast Terpilih",
    "WAPE Terbaik (%)",
]


def _safe_summary_stream(summary, stream):
    """Ambil summary stream secara aman dari summary forecasting."""
    if not isinstance(summary, dict):
        return {}
    value = summary.get(stream, {})
    return value if isinstance(value, dict) else {}


def _summary_value(stream_summary, key):
    value = stream_summary.get(key, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def _format_value(value, decimals=2):
    try:
        number = float(value)
        if pd.isna(number):
            return "-"
        return f"{number:,.{decimals}f}"
    except Exception:
        return "-"


def _parse_period_for_export(value, default_year=None):
    """Parse Bulan menggunakan parser yang sama dengan forecasting.py."""
    try:
        from forecasting import parse_period
        return parse_period(value, default_year=default_year)
    except Exception:
        return pd.NaT


def _prepare_dynamic_example(
    history_df,
    stream,
    periode_forecast,
    history_months=None,
    forecast_df=None,
    preferred_items=("Beef L", "Beef S"),
):
    """
    Menyiapkan contoh perhitungan WAPE dari data histori/backtest nyata.

    Prioritas item:
    1. Beef L
    2. Beef S
    Bila salah satu tidak tersedia, sistem memilih item nyata lain yang
    mempunyai histori paling lengkap.

    Angka contoh tidak dibuat secara dummy. Forecast backtest dihitung
    menggunakan fungsi forecasting yang sama dengan engine aplikasi.
    """
    df = _safe_dataframe(history_df)
    if df.empty:
        return None

    stream = str(stream).upper().strip()
    value_column = "OUT BBB" if stream == "BBB" else "OUT BBT"

    required = {"Bulan", "Nama Barang", value_column}
    if not required.issubset(df.columns):
        return None

    forecast_date = _parse_period_for_export(periode_forecast)
    if forecast_date is None or pd.isna(forecast_date):
        return None

    try:
        from forecasting import (
            prepare_item_history,
            backtest_method_details,
            get_available_methods,
        )
    except Exception:
        return None

    work = df.copy()
    work["Nama Barang"] = (
        work["Nama Barang"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    work = work[work["Nama Barang"] != ""].copy()

    if work.empty:
        return None

    # Bila Satuan tersedia, pertahankan untuk contoh.
    if "Satuan" not in work.columns:
        work["Satuan"] = ""

    # Kandidat item nyata dengan histori terbanyak.
    candidates = []
    for item_name in work["Nama Barang"].drop_duplicates().tolist():
        item_history = prepare_item_history(
            work,
            item_name,
            value_column,
            forecast_date,
            history_months=history_months,
        )
        if item_history is None or item_history.empty:
            continue

        actual_values = pd.to_numeric(
            item_history["value"], errors="coerce"
        ).fillna(0.0)

        history_count = int(len(actual_values))
        positive_count = int((actual_values > 0).sum())

        if history_count < 2:
            continue

        candidates.append(
            (
                item_name,
                history_count,
                positive_count,
                item_history.copy(),
            )
        )

    if not candidates:
        return None

    def candidate_rank(item):
        name, history_count, positive_count, _ = item
        normalized = str(name).strip().casefold()
        preferred_rank = {
            str(preferred_items[0]).casefold(): 0,
            str(preferred_items[1]).casefold(): 1,
        }.get(normalized, 99)
        return (preferred_rank, -positive_count, -history_count)

    candidates.sort(key=candidate_rank)

    selected = []
    selected_names = set()

    # Prioritaskan Beef L dan Beef S bila memang ada.
    for preferred in preferred_items:
        for candidate in candidates:
            if candidate[0].strip().casefold() == str(preferred).strip().casefold():
                selected.append(candidate)
                selected_names.add(candidate[0].strip().casefold())
                break

    # Lengkapi sampai tepat 2 item jika salah satu Beef tidak tersedia.
    for candidate in candidates:
        if len(selected) >= 2:
            break
        if candidate[0].strip().casefold() not in selected_names:
            selected.append(candidate)
            selected_names.add(candidate[0].strip().casefold())

    if not selected:
        return None

    rows = []

    # Gunakan metode keluarga MA sebagai contoh WAPE yang mudah dibaca.
    # Jika MA tidak tersedia pada histori tertentu, coba WMA lalu XGBoost.
    method_priority = ["MA", "WMA", "XGBoost"]

    for item_name, history_count, positive_count, item_history in selected[:2]:
        available = []
        try:
            available = get_available_methods(history_count)
        except Exception:
            available = []

        method = next(
            (m for m in method_priority if m in available),
            None,
        )

        # Dengan histori minimal 2 bulan, MA2/WMA2 dapat digunakan.
        if method is None:
            continue

        details = backtest_method_details(
            item_history["value"].astype(float).tolist(),
            method,
        )

        if not details:
            continue

        # =====================================================
        # KOMPATIBILITAS HASIL BACKTEST
        # forecasting.py mengembalikan dict:
        # {"actual": [...], "forecast": [...], "wape": ...}.
        # Kode export lama menganggap hasilnya list of dict,
        # sehingga muncul error: 'str' object has no attribute 'get'.
        # Normalisasi di sini agar kedua bentuk tetap didukung.
        # =====================================================
        if isinstance(details, dict):
            detail_actual = list(details.get("actual", []) or [])
            detail_forecast = list(details.get("forecast", []) or [])
            details = [
                {"actual": actual_value, "forecast": forecast_value}
                for actual_value, forecast_value in zip(
                    detail_actual,
                    detail_forecast,
                )
            ]
        elif isinstance(details, (list, tuple)):
            details = [
                item
                for item in details
                if isinstance(item, dict)
            ]
        else:
            details = []

        if not details:
            continue

        actuals = pd.to_numeric(
            [x.get("actual") for x in details],
            errors="coerce",
        )
        forecasts = pd.to_numeric(
            [x.get("forecast") for x in details],
            errors="coerce",
        )

        # pd.to_numeric(list) dapat menghasilkan numpy.ndarray pada
        # kombinasi versi pandas tertentu. Ubah ke Series supaya
        # operasi .notna() dan .reset_index() selalu tersedia.
        actuals = pd.Series(actuals, dtype="float64")
        forecasts = pd.Series(forecasts, dtype="float64")

        valid = actuals.notna() & forecasts.notna()
        if not valid.any():
            continue

        actuals = actuals[valid].reset_index(drop=True)
        forecasts = forecasts[valid].reset_index(drop=True)

        errors = (actuals - forecasts).abs()
        total_actual = float(actuals.sum())
        total_error = float(errors.sum())

        if total_actual > 0:
            wape = total_error / total_actual * 100.0
        else:
            wape = float("nan")

        satuan_values = (
            item_history["Satuan"].dropna().astype(str).str.strip().tolist()
            if "Satuan" in item_history.columns
            else []
        )
        satuan = satuan_values[0] if satuan_values else ""

        # Baris detail backtest untuk contoh perhitungan.
        for idx, (actual, forecast, error) in enumerate(
            zip(actuals, forecasts, errors),
            start=1,
        ):
            rows.append(
                {
                    "Item": item_name,
                    "Satuan": satuan,
                    "Periode Backtest": idx,
                    "Actual": float(actual),
                    "Forecast": float(forecast),
                    "Error Absolut": float(error),
                    "WAPE (%)": round(wape, 2),
                    "Metode Contoh": method,
                    "Total Actual": total_actual,
                    "Total Error Absolut": total_error,
                }
            )

    if not rows:
        return None

    result = pd.DataFrame(rows)

    # =====================================================
    # STRUKTUR CONTOH UNTUK SHEET EXCEL
    # _write_dynamic_example() membutuhkan metadata item,
    # histori, dan detail backtest. Bangun dari item nyata
    # yang sama agar tidak ada data dummy.
    # =====================================================
    example_item = selected[0][0]
    example_history = selected[0][3].copy()
    example_satuan_values = (
        example_history["Satuan"].dropna().astype(str).str.strip().tolist()
        if "Satuan" in example_history.columns
        else []
    )
    example_satuan = example_satuan_values[0] if example_satuan_values else ""

    example_methods = {}
    example_values = pd.to_numeric(
        example_history["value"], errors="coerce"
    ).fillna(0.0).astype(float).tolist()

    for example_method in ("MA", "WMA", "XGBoost"):
        try:
            example_methods[example_method] = backtest_method_details(
                example_values,
                example_method,
            )
        except Exception:
            example_methods[example_method] = {
                "actual": [],
                "forecast": [],
                "wape": float("nan"),
            }

    example_forecast_row = pd.DataFrame()
    if forecast_df is not None:
        try:
            forecast_work = _safe_dataframe(forecast_df)
            if not forecast_work.empty and "Nama Barang" in forecast_work.columns:
                mask = (
                    forecast_work["Nama Barang"]
                    .astype(str)
                    .str.strip()
                    .str.casefold()
                    == str(example_item).strip().casefold()
                )
                example_forecast_row = forecast_work.loc[mask].head(1).copy()
        except Exception:
            example_forecast_row = pd.DataFrame()

    return {
        "detail": result,
        "stream": stream,
        "formula": "WAPE = Total Error Absolut ÷ Total Actual × 100%",
        "items": [r["Item"] for r in rows],
        "item": example_item,
        "satuan": example_satuan,
        "history": example_history,
        "methods": example_methods,
        "forecast_row": example_forecast_row,
    }



def _backtest_detail_table(example):
    """Bangun tabel actual-vs-backtest untuk satu item nyata."""
    if not example:
        return pd.DataFrame()

    history = example["history"]
    rows = []
    periods = history["_periode"].tolist()

    # MA/WMA dimulai dari observasi kedua; XGBoost dari observasi ketiga.
    for method in _METHODS:
        details = example["methods"].get(method, {})
        actual = list(details.get("actual", []) or [])
        forecasts = list(details.get("forecast", []) or [])
        n = min(len(actual), len(forecasts))
        if n <= 0:
            continue

        offset = len(periods) - n
        # Untuk metode yang mengembalikan seluruh pasangan mulai dari i=1/i=2,
        # gunakan periode aktual terakhir sebanyak n. Ini identik dengan urutan
        # actual_list yang dibentuk forecasting.py.
        if method in {"MA", "WMA"}:
            offset = 1
        elif method == "XGBoost":
            offset = 2

        for idx in range(n):
            p = periods[offset + idx] if offset + idx < len(periods) else pd.NaT
            rows.append(
                {
                    "Periode": p,
                    "Actual OUT": float(actual[idx]),
                    f"Forecast Backtest {method}": float(forecasts[idx]),
                    f"Error Absolut {method}": abs(float(actual[idx]) - float(forecasts[idx])),
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Gabungkan berdasarkan periode agar satu baris berisi ketiga metode.
    result = result.groupby("Periode", as_index=False).first().sort_values("Periode")
    for method in _METHODS:
        forecast_col = f"Forecast Backtest {method}"
        error_col = f"Error Absolut {method}"
        if forecast_col not in result.columns:
            result[forecast_col] = pd.NA
        if error_col not in result.columns:
            result[error_col] = pd.NA
    return result.reset_index(drop=True)


def _write_title(worksheet, row, text, end_col=5, fill_color="D9EAF7"):
    worksheet.cell(row, 1, text)
    worksheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=end_col)
    cell = worksheet.cell(row, 1)
    cell.font = Font(bold=True, size=13)
    cell.fill = PatternFill("solid", fgColor=fill_color)
    cell.alignment = Alignment(horizontal="left", vertical="center")
    worksheet.row_dimensions[row].height = 24


def _write_table(worksheet, start_row, dataframe, number_columns=None, header_fill="D9EAF7"):
    if dataframe is None:
        dataframe = pd.DataFrame()
    df = dataframe.copy()
    headers = list(df.columns)
    if not headers:
        return start_row

    for col_idx, header in enumerate(headers, start=1):
        cell = worksheet.cell(start_row, col_idx, header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor=header_fill)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(
            left=Side(style="thin", color="B7B7B7"),
            right=Side(style="thin", color="B7B7B7"),
            top=Side(style="thin", color="B7B7B7"),
            bottom=Side(style="thin", color="B7B7B7"),
        )

    for r_offset, values in enumerate(df.itertuples(index=False, name=None), start=1):
        for c_idx, value in enumerate(values, start=1):
            # openpyxl tidak dapat menulis pandas.NA secara langsung.
            # Normalisasi missing value menjadi None agar export Excel
            # tetap berjalan pada tabel yang memiliki kolom kosong.
            if value is pd.NA or pd.isna(value):
                value = None
            cell = worksheet.cell(start_row + r_offset, c_idx, value)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            cell.border = Border(
                left=Side(style="thin", color="D9D9D9"),
                right=Side(style="thin", color="D9D9D9"),
                top=Side(style="thin", color="D9D9D9"),
                bottom=Side(style="thin", color="D9D9D9"),
            )
            if number_columns and c_idx in number_columns and value is not None:
                cell.number_format = "#,##0.00"

    return start_row + len(df)


def _write_performance_section(worksheet, stream, stream_summary, start_row, fill_color):
    """Performance Forecasting seperti dashboard, tanpa Best Method."""
    _write_title(worksheet, start_row, f"Performance Forecasting - {stream}", end_col=5, fill_color=fill_color)
    table = pd.DataFrame([
        {
            "Metode": "MA",
            "WAPE (%)": _summary_value(stream_summary, "wape_ma"),
            "Accuracy (%)": _summary_value(stream_summary, "accuracy_ma"),
            "Total Actual Backtest": _summary_value(stream_summary, "total_actual_ma"),
            "Total Error Absolut": _summary_value(stream_summary, "total_error_ma"),
        },
        {
            "Metode": "WMA",
            "WAPE (%)": _summary_value(stream_summary, "wape_wma"),
            "Accuracy (%)": _summary_value(stream_summary, "accuracy_wma"),
            "Total Actual Backtest": _summary_value(stream_summary, "total_actual_wma"),
            "Total Error Absolut": _summary_value(stream_summary, "total_error_wma"),
        },
        {
            "Metode": "XGBoost",
            "WAPE (%)": _summary_value(stream_summary, "wape_xgboost"),
            "Accuracy (%)": _summary_value(stream_summary, "accuracy_xgboost"),
            "Total Actual Backtest": _summary_value(stream_summary, "total_actual_xgboost"),
            "Total Error Absolut": _summary_value(stream_summary, "total_error_xgboost"),
        },
    ])
    return _write_table(worksheet, start_row + 1, table, number_columns={2, 3, 4, 5}, header_fill=fill_color) + 2


def _write_wape_explanation(worksheet, start_row):
    _write_title(worksheet, start_row, "Dari Mana WAPE Berasal?", end_col=5, fill_color="FFF2CC")
    rows = [
        ["Tahap", "Penjelasan"],
        ["1. Histori Actual", "Sistem mengambil nilai OUT aktual pada histori sebelum periode forecast."],
        ["2. Backtesting", "Untuk setiap metode, sistem melakukan walk-forward: sebagian histori dipakai sebagai training lalu bulan berikutnya diprediksi."],
        ["3. Forecast Backtest", "Prediksi backtest dibandingkan dengan Actual OUT pada bulan yang sama. Forecast masa depan/recursive tidak dipakai untuk WAPE."],
        ["4. Error Absolut", "Error Absolut = |Actual OUT - Forecast Backtest|."],
        ["5. WAPE", "WAPE = Σ|Actual OUT - Forecast Backtest| / Σ|Actual OUT| × 100%."],
        ["6. Accuracy", "Accuracy = max(0, 100 - WAPE)."],
        ["Catatan", "WAPE dihitung terpisah untuk MA, WMA, dan XGBoost. Tidak ada Best Method."],
    ]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    end_row = _write_table(worksheet, start_row + 1, df, header_fill="FFF2CC")
    return end_row + 2


def _write_dynamic_example(worksheet, example, start_row):
    if not example:
        _write_title(worksheet, start_row, "Contoh Perhitungan WAPE dari Data OUT", end_col=5, fill_color="E2F0D9")
        worksheet.cell(start_row + 1, 1, "Contoh dinamis belum tersedia karena data OUT asli tidak tersedia pada saat export.")
        worksheet.merge_cells(start_row=start_row + 1, start_column=1, end_row=start_row + 1, end_column=5)
        return start_row + 3

    item = example["item"]
    stream = example["stream"]
    satuan = example["satuan"]
    _write_title(
        worksheet,
        start_row,
        f"Contoh Dinamis WAPE - {item} ({stream})",
        end_col=8,
        fill_color="E2F0D9",
    )
    worksheet.cell(start_row + 1, 1, "Sumber")
    worksheet.cell(start_row + 1, 2, "Data OUT yang sedang dipakai untuk forecasting")
    worksheet.cell(start_row + 2, 1, "Nama Barang")
    worksheet.cell(start_row + 2, 2, item)
    worksheet.cell(start_row + 3, 1, "Satuan")
    worksheet.cell(start_row + 3, 2, satuan)

    history = example["history"].copy()
    hist_table = pd.DataFrame({
        "Periode": history["_periode"],
        "Actual OUT": pd.to_numeric(history["value"], errors="coerce"),
    })
    row = start_row + 5
    worksheet.cell(row, 1, "Histori Actual yang Digunakan")
    worksheet.cell(row, 1).font = Font(bold=True)
    row = _write_table(worksheet, row + 1, hist_table, number_columns={2}, header_fill="E2F0D9") + 2

    # Projection calculations: use same MA/WMA functions and the final system output.
    try:
        from forecasting import moving_average, weighted_moving_average
        values = history["value"].to_numpy(dtype=float)
        ma_calc = moving_average(values, min(3, len(values))) if len(values) else float("nan")
        wma_calc = weighted_moving_average(values, min(3, len(values))) if len(values) else float("nan")
    except Exception:
        ma_calc = wma_calc = float("nan")

    forecast_row = example.get("forecast_row", pd.DataFrame())
    def get_forecast(column):
        if forecast_row is not None and not forecast_row.empty and column in forecast_row.columns:
            try:
                return float(forecast_row.iloc[0][column])
            except Exception:
                pass
        return float("nan")

    projection = pd.DataFrame([
        {
            "Metode": "MA",
            "Dasar Perhitungan": f"Rata-rata {min(3, len(values))} histori terakhir",
            "Perhitungan": f"{_format_value(ma_calc)}",
            "Forecast Sistem": get_forecast("Forecast MA"),
        },
        {
            "Metode": "WMA",
            "Dasar Perhitungan": "Weighted Moving Average; bobot makin besar untuk periode terbaru",
            "Perhitungan": f"{_format_value(wma_calc)}",
            "Forecast Sistem": get_forecast("Forecast WMA"),
        },
        {
            "Metode": "XGBoost",
            "Dasar Perhitungan": "Model XGBoost dilatih dari histori item; prediksi final berasal dari proses forecasting sistem",
            "Perhitungan": "Tidak diringkas menjadi satu rumus aritmetika",
            "Forecast Sistem": get_forecast("Forecast XGBoost"),
        },
    ])
    worksheet.cell(row, 1, "Projection / Forecast")
    worksheet.cell(row, 1).font = Font(bold=True)
    row = _write_table(worksheet, row + 1, projection, number_columns={4}, header_fill="E2F0D9") + 2

    # Actual vs backtest detail.
    detail = _backtest_detail_table(example)
    if detail.empty:
        worksheet.cell(row, 1, "Backtest detail tidak tersedia untuk item ini.")
        worksheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        return row + 3

    # Reformat to one table, with blanks where a method has no backtest for a period.
    detail = detail.rename(columns={
        "Forecast Backtest MA": "Forecast MA (Backtest)",
        "Error Absolut MA": "Error MA",
        "Forecast Backtest WMA": "Forecast WMA (Backtest)",
        "Error Absolut WMA": "Error WMA",
        "Forecast Backtest XGBoost": "Forecast XGBoost (Backtest)",
        "Error Absolut XGBoost": "Error XGBoost",
    })
    cols = [
        "Periode", "Actual OUT",
        "Forecast MA (Backtest)", "Error MA",
        "Forecast WMA (Backtest)", "Error WMA",
        "Forecast XGBoost (Backtest)", "Error XGBoost",
    ]
    for c in cols:
        if c not in detail.columns:
            detail[c] = float("nan")
    detail = detail[cols]
    detail["Periode"] = detail["Periode"].map(lambda x: x.strftime("%B %Y") if hasattr(x, "strftime") else str(x))

    worksheet.cell(row, 1, "Actual vs Forecast Backtest")
    worksheet.cell(row, 1).font = Font(bold=True)
    row = _write_table(worksheet, row + 1, detail, number_columns={2, 3, 4, 5, 6, 7, 8}, header_fill="E2F0D9") + 2

    # Per-method WAPE calculation from the exact detail used above.
    formula_rows = []
    for method in _METHODS:
        details = example["methods"].get(method, {})
        actual = list(details.get("actual", []) or [])
        forecasts = list(details.get("forecast", []) or [])
        total_actual = sum(abs(float(x)) for x in actual)
        total_error = sum(abs(float(a) - float(f)) for a, f in zip(actual, forecasts))
        wape = details.get("wape", float("nan"))
        formula_rows.append({
            "Metode": method,
            "Σ Actual Absolut": total_actual,
            "Σ Error Absolut": total_error,
            "WAPE Hasil Backtest (%)": wape,
            "Rumus": "Σ|Actual - Forecast Backtest| / Σ|Actual| × 100%",
        })
    formula_df = pd.DataFrame(formula_rows)
    worksheet.cell(row, 1, "Perhitungan WAPE Item Ini")
    worksheet.cell(row, 1).font = Font(bold=True)
    row = _write_table(worksheet, row + 1, formula_df, number_columns={2, 3, 4}, header_fill="E2F0D9") + 2
    worksheet.cell(row, 1, "Catatan: angka pada bagian ini dihitung ulang dari histori item nyata dan backtest method yang sama dengan forecasting.py.")
    worksheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    return row + 3


def _build_comparing_dataframe(
    dataframe,
    stream_summary=None,
):
    """
    Membuat recap per item dengan urutan kolom:
    Forecast MA -> WAPE MA -> Forecast WMA -> WAPE WMA ->
    Forecast XGBoost -> WAPE XGBoost -> Best Method ->
    Forecast Terpilih -> WAPE Terbaik.

    Metode yang tidak tersedia dibiarkan kosong.
    BBB dan BBT diproses terpisah.
    """
    df = _safe_dataframe(dataframe)

    if df.empty:
        return pd.DataFrame(columns=_COMPARING_COLUMNS)

    result = pd.DataFrame()
    result["Nama Barang"] = (
        df["Nama Barang"].astype(str)
        if "Nama Barang" in df.columns
        else ""
    )
    result["Satuan"] = (
        df["Satuan"].astype(str)
        if "Satuan" in df.columns
        else ""
    )

    def copy_column(target, candidates):
        for candidate in candidates:
            if candidate in df.columns:
                result[target] = df[candidate]
                return
        result[target] = pd.NA

    copy_column("Forecast MA", ["Forecast MA"])
    copy_column("WAPE MA (%)", ["WAPE MA (%)", "WAPE MA"])
    copy_column("Forecast WMA", ["Forecast WMA"])
    copy_column("WAPE WMA (%)", ["WAPE WMA (%)", "WAPE WMA"])
    copy_column("Forecast XGBoost", ["Forecast XGBoost"])
    copy_column(
        "WAPE XGBoost (%)",
        ["WAPE XGBoost (%)", "WAPE XGBoost"],
    )

    # Ambil item-level best method bila tersedia dari forecasting.py.
    item_summaries = {}
    if isinstance(stream_summary, dict):
        raw_items = stream_summary.get("items", {})
        if isinstance(raw_items, dict):
            item_summaries = raw_items
        elif isinstance(raw_items, (list, tuple)):
            # forecasting.py menyimpan item detail sebagai list of dict.
            # Ubah menjadi mapping berdasarkan Nama Barang agar lookup
            # meta.get(...) di bawah tetap aman.
            item_summaries = {}
            for item_meta in raw_items:
                if not isinstance(item_meta, dict):
                    continue
                item_name = str(
                    item_meta.get("Nama Barang", "")
                ).strip()
                if item_name:
                    item_summaries[item_name] = item_meta

    best_methods = []
    selected_forecasts = []
    best_wapes = []

    for _, row in result.iterrows():
        name = str(row["Nama Barang"]).strip()
        meta = item_summaries.get(name, {})
        if not isinstance(meta, dict):
            meta = {}

        best_method = meta.get("best_method", "")
        best_forecast = meta.get("best_forecast", meta.get("forecast", pd.NA))
        best_wape = meta.get("best_wape", pd.NA)

        if not best_method:
            # Backward-compatible fallback: pilih WAPE terendah yang ada.
            options = [
                ("MA", row.get("WAPE MA (%)"), row.get("Forecast MA")),
                ("WMA", row.get("WAPE WMA (%)"), row.get("Forecast WMA")),
                ("XGBoost", row.get("WAPE XGBoost (%)"), row.get("Forecast XGBoost")),
            ]
            valid_options = []
            for method_name, wape_value, forecast_value in options:
                try:
                    wape_num = float(wape_value)
                    if pd.notna(wape_num):
                        valid_options.append(
                            (wape_num, method_name, forecast_value)
                        )
                except Exception:
                    pass

            if valid_options:
                valid_options.sort(key=lambda x: x[0])
                best_wape, best_method, best_forecast = valid_options[0]
            else:
                best_method = ""
                best_forecast = pd.NA
                best_wape = pd.NA

        best_methods.append(best_method or "-")
        selected_forecasts.append(best_forecast)
        best_wapes.append(best_wape)

    result["Best Method"] = best_methods
    result["Forecast Terpilih"] = selected_forecasts
    result["WAPE Terbaik (%)"] = best_wapes

    return result[_COMPARING_COLUMNS]




# =========================================================
# OPENPYXL SAFE VALUE
# =========================================================

def _excel_safe_value(value):
    """
    Mengubah pandas.NA/NaN/NaT menjadi None sebelum ditulis
    langsung menggunakan openpyxl.

    Ini penting untuk sheet Comparing karena beberapa kolom
    memang boleh kosong bila metode tidak tersedia.
    """
    if value is None:
        return None

    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except Exception:
        pass

    return value

def _write_comparing_section(worksheet, dataframe, title, start_row, fill_color):
    worksheet.cell(start_row, 1, title)
    worksheet.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=5)
    worksheet.cell(start_row, 1).font = Font(bold=True, size=12)
    worksheet.cell(start_row, 1).fill = PatternFill("solid", fgColor=fill_color)
    header_row = start_row + 1
    for idx, column in enumerate(_COMPARING_COLUMNS, start=1):
        cell = worksheet.cell(header_row, idx, column)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor=fill_color)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    df = _safe_dataframe(dataframe)
    if df.empty:
        df = pd.DataFrame(columns=_COMPARING_COLUMNS)
    for r, values in enumerate(df[_COMPARING_COLUMNS].itertuples(index=False, name=None), start=header_row + 1):
        for c, value in enumerate(values, start=1):
            cell = worksheet.cell(r, c, _excel_safe_value(value))
            if c >= 3 and value is not None:
                cell.number_format = "#,##0.00"
    return header_row + max(1, len(df))


def _write_forecast_monthly_sheet(writer, summary, periode_forecast, history_df, history_months, forecast_bbb, forecast_bbt):
    """Dashboard-style Performance Forecasting + explanation WAPE."""
    ws = writer.book.create_sheet("Performance")
    ws.sheet_view.showGridLines = False

    ws["A1"] = "PERFORMANCE FORECASTING"
    ws.merge_cells("A1:H1")
    ws["A1"].font = Font(bold=True, size=15)
    ws["A1"].alignment = Alignment(horizontal="left")
    ws["A2"] = f"Periode Forecast: {periode_forecast or '-'}"
    ws.merge_cells("A2:H2")
    ws["A3"] = "MA, WMA, dan XGBoost dihitung dan ditampilkan secara independen. Tidak ada Best Method."
    ws.merge_cells("A3:H3")

    row = 5
    for stream, stream_df, fill in (
        ("BBB", forecast_bbb, "BDD7EE"),
        ("BBT", forecast_bbt, "C6E0B4"),
    ):
        stream_summary = _safe_summary_stream(summary, stream.lower())
        row = _write_performance_section(ws, stream, stream_summary, row, fill)

    # Penjelasan WAPE dan contoh Data OUT hanya satu kali,
    # diletakkan paling bawah setelah Performance BBB dan BBT.
    row = _write_wape_explanation(ws, row + 1)

    # Pilih satu item nyata dari Data OUT. Prioritas BBB, lalu BBT.
    # Tidak ada angka contoh/dummy. Semua angka berasal dari data dan
    # fungsi backtesting yang sama dengan forecasting.py.
    example = _prepare_dynamic_example(
        history_df,
        "BBB",
        periode_forecast,
        history_months=history_months,
        forecast_df=forecast_bbb,
    )
    if not example:
        example = _prepare_dynamic_example(
            history_df,
            "BBT",
            periode_forecast,
            history_months=history_months,
            forecast_df=forecast_bbt,
        )
    row = _write_dynamic_example(ws, example, row + 1)

    widths = {
        "A": 30, "B": 24, "C": 24, "D": 24,
        "E": 24, "F": 24, "G": 26, "H": 22,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A5"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    return ws


# =========================================================
# EXPORT EXCEL
# =========================================================

def export_forecast_excel(
    forecast_bbb=None,
    forecast_bbt=None,
    periode_forecast="",
    nama_user="",
    recursive_detail_bbb=None,
    recursive_detail_bbt=None,
    summary=None,
    history_df=None,
    history_months=None,
):
    """
    Membuat file Excel hasil forecasting.

    Struktur workbook:

    1. Performance
       - Performance Forecasting BBB dan BBT
       - WAPE, Accuracy, total actual backtest, total error per metode
       - penjelasan sumber WAPE
       - contoh dinamis 1 item dari Data OUT asli

    2. Detail BBB
    3. Detail BBT
    4. Comparing
       - BBB dan BBT dipisah
       - HANYA Nama Barang, Satuan, Forecast MA, Forecast WMA,
         Forecast XGBoost

    5-6. Recursive BBB/BBT bila diberikan.

    Parameter baru history_df dan history_months opsional agar pemanggilan
    lama tetap kompatibel. Bila history_df tidak tersedia, bagian contoh
    dinamis diberi keterangan bahwa data sumber tidak tersedia.
    """

    df_bbb = _format_forecast_dataframe(forecast_bbb)
    df_bbt = _format_forecast_dataframe(forecast_bbt)
    df_recursive_bbb = _format_recursive_dataframe(recursive_detail_bbb)
    df_recursive_bbt = _format_recursive_dataframe(recursive_detail_bbt)

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Forecast Bulanan dibuat manual agar tidak lagi menjadi tabel item forecast.
        _write_forecast_monthly_sheet(
            writer=writer,
            summary=summary,
            periode_forecast=periode_forecast,
            history_df=history_df,
            history_months=history_months,
            forecast_bbb=forecast_bbb,
            forecast_bbt=forecast_bbt,
        )

        # Detail tetap tersedia untuk audit item per item.
        _write_dataframe_sheet(
            writer,
            df_bbb,
            "Detail BBB",
            _empty_export_dataframe(),
        )
        _write_dataframe_sheet(
            writer,
            df_bbt,
            "Detail BBT",
            _empty_export_dataframe(),
        )

        # Histori yang diekspor dibatasi hanya pada histori yang digunakan
        # forecasting, lalu BBB dan BBT dipisahkan.
        _write_history_used_sheet(
            writer=writer,
            history_df=history_df,
            periode_forecast=periode_forecast,
            history_months=history_months,
            stream="BBB",
        )
        _write_history_used_sheet(
            writer=writer,
            history_df=history_df,
            periode_forecast=periode_forecast,
            history_months=history_months,
            stream="BBT",
        )

        # Comparing hanya berisi lima kolom yang diminta.
        ws_comparing = writer.book.create_sheet("Comparing")
        ws_comparing.sheet_view.showGridLines = False
        next_row = _write_comparing_section(
            ws_comparing,
            _build_comparing_dataframe(
                forecast_bbb,
                _safe_summary_stream(summary, "BBB"),
            ),
            "COMPARING - BBB",
            start_row=1,
            fill_color="BDD7EE",
        )
        _write_comparing_section(
            ws_comparing,
            _build_comparing_dataframe(
                forecast_bbt,
                _safe_summary_stream(summary, "BBT"),
            ),
            "COMPARING - BBT",
            start_row=next_row + 3,
            fill_color="C6E0B4",
        )
        for column, width in {
            "A": 30,
            "B": 14,
            "C": 18,
            "D": 14,
            "E": 18,
            "F": 14,
            "G": 20,
            "H": 14,
            "I": 18,
            "J": 18,
            "K": 16,
        }.items():
            ws_comparing.column_dimensions[column].width = width
        ws_comparing.freeze_panes = "A3"

        if not df_recursive_bbb.empty:
            df_recursive_bbb.to_excel(writer, sheet_name="Recursive BBB", index=False)
        if not df_recursive_bbt.empty:
            df_recursive_bbt.to_excel(writer, sheet_name="Recursive BBT", index=False)

        # Apply common styling to detail/recursive sheets only.
        for sheet_name in writer.book.sheetnames:
            ws = writer.book[sheet_name]
            if sheet_name not in {"Performance", "Comparing"}:
                _format_worksheet(ws)

        workbook = writer.book
        workbook.properties.title = "Demand Planning Forecast"
        workbook.properties.subject = "Forecast MA, WMA, XGBoost dan Performance Forecasting"
        workbook.properties.creator = _safe_text(nama_user) or "Demand Planner"
        workbook.properties.description = (
            "Forecast demand dengan MA, WMA, dan XGBoost. "
            "WAPE dihitung dari walk-forward backtesting."
        )
        workbook.properties.keywords = "Demand Planning, Forecast, BBB, BBT, MA, WMA, XGBoost, WAPE"

    output.seek(0)
    return output.getvalue()


# =========================================================
# DATA HISTORI YANG BENAR-BENAR DIGUNAKAN FORECASTING
# =========================================================

def _prepare_history_used_by_forecasting(
    history_df,
    periode_forecast,
    history_months=None,
):
    """
    Mengambil hanya histori yang masuk ke periode forecasting.

    Output tetap satu dataframe sumber, lalu dipisahkan menjadi BBB dan BBT
    ketika ditulis ke Excel.
    """
    df = _safe_dataframe(history_df)
    if df.empty:
        return pd.DataFrame()

    required = {"Bulan", "Nama Barang"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    forecast_date = _parse_period_for_export(periode_forecast)
    if forecast_date is None or pd.isna(forecast_date):
        return pd.DataFrame()

    work = df.copy()
    work["__period__"] = work["Bulan"].apply(
        lambda x: _parse_period_for_export(x, default_year=forecast_date.year)
    )
    work = work[work["__period__"].notna()].copy()

    if work.empty:
        return pd.DataFrame()

    # Gunakan jumlah histori yang sama dengan forecasting jika tersedia.
    if history_months is not None:
        try:
            n_months = int(history_months)
        except Exception:
            n_months = None
    else:
        n_months = None

    periods = sorted(
        work["__period__"].dropna().unique().tolist()
    )

    # Hanya bulan sebelum periode forecast.
    periods = [p for p in periods if p < forecast_date]

    if n_months and n_months > 0:
        periods = periods[-n_months:]

    if not periods:
        return pd.DataFrame()

    used = work[work["__period__"].isin(periods)].copy()
    used["Bulan"] = used["__period__"].dt.strftime("%B %Y")
    used = used.drop(columns=["__period__"], errors="ignore")

    # Agregasi mengikuti kebutuhan item-bulan forecasting.
    base_columns = ["Bulan", "Nama Barang"]
    if "Satuan" in used.columns:
        base_columns.append("Satuan")

    value_columns = [
        column
        for column in ["OUT BBB", "OUT BBT"]
        if column in used.columns
    ]

    if not value_columns:
        return pd.DataFrame()

    for column in value_columns:
        used[column] = pd.to_numeric(
            used[column], errors="coerce"
        ).fillna(0.0)

    group_columns = base_columns
    used = (
        used.groupby(group_columns, as_index=False)[value_columns]
        .sum()
    )

    # Urutkan bulan kronologis dan item.
    month_order = {
        month.strftime("%B %Y"): month
        for month in periods
    }
    used["__sort_month__"] = used["Bulan"].map(month_order)
    used = used.sort_values(
        ["__sort_month__", "Nama Barang"],
        kind="stable",
    ).drop(columns=["__sort_month__"])

    return used


def _write_history_used_sheet(
    writer,
    history_df,
    periode_forecast,
    history_months,
    stream,
):
    """
    Menulis:
      Data Histori BBB -> Bulan | Nama Barang | Satuan | OUT BBB
      Data Histori BBT -> Bulan | Nama Barang | Satuan | OUT BBT
    """
    used = _prepare_history_used_by_forecasting(
        history_df=history_df,
        periode_forecast=periode_forecast,
        history_months=history_months,
    )

    if stream == "BBB":
        value_column = "OUT BBB"
        sheet_name = "Data Histori BBB"
    else:
        value_column = "OUT BBT"
        sheet_name = "Data Histori BBT"

    columns = ["Bulan", "Nama Barang", "Satuan", value_column]

    if used.empty or value_column not in used.columns:
        pd.DataFrame(columns=columns).to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
        )
        return

    export_df = used.copy()
    if "Satuan" not in export_df.columns:
        export_df["Satuan"] = ""

    export_df = export_df[columns]
    export_df.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
    )



# =========================================================
# NAMA FILE
# =========================================================

def generate_export_filename(
    periode_forecast="",
    nama_user=""
):
    """
    Membuat nama file Excel otomatis.

    Contoh:
    Forecast_September_2026_Budi.xlsx
    Forecast_September_2026_Budi_Santoso.xlsx

    ``nama_user`` dibuat opsional agar pemanggilan lama tetap kompatibel.
    """

    periode = _safe_text(
        periode_forecast
    )

    if not periode:

        periode = "Forecast"

    # -----------------------------------------------------
    # Karakter invalid Windows
    # -----------------------------------------------------

    invalid_characters = [
        "\\",
        "/",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
    ]

    for char in invalid_characters:

        periode = (
            periode.replace(
                char,
                "-",
            )
        )

    # -----------------------------------------------------
    # Rapikan spasi
    # -----------------------------------------------------

    periode = " ".join(
        periode.split()
    )

    # -----------------------------------------------------
    # Nama file
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Nama user (opsional)
    # -----------------------------------------------------

    user = _safe_text(
        nama_user
    )

    if user:

        for char in invalid_characters:

            user = (
                user.replace(
                    char,
                    "-",
                )
            )

        user = " ".join(
            user.split()
        )

    # -----------------------------------------------------
    # Nama file final
    # -----------------------------------------------------

    if user:

        return (
            "Forecast_"
            f"{periode.replace(' ', '_')}_"
            f"{user.replace(' ', '_')}.xlsx"
        )

    return (
        "Forecast_"
        f"{periode.replace(' ', '_')}.xlsx"
    )


# =========================================================
# TEST EXPORT
# =========================================================

if __name__ == "__main__":

    sample_bbb = pd.DataFrame(
        [
            {
                "Nama Barang": "Contoh Barang A",
                "Satuan": "PCS",
                "Histori": 8,
                "Forecast MA": 125.50,
                "WAPE MA": 12.35,
                "Accuracy MA": 87.65,
                "Forecast WMA": 128.25,
                "WAPE WMA": 10.75,
                "Accuracy WMA": 89.25,
                "Forecast XGBoost": 126.80,
                "WAPE XGBoost": 9.85,
                "Accuracy XGBoost": 90.15,
            },
        ]
    )

    sample_bbt = pd.DataFrame(
        [
            {
                "Nama Barang": "Contoh Barang B",
                "Satuan": "BOX",
                "Histori": 8,
                "Forecast MA": 97.50,
                "WAPE MA": 10.25,
                "Accuracy MA": 89.75,
                "Forecast WMA": 98.25,
                "WAPE WMA": 8.75,
                "Accuracy WMA": 91.25,
                "Forecast XGBoost": 99.10,
                "WAPE XGBoost": 8.10,
                "Accuracy XGBoost": 91.90,
            },
        ]
    )

    sample_recursive_bbb = pd.DataFrame(
        [
            {
                "Nama Barang": "Contoh Barang A",
                "Periode": "2026-08",
                "Nilai": 100,
                "Sumber": "actual",
                "Method": "Actual",
            },
            {
                "Nama Barang": "Contoh Barang A",
                "Periode": "2026-09",
                "Nilai": 110,
                "Sumber": "forecast",
                "Method": "MA3",
            },
            {
                "Nama Barang": "Contoh Barang A",
                "Periode": "2026-10",
                "Nilai": 120,
                "Sumber": "forecast",
                "Method": "MA3",
            },
        ]
    )

    sample_recursive_bbt = pd.DataFrame(
        [
            {
                "Nama Barang": "Contoh Barang B",
                "Periode": "2026-08",
                "Nilai": 90,
                "Sumber": "actual",
                "Method": "Actual",
            },
            {
                "Nama Barang": "Contoh Barang B",
                "Periode": "2026-09",
                "Nilai": 95,
                "Sumber": "forecast",
                "Method": "WMA3",
            },
        ]
    )

    excel_bytes = export_forecast_excel(
        forecast_bbb=sample_bbb,
        forecast_bbt=sample_bbt,
        periode_forecast="October 2026",
        nama_user="Demand Planner",
        recursive_detail_bbb=sample_recursive_bbb,
        recursive_detail_bbt=sample_recursive_bbt,
    )

    filename = generate_export_filename(
        "October 2026"
    )

    output_path = "export_test.xlsx"

    with open(
        output_path,
        "wb",
    ) as file:

        file.write(
            excel_bytes
        )

    print(
        f"Export berhasil: {output_path}"
    )

    print(
        f"Nama file: {filename}"
    )
