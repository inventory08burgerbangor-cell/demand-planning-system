import pandas as pd
from io import BytesIO

from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter


# =========================================================
# CONSTANT
# =========================================================

FORECAST_COLUMNS = [
    "Nama Barang",
    "Satuan",
    "Histori",
    "Best Method",
    "WAPE",
    "Forecast",
]


EXPORT_COLUMNS = [
    "Nama Barang",
    "Satuan",
    "Jumlah Histori",
    "Metode Terbaik",
    "WAPE (%)",
    "Forecast OUT",
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

    # -----------------------------------------------------
    # Kolom utama
    # -----------------------------------------------------

    preferred_columns = [
        "Nama Barang",
        "Satuan",
        "Histori",
        "Best Method",
        "WAPE",
        "Forecast",
    ]

    result = _ordered_columns(
        result,
        preferred_columns,
    )

    # -----------------------------------------------------
    # Rename
    # -----------------------------------------------------

    rename_map = {

        "Histori":
            "Jumlah Histori",

        "Best Method":
            "Metode Terbaik",

        "WAPE":
            "WAPE (%)",

        "Forecast":
            "Forecast OUT",

    }

    result = result.rename(
        columns=rename_map
    )

    # -----------------------------------------------------
    # Numeric Forecast
    # -----------------------------------------------------

    if (
        "Forecast OUT"
        in result.columns
    ):

        result[
            "Forecast OUT"
        ] = _numeric_series(
            result[
                "Forecast OUT"
            ],
            decimals=2,
        )

    # -----------------------------------------------------
    # Numeric WAPE
    # -----------------------------------------------------

    if (
        "WAPE (%)"
        in result.columns
    ):

        result[
            "WAPE (%)"
        ] = _numeric_series(
            result[
                "WAPE (%)"
            ],
            decimals=2,
        )

    # -----------------------------------------------------
    # Numeric Histori
    # -----------------------------------------------------

    if (
        "Jumlah Histori"
        in result.columns
    ):

        history_numeric = pd.to_numeric(
            result[
                "Jumlah Histori"
            ],
            errors="coerce",
        )

        result[
            "Jumlah Histori"
        ] = history_numeric.fillna(
            0
        ).astype(
            int
        )

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
    # WAPE
    # -----------------------------------------------------

    if "WAPE (%)" in headers:

        column_number = (
            headers[
                "WAPE (%)"
            ]
        )

        for row in worksheet.iter_rows(
            min_row=2,
            min_col=column_number,
            max_col=column_number,
        ):

            row[0].number_format = (
                "0.00"
            )

    # -----------------------------------------------------
    # Forecast
    # -----------------------------------------------------

    if "Forecast OUT" in headers:

        column_number = (
            headers[
                "Forecast OUT"
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
    # Jumlah Histori
    # -----------------------------------------------------

    if (
        "Jumlah Histori"
        in headers
    ):

        column_number = (
            headers[
                "Jumlah Histori"
            ]
        )

        for row in worksheet.iter_rows(
            min_row=2,
            min_col=column_number,
            max_col=column_number,
        ):

            row[0].number_format = (
                "0"
            )

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
# EXPORT EXCEL
# =========================================================

def export_forecast_excel(
    forecast_bbb=None,
    forecast_bbt=None,
    periode_forecast="",
    nama_user="",
    recursive_detail_bbb=None,
    recursive_detail_bbt=None,
):
    """
    Membuat file Excel hasil forecasting.

    Sheet utama:

    1. Forecast Bulanan
    2. Detail BBB
    3. Detail BBT

    Sheet tambahan bila detail recursive diberikan:

    4. Recursive BBB
    5. Recursive BBT

    Parameter recursive_detail_bbb dan recursive_detail_bbt
    bersifat opsional agar pemanggilan lama tetap kompatibel.

    Return:
        bytes Excel yang bisa digunakan
        oleh st.download_button().
    """

    # -----------------------------------------------------
    # Prepare BBB
    # -----------------------------------------------------

    df_bbb = (
        _format_forecast_dataframe(
            forecast_bbb
        )
    )

    # -----------------------------------------------------
    # Prepare BBT
    # -----------------------------------------------------

    df_bbt = (
        _format_forecast_dataframe(
            forecast_bbt
        )
    )

    # -----------------------------------------------------
    # Prepare recursive detail
    # -----------------------------------------------------

    df_recursive_bbb = (
        _format_recursive_dataframe(
            recursive_detail_bbb
        )
    )

    df_recursive_bbt = (
        _format_recursive_dataframe(
            recursive_detail_bbt
        )
    )

    # -----------------------------------------------------
    # Output memory
    # -----------------------------------------------------

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        # =================================================
        # SHEET 1
        # FORECAST BULANAN
        # =================================================

        rows = []

        # -------------------------------------------------
        # BBB
        # -------------------------------------------------

        if not df_bbb.empty:

            temp_bbb = (
                df_bbb.copy()
            )

            temp_bbb.insert(
                0,
                "Outlet",
                "BBB",
            )

            rows.append(
                temp_bbb
            )

        # -------------------------------------------------
        # BBT
        # -------------------------------------------------

        if not df_bbt.empty:

            temp_bbt = (
                df_bbt.copy()
            )

            temp_bbt.insert(
                0,
                "Outlet",
                "BBT",
            )

            rows.append(
                temp_bbt
            )

        # -------------------------------------------------
        # Gabungkan
        # -------------------------------------------------

        if rows:

            df_summary = (
                pd.concat(
                    rows,
                    ignore_index=True,
                )
            )

        else:

            df_summary = pd.DataFrame(
                columns=[
                    "Outlet",
                    "Nama Barang",
                    "Satuan",
                    "Jumlah Histori",
                    "Metode Terbaik",
                    "WAPE (%)",
                    "Forecast OUT",
                ]
            )

        df_summary.to_excel(
            writer,
            sheet_name="Forecast Bulanan",
            index=False,
        )

        # =================================================
        # SHEET 2
        # DETAIL BBB
        # =================================================

        _write_dataframe_sheet(
            writer,
            df_bbb,
            "Detail BBB",
            _empty_export_dataframe(),
        )

        # =================================================
        # SHEET 3
        # DETAIL BBT
        # =================================================

        _write_dataframe_sheet(
            writer,
            df_bbt,
            "Detail BBT",
            _empty_export_dataframe(),
        )

        # =================================================
        # SHEET 4
        # RECURSIVE BBB
        # =================================================

        if not df_recursive_bbb.empty:

            df_recursive_bbb.to_excel(
                writer,
                sheet_name="Recursive BBB",
                index=False,
            )

        # =================================================
        # SHEET 5
        # RECURSIVE BBT
        # =================================================

        if not df_recursive_bbt.empty:

            df_recursive_bbt.to_excel(
                writer,
                sheet_name="Recursive BBT",
                index=False,
            )

        # =================================================
        # FORMAT SEMUA SHEET
        # =================================================

        workbook = writer.book

        for worksheet in (
            workbook.worksheets
        ):

            _format_worksheet(
                worksheet
            )

        # =================================================
        # WORKBOOK METADATA
        # =================================================

        periode_text = _safe_text(
            periode_forecast
        )

        user_text = _safe_text(
            nama_user
        )

        workbook.properties.title = (
            "Demand Planning Forecast"
        )

        workbook.properties.subject = (
            f"Forecast {periode_text}"
            if periode_text
            else "Demand Forecast"
        )

        workbook.properties.creator = (
            user_text
            if user_text
            else "Demand Planner"
        )

        workbook.properties.description = (
            "Forecast demand "
            "Main Warehouse Batu Ceper"
        )

        workbook.properties.keywords = (
            "Demand Planning, "
            "Forecast, BBB, BBT, "
            "Recursive Forecast"
        )

    # -----------------------------------------------------
    # Reset pointer
    # -----------------------------------------------------

    output.seek(0)

    return output.getvalue()


# =========================================================
# NAMA FILE
# =========================================================

def generate_export_filename(
    periode_forecast=""
):
    """
    Membuat nama file Excel otomatis.

    Contoh:
    Demand_Planning_September_2026.xlsx
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

    return (
        "Demand_Planning_"
        f"{periode}.xlsx"
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
                "Best Method": "MA3",
                "WAPE": 12.35,
                "Forecast": 125.50,
            },
        ]
    )

    sample_bbt = pd.DataFrame(
        [
            {
                "Nama Barang": "Contoh Barang B",
                "Satuan": "BOX",
                "Histori": 8,
                "Best Method": "WMA3",
                "WAPE": 8.75,
                "Forecast": 98.25,
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
