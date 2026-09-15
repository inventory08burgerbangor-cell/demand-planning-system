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


# =========================================================
# FORMAT FORECAST DATAFRAME
# =========================================================

def _format_forecast_dataframe(
    df
):
    """
    Menyiapkan DataFrame forecast
    agar rapi ketika masuk Excel.
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

    existing_columns = [
        col
        for col in preferred_columns
        if col in result.columns
    ]

    other_columns = [
        col
        for col in result.columns
        if col not in existing_columns
    ]

    result = result[
        existing_columns
        +
        other_columns
    ]

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
        ] = pd.to_numeric(
            result[
                "Forecast OUT"
            ],
            errors="coerce",
        ).round(2)

    # -----------------------------------------------------
    # Numeric WAPE
    # -----------------------------------------------------

    if (
        "WAPE (%)"
        in result.columns
    ):

        result[
            "WAPE (%)"
        ] = pd.to_numeric(
            result[
                "WAPE (%)"
            ],
            errors="coerce",
        ).round(2)

    # -----------------------------------------------------
    # Numeric Histori
    # -----------------------------------------------------

    if (
        "Jumlah Histori"
        in result.columns
    ):

        result[
            "Jumlah Histori"
        ] = pd.to_numeric(
            result[
                "Jumlah Histori"
            ],
            errors="coerce",
        ).fillna(
            0
        ).astype(
            int
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
    # Tinggi header
    # -----------------------------------------------------

    worksheet.row_dimensions[
        1
    ].height = 24


# =========================================================
# EXPORT EXCEL
# =========================================================

def export_forecast_excel(
    forecast_bbb=None,
    forecast_bbt=None,
    periode_forecast="",
    nama_user="",
):
    """
    Membuat file Excel hasil forecasting.

    Sheet:

    1. Forecast Bulanan
    2. Detail BBB
    3. Detail BBT

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

        if df_bbb.empty:

            (
                _empty_export_dataframe()
                .to_excel(
                    writer,
                    sheet_name="Detail BBB",
                    index=False,
                )
            )

        else:

            df_bbb.to_excel(
                writer,
                sheet_name="Detail BBB",
                index=False,
            )

        # =================================================
        # SHEET 3
        # DETAIL BBT
        # =================================================

        if df_bbt.empty:

            (
                _empty_export_dataframe()
                .to_excel(
                    writer,
                    sheet_name="Detail BBT",
                    index=False,
                )
            )

        else:

            df_bbt.to_excel(
                writer,
                sheet_name="Detail BBT",
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

        workbook.properties.title = (
            "Demand Planning Forecast"
        )

        workbook.properties.subject = (
            f"Forecast {periode_forecast}"
        )

        workbook.properties.creator = (
            nama_user
            if nama_user
            else "Demand Planner"
        )

        workbook.properties.description = (
            "Forecast demand "
            "Main Warehouse Batu Ceper"
        )

        workbook.properties.keywords = (
            "Demand Planning, "
            "Forecast, BBB, BBT"
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

    periode = str(
        periode_forecast
    ).strip()

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