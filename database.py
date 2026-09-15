import sqlite3
import json
from datetime import datetime
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

DB_FILE = str(Path(__file__).resolve().parent / "demand_planning.db")


# =========================================================
# CONNECTION
# =========================================================

def get_connection():

    conn = sqlite3.connect(
        DB_FILE,
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    # -----------------------------------------------------
    # SQLite lebih aman untuk aplikasi Streamlit yang
    # dapat melakukan beberapa operasi database berdekatan.
    # -----------------------------------------------------

    try:

        conn.execute(
            "PRAGMA busy_timeout = 30000"
        )

    except Exception:

        pass

    return conn


# =========================================================
# HISTORY MONTHS HELPER
# =========================================================
#
# forecasting.py menggunakan:
#
#     None
#     ""
#     "all"
#     "semua"
#     0
#
# sebagai penanda "gunakan semua histori".
#
# Database TIDAK menyimpan sentinel tersebut.
# Database menyimpan JUMLAH HISTORI AKTUAL yang dipakai,
# misalnya 8, 10, 12, dst.
#
# Dengan begitu data history lama tetap kompatibel dengan
# main.py dan forecasting.py.
# =========================================================

def normalize_history_months(
    history_months,
    default=3,
    allow_all=False,
):

    # -----------------------------------------------------
    # Nilai kosong
    # -----------------------------------------------------

    if history_months is None:

        if allow_all:

            return None

        return int(
            default
        )

    # -----------------------------------------------------
    # String
    # -----------------------------------------------------

    if isinstance(
        history_months,
        str,
    ):

        value = (
            history_months
            .strip()
            .lower()
        )

        # -------------------------------------------------
        # Mode semua histori
        # -------------------------------------------------

        if value in (
            "",
            "all",
            "semua",
            "semua histori",
            "all history",
        ):

            if allow_all:

                return None

            return int(
                default
            )

        # -------------------------------------------------
        # Konversi angka
        # -------------------------------------------------

        try:

            history_months = int(
                float(value)
            )

        except Exception:

            history_months = int(
                default
            )

    # -----------------------------------------------------
    # Numeric
    # -----------------------------------------------------

    try:

        history_months = int(
            history_months
        )

    except Exception:

        history_months = int(
            default
        )

    # -----------------------------------------------------
    # Nilai <= 0
    # -----------------------------------------------------

    if history_months <= 0:

        if allow_all:

            return None

        history_months = int(
            default
        )

    # -----------------------------------------------------
    # Minimal 1 bulan
    # -----------------------------------------------------

    if history_months < 1:

        history_months = 1

    return int(
        history_months
    )


# =========================================================
# INIT DATABASE
# =========================================================

def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # -------------------------------------------------
        # CREATE TABLE
        # -------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_history (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                nama_user TEXT NOT NULL,

                periode_forecast TEXT NOT NULL,

                history_months INTEGER DEFAULT 3,

                created_at TEXT NOT NULL,

                status TEXT DEFAULT 'Saved',

                forecast_bbb TEXT,

                forecast_bbt TEXT,

                summary TEXT

            )
            """
        )

        # -------------------------------------------------
        # MIGRATION DATABASE LAMA
        # -------------------------------------------------

        cursor.execute(
            """
            PRAGMA table_info(
                forecast_history
            )
            """
        )

        existing_columns = {
            row["name"]
            for row in cursor.fetchall()
        }

        # -------------------------------------------------
        # Kolom yang wajib tersedia
        # -------------------------------------------------

        required_columns = {

            "history_months":
                "INTEGER DEFAULT 3",

            "forecast_bbb":
                "TEXT",

            "forecast_bbt":
                "TEXT",

            "summary":
                "TEXT",

        }

        # -------------------------------------------------
        # Tambahkan kolom jika belum ada
        # -------------------------------------------------

        for (
            column_name,
            column_definition,
        ) in required_columns.items():

            if (
                column_name
                not in existing_columns
            ):

                cursor.execute(
                    f"""
                    ALTER TABLE forecast_history
                    ADD COLUMN
                    {column_name}
                    {column_definition}
                    """
                )

        # -------------------------------------------------
        # Normalisasi data history_months lama.
        #
        # Database tetap menyimpan angka positif.
        # -------------------------------------------------

        cursor.execute(
            """
            UPDATE forecast_history
            SET history_months = 3
            WHERE history_months IS NULL
               OR history_months <= 0
            """
        )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# =========================================================
# SERIALIZE DATAFRAME
# =========================================================

def dataframe_to_json(
    data
):

    # -----------------------------------------------------
    # Jika kosong
    # -----------------------------------------------------

    if data is None:

        return "[]"

    try:

        # -------------------------------------------------
        # DataFrame
        # -------------------------------------------------

        if hasattr(
            data,
            "to_dict",
        ):

            records = data.to_dict(
                orient="records"
            )

        # -------------------------------------------------
        # List / dict / iterable
        # -------------------------------------------------

        else:

            records = data

        # -------------------------------------------------
        # JSON
        # -------------------------------------------------

        return json.dumps(
            records,
            ensure_ascii=False,
            default=str,
        )

    except Exception:

        return "[]"


# =========================================================
# SERIALIZE SUMMARY
# =========================================================

def summary_to_json(
    summary
):

    if summary is None:

        summary = {}

    try:

        return json.dumps(
            summary,
            ensure_ascii=False,
            default=str,
        )

    except Exception:

        return "{}"


# =========================================================
# SAFE JSON LOAD
# =========================================================

def safe_json_load(
    value,
    default,
):

    # -----------------------------------------------------
    # Null
    # -----------------------------------------------------

    if value is None:

        return default

    # -----------------------------------------------------
    # Empty
    # -----------------------------------------------------

    if not value:

        return default

    # -----------------------------------------------------
    # Already Python object
    # -----------------------------------------------------

    if isinstance(
        value,
        (
            list,
            dict,
        ),
    ):

        return value

    # -----------------------------------------------------
    # JSON decode
    # -----------------------------------------------------

    try:

        result = json.loads(
            value
        )

        return result

    except Exception:

        return default


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(
    nama_user,
    periode_forecast,
    history_months=3,
    forecast_bbb=None,
    forecast_bbt=None,
    summary=None,
):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # -------------------------------------------------
        # VALIDASI NAMA USER
        # -------------------------------------------------

        nama_user = str(
            nama_user
        ).strip()

        if not nama_user:

            raise ValueError(
                "Nama user tidak boleh kosong."
            )

        # -------------------------------------------------
        # VALIDASI PERIODE
        # -------------------------------------------------

        periode_forecast = str(
            periode_forecast
        ).strip()

        if not periode_forecast:

            raise ValueError(
                "Periode forecast tidak boleh kosong."
            )

        # -------------------------------------------------
        # VALIDASI HISTORY MONTHS
        # -------------------------------------------------
        #
        # Database menyimpan angka aktual.
        # Mode semua histori sudah diselesaikan oleh main.py
        # sebelum data masuk ke fungsi ini.
        # -------------------------------------------------

        history_months = normalize_history_months(
            history_months,
            default=3,
            allow_all=False,
        )

        # -------------------------------------------------
        # DATAFRAME → JSON
        # -------------------------------------------------

        forecast_bbb_json = (
            dataframe_to_json(
                forecast_bbb
            )
        )

        forecast_bbt_json = (
            dataframe_to_json(
                forecast_bbt
            )
        )

        # -------------------------------------------------
        # SUMMARY → JSON
        #
        # Summary dapat berisi:
        #
        # BBB
        # BBT
        # WAPE MA
        # Accuracy MA
        # WAPE WMA
        # Accuracy WMA
        # WAPE XGBoost
        # Accuracy XGBoost
        # Recursive Forecasting status
        # dan informasi forecasting lainnya.
        #
        # Database hanya menyimpan summary JSON.
        # Pemilihan Best Method tidak dilakukan oleh sistem.
        # -------------------------------------------------

        summary_json = (
            summary_to_json(
                summary
            )
        )

        # -------------------------------------------------
        # TIMESTAMP
        # -------------------------------------------------

        created_at = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        # -------------------------------------------------
        # INSERT
        # -------------------------------------------------

        cursor.execute(
            """
            INSERT INTO forecast_history
            (
                nama_user,
                periode_forecast,
                history_months,
                created_at,
                status,
                forecast_bbb,
                forecast_bbt,
                summary
            )
            VALUES
            (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                nama_user,
                periode_forecast,
                history_months,
                created_at,
                "Saved",
                forecast_bbb_json,
                forecast_bbt_json,
                summary_json,
            ),
        )

        conn.commit()

        new_id = cursor.lastrowid

        return new_id

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# =========================================================
# LOAD ALL HISTORY
# =========================================================

def load_history():

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT
                id,
                nama_user,
                periode_forecast,
                history_months,
                created_at,
                status,
                forecast_bbb,
                forecast_bbt,
                summary

            FROM forecast_history

            ORDER BY
                id DESC
            """
        )

        data = cursor.fetchall()

        return data

    finally:

        conn.close()


# =========================================================
# LOAD HISTORY BY ID
# =========================================================

def load_history_by_id(
    history_id
):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # -------------------------------------------------
        # Validasi ID
        # -------------------------------------------------

        try:

            history_id = int(
                history_id
            )

        except Exception:

            return None

        # -------------------------------------------------
        # Ambil history
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                nama_user,
                periode_forecast,
                history_months,
                created_at,
                status,
                forecast_bbb,
                forecast_bbt,
                summary

            FROM forecast_history

            WHERE id = ?
            """,
            (
                history_id,
            ),
        )

        row = cursor.fetchone()

        # -------------------------------------------------
        # Tidak ditemukan
        # -------------------------------------------------

        if row is None:

            return None

        # -------------------------------------------------
        # JSON → PYTHON
        # -------------------------------------------------

        forecast_bbb = (
            safe_json_load(
                row["forecast_bbb"],
                [],
            )
        )

        forecast_bbt = (
            safe_json_load(
                row["forecast_bbt"],
                [],
            )
        )

        summary = (
            safe_json_load(
                row["summary"],
                {},
            )
        )

        # -------------------------------------------------
        # History months
        # -------------------------------------------------

        history_months = (
            row["history_months"]
        )

        history_months = normalize_history_months(
            history_months,
            default=3,
            allow_all=False,
        )

        # -------------------------------------------------
        # Return
        # -------------------------------------------------

        return {

            "id":
                row["id"],

            "nama_user":
                row["nama_user"],

            "periode_forecast":
                row[
                    "periode_forecast"
                ],

            "history_months":
                history_months,

            "created_at":
                row["created_at"],

            "status":
                row["status"],

            "forecast_bbb":
                forecast_bbb,

            "forecast_bbt":
                forecast_bbt,

            "summary":
                summary,

        }

    finally:

        conn.close()


# =========================================================
# DELETE ONE HISTORY
# =========================================================

def delete_history(
    history_id
):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # -------------------------------------------------
        # Validasi ID
        # -------------------------------------------------

        try:

            history_id = int(
                history_id
            )

        except Exception:

            return False

        cursor.execute(
            """
            DELETE FROM forecast_history

            WHERE id = ?
            """,
            (
                history_id,
            ),
        )

        deleted_rows = (
            cursor.rowcount
        )

        conn.commit()

        return (
            deleted_rows > 0
        )

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# =========================================================
# DELETE ALL HISTORY
# =========================================================

def delete_all_history():

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            DELETE FROM forecast_history
            """
        )

        deleted_rows = (
            cursor.rowcount
        )

        conn.commit()

        return deleted_rows

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# =========================================================
# COUNT HISTORY
# =========================================================

def count_history():

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT
                COUNT(*) AS total
            FROM forecast_history
            """
        )

        row = cursor.fetchone()

        if row is None:

            return 0

        return int(
            row["total"]
        )

    finally:

        conn.close()


# =========================================================
# MODULE TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "DATABASE MODULE"
    )

    print(
        "=" * 60
    )

    init_db()

    print(
        "Database       :",
        DB_FILE,
    )

    print(
        "Total history  :",
        count_history(),
    )

    print(
        "Status         : OK"
    )
