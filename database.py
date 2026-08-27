"""Anonymous search logging for the Aurora Forecast app.

Connects to Neon PostgreSQL and stores one row per Generate click.
A database problem never raises, so the forecast can still run.
"""

import math

import pandas as pd
import psycopg


INSERT_SQL = """
INSERT INTO forecast_searches (
    destination,
    forecast_date,
    aurora_probability,
    cloud_cover,
    visibility,
    forecast_succeeded,
    error_type,
    sky_too_bright,
    viewing_outcome
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

SELECT_SQL = """
SELECT
    searched_at,
    destination,
    forecast_date,
    aurora_probability,
    cloud_cover,
    visibility,
    forecast_succeeded,
    error_type,
    sky_too_bright,
    viewing_outcome
FROM forecast_searches
ORDER BY searched_at DESC
LIMIT %s
"""


def to_sql_null(value):
    """Turn pandas/NumPy missing numbers into None so PostgreSQL stores NULL."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def get_connection(database_url):
    """Open a PostgreSQL connection, or return None if connect fails."""
    if not database_url:
        return None
    try:
        return psycopg.connect(database_url, connect_timeout=5)
    except Exception:
        return None


def log_search(
    database_url,
    destination,
    forecast_date,
    aurora_probability,
    cloud_cover,
    visibility,
    forecast_succeeded,
    error_type,
    sky_too_bright,
    viewing_outcome
):
    """Insert one anonymous search row. Never raises to the caller."""
    connection = get_connection(database_url)
    if connection is None:
        return

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    INSERT_SQL,
                    (
                        destination or "unknown",
                        forecast_date,
                        to_sql_null(aurora_probability),
                        to_sql_null(cloud_cover),
                        to_sql_null(visibility),
                        bool(forecast_succeeded),
                        error_type,
                        sky_too_bright,
                        viewing_outcome
                    )
                )
    except Exception:
        pass
    finally:
        try:
            connection.close()
        except Exception:
            pass


def fetch_searches(database_url, limit=500):
    """Return recent search rows as a DataFrame, or an empty DataFrame on failure."""
    connection = get_connection(database_url)
    if connection is None:
        return pd.DataFrame()

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(SELECT_SQL, (limit,))
                rows = cursor.fetchall()
                columns = [
                    "searched_at",
                    "destination",
                    "forecast_date",
                    "aurora_probability",
                    "cloud_cover",
                    "visibility",
                    "forecast_succeeded",
                    "error_type",
                    "sky_too_bright",
                    "viewing_outcome"
                ]
                return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            connection.close()
        except Exception:
            pass
