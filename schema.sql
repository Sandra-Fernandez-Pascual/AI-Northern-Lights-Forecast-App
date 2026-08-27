-- Anonymous aurora forecast search log.
-- Run this once in the Neon SQL Editor.

CREATE TABLE IF NOT EXISTS forecast_searches (
    id BIGSERIAL PRIMARY KEY,
    searched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    destination TEXT NOT NULL,
    forecast_date DATE NOT NULL,
    aurora_probability NUMERIC(5, 2),
    cloud_cover NUMERIC(5, 2),
    visibility NUMERIC(10, 2),
    forecast_succeeded BOOLEAN NOT NULL,
    error_type TEXT,
    sky_too_bright BOOLEAN,
    viewing_outcome TEXT NOT NULL
);
