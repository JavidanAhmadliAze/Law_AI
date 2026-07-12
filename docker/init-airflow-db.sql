-- Separate database for Airflow metadata (same Postgres instance, isolated DB)
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO lawai;
