import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

DB_NAME = "inspections.db"


def get_db_connection(db_path: str = DB_NAME) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_NAME) -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            status TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            threshold REAL NOT NULL,
            image_path TEXT NOT NULL,
            heatmap_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_inspection(
    filename: str,
    category: str,
    status: str,
    anomaly_score: float | None = None,
    threshold: float = 3.0,
    image_path: str = "",
    heatmap_path: str = "",
    db_path: str = DB_NAME,
    **kwargs: Any,
) -> int:
    init_db(db_path)

    # Support aliases for backward compatibility (e.g., score, img_path, heat_path)
    if anomaly_score is None:
        anomaly_score = kwargs.get("score", 0.0)
    if not image_path:
        image_path = kwargs.get("img_path", "")
    if not heatmap_path:
        heatmap_path = kwargs.get("heat_path", "")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute(
        """
        INSERT INTO inspections 
        (timestamp, filename, category, status, anomaly_score, threshold, image_path, heatmap_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            timestamp,
            str(filename),
            str(category),
            str(status),
            float(anomaly_score),
            float(threshold),
            str(image_path),
            str(heatmap_path),
        ),
    )
    inserted_id = c.lastrowid or 0
    conn.commit()
    conn.close()
    return inserted_id


def get_all_inspections(db_path: str = DB_NAME) -> list[tuple[Any, ...]]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, filename, category, status, anomaly_score, threshold, image_path, heatmap_path
        FROM inspections
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def get_inspections_df(db_path: str = DB_NAME) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    query = """
        SELECT id, timestamp, filename, category, status, anomaly_score, threshold, image_path, heatmap_path
        FROM inspections
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_summary_stats(db_path: str = DB_NAME) -> dict[str, Any]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM inspections")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM inspections WHERE status = 'Defect Detected'")
    defects = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM inspections WHERE status = 'Normal'")
    normal = c.fetchone()[0]

    c.execute("SELECT AVG(anomaly_score) FROM inspections")
    avg_score_row = c.fetchone()[0]
    avg_score = float(avg_score_row) if avg_score_row is not None else 0.0

    conn.close()
    defect_rate = (defects / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "defects": defects,
        "normal": normal,
        "defect_rate": defect_rate,
        "avg_score": avg_score,
    }


def seed_demo_data_if_empty(db_path: str = DB_NAME) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM inspections")
    count = c.fetchone()[0]
    conn.close()

    if count > 0:
        return

    # Seed initial representative inspection records across categories
    sample_records = [
        ("000.png", "bottle", "Normal", 1.47, 3.0, "data/bottle/test/good/000.png", "results/gradcam_mask_overlay.png"),
        ("001.png", "bottle", "Defect Detected", 3.82, 3.0, "data/bottle/test/broken_large/000.png", "results/gradcam_mask_overlay.png"),
        ("002.png", "bottle", "Defect Detected", 3.15, 3.0, "data/bottle/test/contamination/000.png", "results/gradcam_mask_overlay.png"),
        ("003.png", "cable", "Normal", 1.84, 3.0, "data/cable/test/good/000.png", "results/gradcam_mask_overlay.png"),
        ("004.png", "cable", "Defect Detected", 4.12, 3.0, "data/cable/test/bent_wire/000.png", "results/gradcam_mask_overlay.png"),
        ("005.png", "metal_nut", "Normal", 2.15, 3.0, "data/metal_nut/test/good/000.png", "results/gradcam_mask_overlay.png"),
        ("006.png", "metal_nut", "Defect Detected", 3.64, 3.0, "data/metal_nut/test/scratch/000.png", "results/gradcam_mask_overlay.png"),
        ("007.png", "leather", "Normal", 1.32, 3.0, "data/leather/test/good/000.png", "results/gradcam_mask_overlay.png"),
        ("008.png", "leather", "Defect Detected", 4.78, 3.0, "data/leather/test/cut/000.png", "results/gradcam_mask_overlay.png"),
        ("009.png", "bottle", "Normal", 1.62, 3.0, "data/bottle/test/good/001.png", "results/gradcam_mask_overlay.png"),
    ]

    for fname, cat, stat, score, thresh, img_p, heat_p in sample_records:
        log_inspection(fname, cat, stat, score, thresh, img_p, heat_p, db_path=db_path)
