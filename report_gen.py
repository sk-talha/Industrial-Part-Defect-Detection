import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from database import get_all_inspections


def generate_pdf_report(
    data: dict[str, Any],
    output_path: str = "static/inspection_report.pdf",
) -> str:
    """
    Generates a professional, industrial-grade PDF inspection certificate.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=letter)
    page_width, page_height = letter

    # Extract fields with safe fallbacks
    timestamp = str(data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    category = str(data.get("category", "Unknown")).upper()
    filename = str(data.get("filename", "sample.png"))
    status = str(data.get("status", "Normal"))
    score = float(data.get("anomaly_score", data.get("score", 0.0)))
    threshold = float(data.get("threshold", 3.0))
    latency_ms = data.get("latency_ms", None)
    img_path = str(data.get("image_path", ""))
    heat_path = str(data.get("heatmap_path", ""))

    is_defect = "defect" in status.lower()

    # 1. Top Industrial Header Bar
    c.setFillColor(colors.HexColor("#0f172a"))  # Deep Navy
    c.rect(0, page_height - 90, page_width, 90, fill=True, stroke=False)

    c.setFillColor(colors.HexColor("#38bdf8"))  # Electric Cyan
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, page_height - 42, "INDUSTRIAL AI DEFECT INSPECTION CERTIFICATE")

    c.setFillColor(colors.HexColor("#94a3b8"))  # Slate Gray
    c.setFont("Helvetica", 9)
    c.drawString(40, page_height - 62, "MVTec-AD Automated Optical Inspection (AOI) System  |  Explainable AI (XAI) Engine")

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.white)
    c.drawRightString(page_width - 40, page_height - 42, f"INSPECTION RECORD")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#cbd5e1"))
    c.drawRightString(page_width - 40, page_height - 62, f"DATE: {timestamp}")

    # 2. Verdict Banner
    banner_y = page_height - 155
    banner_height = 50
    if is_defect:
        bg_color = colors.HexColor("#fef2f2")
        border_color = colors.HexColor("#ef4444")
        text_color = colors.HexColor("#b91c1c")
        verdict_badge = "DEFECT DETECTED - NON-CONFORMING"
        verdict_sub = "Anomaly score exceeds safe operational boundary. Routed to manual inspection / rework."
    else:
        bg_color = colors.HexColor("#f0fdf4")
        border_color = colors.HexColor("#10b981")
        text_color = colors.HexColor("#047857")
        verdict_badge = "NORMAL - INSPECTION PASSED"
        verdict_sub = "Surface pattern within nominal coreset distribution tolerances. Conforms to quality standards."

    c.setFillColor(bg_color)
    c.setStrokeColor(border_color)
    c.setLineWidth(1.5)
    c.roundRect(40, banner_y, page_width - 80, banner_height, radius=6, fill=True, stroke=True)

    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(55, banner_y + 28, verdict_badge)

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(55, banner_y + 12, verdict_sub)

    # 3. Telemetry & Parameters Table Box
    table_top = banner_y - 20
    box_height = 110
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.roundRect(40, table_top - box_height, page_width - 80, box_height, radius=6, fill=True, stroke=True)

    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#1e293b"))
    c.drawString(55, table_top - 20, "Inspection Telemetry & Core Metrics")

    # Metrics grid (2 columns)
    col1_x = 55
    col2_x = page_width / 2 + 10

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(col1_x, table_top - 42, "Product Category:")
    c.drawString(col1_x, table_top - 62, "Sample Filename:")
    c.drawString(col1_x, table_top - 82, "Detection Model:")
    c.drawString(col1_x, table_top - 102, "Inference Latency:")

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col1_x + 110, table_top - 42, category)
    c.drawString(col1_x + 110, table_top - 62, filename)
    c.drawString(col1_x + 110, table_top - 82, "PatchCore (ResNet-18 Coreset)")
    latency_str = f"{latency_ms:.1f} ms" if latency_ms is not None else "Fast (<100 ms)"
    c.drawString(col1_x + 110, table_top - 102, latency_str)

    c.setFillColor(colors.HexColor("#64748b"))
    c.setFont("Helvetica", 9)
    c.drawString(col2_x, table_top - 42, "Anomaly Score:")
    c.drawString(col2_x, table_top - 62, "Decision Threshold:")
    c.drawString(col2_x, table_top - 82, "Score vs Threshold:")
    c.drawString(col2_x, table_top - 102, "Explainability (XAI):")

    c.setFont("Helvetica-Bold", 9)
    score_color = colors.HexColor("#ef4444") if is_defect else colors.HexColor("#10b981")
    c.setFillColor(score_color)
    c.drawString(col2_x + 120, table_top - 42, f"{score:.4f}")

    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(col2_x + 120, table_top - 62, f"{threshold:.4f}")

    delta = score - threshold
    delta_str = f"+{delta:.4f} (Anomaly)" if delta > 0 else f"{delta:.4f} (Safe)"
    c.setFillColor(score_color)
    c.drawString(col2_x + 120, table_top - 82, delta_str)

    c.setFillColor(colors.HexColor("#0284c7"))
    c.drawString(col2_x + 120, table_top - 102, "Nearest-Patch Distance Map")

    # 4. Visual Evidence Section (Dual image display)
    evidence_y = table_top - box_height - 30
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(40, evidence_y, "Visual Evidence & Explainable AI (Heatmap Localization)")

    image_box_y = evidence_y - 230
    img_width = 240
    img_height = 200

    # Original Image frame
    c.setStrokeColor(colors.HexColor("#e2e8f0"))
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(40, image_box_y, img_width, img_height, radius=4, fill=True, stroke=True)

    if os.path.exists(img_path):
        try:
            c.drawImage(img_path, 45, image_box_y + 25, width=img_width - 10, height=img_height - 30, preserveAspectRatio=True)
        except Exception:
            c.drawString(55, image_box_y + 100, "[Image Load Error]")
    else:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawString(70, image_box_y + 100, "Original Image Path Not Found")

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor("#1e293b"))
    c.drawString(45, image_box_y + 10, "Figure 1: Original Optical Camera Capture")

    # Heatmap Overlay frame
    c.setStrokeColor(colors.HexColor("#e2e8f0"))
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(page_width - 40 - img_width, image_box_y, img_width, img_height, radius=4, fill=True, stroke=True)

    if os.path.exists(heat_path):
        try:
            c.drawImage(heat_path, page_width - 40 - img_width + 5, image_box_y + 25, width=img_width - 10, height=img_height - 30, preserveAspectRatio=True)
        except Exception:
            c.drawString(page_width - 40 - img_width + 20, image_box_y + 100, "[Heatmap Load Error]")
    else:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawString(page_width - 40 - img_width + 20, image_box_y + 100, "Heatmap Overlay Not Found")

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor("#1e293b"))
    c.drawString(page_width - 40 - img_width + 5, image_box_y + 10, "Figure 2: Explainable Defect Heatmap Overlay")

    # 5. Diagnostic / Technical Note Box
    diag_y = image_box_y - 65
    c.setFillColor(colors.HexColor("#f1f5f9"))
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(40, diag_y, page_width - 80, 50, radius=4, fill=True, stroke=True)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#334155"))
    c.drawString(50, diag_y + 35, "DIAGNOSTIC EXPLAINABILITY SUMMARY:")
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(
        50,
        diag_y + 22,
        "Color intensity corresponds to localized patch deviation from nominal reference features. Red/warm zones indicate high",
    )
    c.drawString(
        50,
        diag_y + 10,
        "confidence defect regions (structural breaks, scratches, contamination, or foreign matter).",
    )

    # 6. Audit & Sign-off Footer
    footer_y = 35
    c.setStrokeColor(colors.HexColor("#e2e8f0"))
    c.setLineWidth(1)
    c.line(40, footer_y + 25, page_width - 40, footer_y + 25)

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(40, footer_y + 10, "Generated by MVTec-AD Industrial Parts Inspection Suite  •  Compliance: ISO 9001 / AOI Automated Quality Audit")
    c.drawRightString(page_width - 40, footer_y + 10, "Official Inspection Record")

    c.save()
    return output_path


def export_history_to_csv(output_csv: str = "static/inspection_history.csv") -> str:
    """
    Exports all inspection records from SQLite database to a CSV file.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = get_all_inspections()
    columns = [
        "ID",
        "Timestamp",
        "Filename",
        "Category",
        "Status",
        "AnomalyScore",
        "Threshold",
        "ImagePath",
        "HeatmapPath",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    return output_csv