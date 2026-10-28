import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import database
import report_gen
from src.inference import (
    CATEGORIES,
    build_or_load_memory_bank,
    get_category_samples,
    inspect_image,
)

# Initialize Streamlit Page Config
st.set_page_config(
    page_title="Industrial Defect Detection AI | MVTec-AD",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database and Seed initial demo records if empty
database.init_db()
database.seed_demo_data_if_empty()

# Custom Industrial Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Dark Industrial Theme Overrides */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #080b10 90%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top Header Bar */
    .industrial-header {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .industrial-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #f0f6fc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .industrial-sub {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 4px;
    }
    .badge-system {
        background: #21262d;
        border: 1px solid #388bfd;
        color: #58a6ff;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Metric HUD Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 700;
        font-family: 'SF Mono', Consolas, Monaco, monospace;
        margin: 4px 0;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status Badges */
    .status-box-normal {
        background: rgba(46, 160, 67, 0.15);
        border: 2px solid #2ea043;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.2);
    }
    .status-box-defect {
        background: rgba(248, 81, 73, 0.15);
        border: 2px solid #f85149;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 0 15px rgba(248, 81, 73, 0.25);
    }
    .status-text-normal {
        color: #3fb950;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 0;
    }
    .status-text-defect {
        color: #ff7b72;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 0;
    }

    /* Image Containers */
    .image-frame {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .image-caption {
        font-size: 0.82rem;
        color: #8b949e;
        font-weight: 600;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 6px 6px 0px 0px;
        color: #8b949e;
        border: 1px solid #30363d;
        border-bottom: none;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        border-top: 2px solid #58a6ff !important;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Top Bar Header
st.markdown(
    """
    <div class="industrial-header">
        <div>
            <div class="industrial-title">
                <span>🏭</span> MVTec-AD Automated Optical Inspection (AOI)
            </div>
            <div class="industrial-sub">
                Explainable AI Defect Localization • PatchCore ResNet-18 Subsurface Diagnostics
            </div>
        </div>
        <div>
            <span class="badge-system">STATION #1 • LIVE INLINE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cache memory bank preloading for snappy category switching
@st.cache_resource(show_spinner=False)
def preload_category_bank(category: str):
    return build_or_load_memory_bank(category)

# Multi-tab layout
tab1, tab2, tab3 = st.tabs([
    "🔬 Live Inspection",
    "📊 Plant Quality Analytics",
    "📋 Audit Trail & Inspection History",
])

# ==============================================================================
# TAB 1: LIVE INSPECTION
# ==============================================================================
with tab1:
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Inspection Configuration")

        # Category Dropdown
        category_icons = {
            "bottle": "🍾 bottle",
            "metal_nut": "🔩 metal_nut",
            "leather": "🪡 leather",
            "cable": "🔌 cable",
        }
        selected_category = st.selectbox(
            "Product Category",
            options=CATEGORIES,
            format_func=lambda x: category_icons.get(x, x),
            index=0,
            help="Select the industrial component family to evaluate.",
        )

        # Trigger preloading in background
        preload_category_bank(selected_category)

        # Decision Threshold Slider
        threshold = st.slider(
            "Decision Threshold",
            min_value=1.0,
            max_value=6.0,
            value=3.0,
            step=0.1,
            help="Anomaly scores equal or exceeding this threshold are flagged as defective (Default: 3.0).",
        )

        st.markdown("---")
        st.markdown("### 📷 Sample Input Selection")

        input_mode = st.radio(
            "Image Source",
            options=["MVTec AD Test Set Samples", "Upload Custom Image"],
            index=0,
        )

        image_to_inspect = None
        sample_label = "unknown"
        display_filename = ""

        if input_mode == "MVTec AD Test Set Samples":
            samples_dict = get_category_samples(selected_category)
            if samples_dict:
                # Defect subfolder dropdown
                defect_types = list(samples_dict.keys())
                # Default to a defect if available for immediate demonstration
                default_idx = 0
                for i, d in enumerate(defect_types):
                    if d != "good":
                        default_idx = i
                        break

                selected_type = st.selectbox("Defect Subfolder", defect_types, index=default_idx)
                files_for_type = samples_dict[selected_type]

                selected_file_path = st.selectbox(
                    "Image Sample",
                    files_for_type,
                    format_func=lambda p: Path(p).name,
                )

                if selected_file_path and os.path.exists(selected_file_path):
                    image_to_inspect = selected_file_path
                    sample_label = selected_type
                    display_filename = Path(selected_file_path).name
            else:
                st.warning(f"No test samples found in data/{selected_category}/test")

        else:
            uploaded_file = st.file_uploader(
                "Upload Part Image",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Upload a 2D optical surface capture for real-time inspection.",
            )
            if uploaded_file is not None:
                image_to_inspect = Image.open(uploaded_file).convert("RGB")
                display_filename = uploaded_file.name
                sample_label = "custom_upload"

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#8b949e;">
                <b>Model Architecture:</b><br/>
                PatchCore Anomaly Detector<br/>
                Backbone: ResNet-18 (layer2+layer3)<br/>
                XAI: Exact Patch Deviation Heatmap
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Area: 3-column view
    if image_to_inspect is not None:
        # Perform Inspection
        with st.spinner("Analyzing optical scan & computing patch embeddings..."):
            result = inspect_image(
                image=image_to_inspect,
                category=selected_category,
                threshold=threshold,
                save_artifacts=True,
                output_prefix="live",
            )

        # Log inspection automatically to SQLite
        try:
            database.log_inspection(
                filename=display_filename or result["filename"],
                category=selected_category,
                status=result["status"],
                anomaly_score=result["anomaly_score"],
                threshold=result["threshold"],
                image_path=result["image_path"],
                heatmap_path=result["heatmap_path"],
            )
        except Exception as e:
            st.error(f"Failed to log inspection to database: {e}")

        # 3-Column Display: Original | Heatmap Overlay | Verdict & Metrics
        col1, col2, col3 = st.columns([1, 1, 1.1], gap="medium")

        with col1:
            st.markdown(
                """
                <div class="image-frame">
                    <div class="image-caption">📷 Optical Camera Capture</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(result["original_image"], use_container_width=True)
            st.caption(f"File: **{display_filename or result['filename']}** | Category: **{selected_category}**")

        with col2:
            st.markdown(
                """
                <div class="image-frame">
                    <div class="image-caption">🔥 Explainable AI Heatmap (XAI)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(result["heatmap_overlay"], use_container_width=True)
            st.caption("Warm regions (red/yellow) indicate anomalous deviation from golden coreset.")

        with col3:
            # Inspection Verdict Box
            if result["is_defect"]:
                st.markdown(
                    f"""
                    <div class="status-box-defect">
                        <div class="status-text-defect">⚠️ DEFECT DETECTED</div>
                        <div style="font-size:0.85rem; color:#f85149; margin-top:4px;">
                            Anomaly score exceeds threshold limit.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-box-normal">
                        <div class="status-text-normal">✅ NORMAL (PASS)</div>
                        <div style="font-size:0.85rem; color:#3fb950; margin-top:4px;">
                            Surface pattern conforms to reference distribution.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Telemetry Metrics Grid
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Anomaly Score</div>
                        <div class="metric-val" style="color:{'#ff7b72' if result['is_defect'] else '#3fb950'};">
                            {result['anomaly_score']:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Threshold</div>
                        <div class="metric-val" style="color:#58a6ff;">
                            {result['threshold']:.1f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            m3, m4 = st.columns(2)
            with m3:
                delta = result["score_delta"]
                delta_sign = "+" if delta > 0 else ""
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Threshold Delta</div>
                        <div class="metric-val" style="font-size:1.4rem; color:{'#ff7b72' if delta>0 else '#3fb950'};">
                            {delta_sign}{delta:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Inference Latency</div>
                        <div class="metric-val" style="font-size:1.4rem; color:#e6edf3;">
                            {result['latency_ms']:.1f} ms
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Anomaly Gauge Indicator
            max_gauge = max(6.0, result["anomaly_score"] + 1.0)
            score_pct = min(1.0, max(0.0, result["anomaly_score"] / max_gauge))
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="font-size:0.75rem; color:#8b949e; display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span>Anomaly Level vs Scale (Max {max_gauge:.1f})</span>
                    <span>Threshold: {result['threshold']:.1f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(score_pct)

            # Action: Download PDF Report
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            pdf_report_path = "static/inspection_report.pdf"
            report_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "category": selected_category,
                "filename": display_filename or result["filename"],
                "status": result["status"],
                "anomaly_score": result["anomaly_score"],
                "threshold": result["threshold"],
                "latency_ms": result["latency_ms"],
                "image_path": result["image_path"],
                "heatmap_path": result["heatmap_path"],
            }
            report_gen.generate_pdf_report(report_data, output_path=pdf_report_path)

            if os.path.exists(pdf_report_path):
                with open(pdf_report_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

                st.download_button(
                    label="📄 Download PDF Inspection Report",
                    data=pdf_bytes,
                    file_name=f"inspection_report_{selected_category}_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    else:
        st.info("👈 Please select a sample image or upload a custom image from the sidebar to inspect.")


# ==============================================================================
# TAB 2: PLANT QUALITY ANALYTICS
# ==============================================================================
with tab2:
    st.markdown("### 📊 Plant Quality Control & Metrology Analytics")
    stats = database.get_summary_stats()
    df_all = database.get_inspections_df()

    # KPI Top Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Total Scans Performed</div>
                <div class="metric-val" style="color:#58a6ff;">{stats['total']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Defects Detected</div>
                <div class="metric-val" style="color:#f85149;">{stats['defects']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Defect Rate</div>
                <div class="metric-val" style="color:{'#ff7b72' if stats['defect_rate']>15 else '#3fb950'};">
                    {stats['defect_rate']:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Mean Anomaly Score</div>
                <div class="metric-val" style="color:#e6edf3;">{stats['avg_score']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    if not df_all.empty:
        # Plotly Charts Row
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Defect Distribution Pie Chart
            status_counts = df_all["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            color_map = {
                "Normal": "#2ea043",
                "Defect Detected": "#f85149",
            }

            fig_pie = px.pie(
                status_counts,
                names="Status",
                values="Count",
                title="Inspection Verdict Distribution",
                hole=0.45,
                color="Status",
                color_discrete_map=color_map,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6edf3"},
                margin=dict(t=50, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            # Anomaly Score Distribution Histogram
            fig_hist = px.histogram(
                df_all,
                x="anomaly_score",
                color="status",
                nbins=20,
                title="Anomaly Score Distribution vs. Decision Threshold",
                color_discrete_map=color_map,
                barmode="overlay",
                opacity=0.75,
            )
            # Add vertical threshold marker at 3.0
            fig_hist.add_vline(
                x=3.0,
                line_width=2.5,
                line_dash="dash",
                line_color="#e3b341",
                annotation_text="Standard Threshold (3.0)",
                annotation_position="top right",
                annotation_font_color="#e3b341",
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6edf3"},
                xaxis=dict(title="Anomaly Score", gridcolor="#30363d"),
                yaxis=dict(title="Inspection Count", gridcolor="#30363d"),
                margin=dict(t=50, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Category Breakdown Bar Chart
        cat_counts = df_all.groupby(["category", "status"]).size().reset_index(name="Count")
        fig_bar = px.bar(
            cat_counts,
            x="category",
            y="Count",
            color="status",
            barmode="group",
            title="Part Category Inspection Breakdown (Normal vs Defective)",
            color_discrete_map=color_map,
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6edf3"},
            xaxis=dict(title="Product Category", gridcolor="#30363d"),
            yaxis=dict(title="Scans Count", gridcolor="#30363d"),
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No inspection records available yet to generate analytics.")


# ==============================================================================
# TAB 3: AUDIT TRAIL & INSPECTION HISTORY
# ==============================================================================
with tab3:
    st.markdown("### 📋 Audit Trail & Inspection Database")
    df_history = database.get_inspections_df()

    if not df_history.empty:
        # Filter controls
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            filter_cat = st.multiselect("Filter by Category", options=df_history["category"].unique(), default=[])
        with f_col2:
            filter_stat = st.multiselect("Filter by Status", options=df_history["status"].unique(), default=[])
        with f_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            csv_export_path = report_gen.export_history_to_csv()
            with open(csv_export_path, "rb") as f:
                csv_bytes = f.read()
            st.download_button(
                label="📥 Export History to CSV",
                data=csv_bytes,
                file_name="inspection_audit_trail.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Apply filters
        filtered_df = df_history.copy()
        if filter_cat:
            filtered_df = filtered_df[filtered_df["category"].isin(filter_cat)]
        if filter_stat:
            filtered_df = filtered_df[filtered_df["status"].isin(filter_stat)]

        # Display Interactive Dataframe
        st.dataframe(
            filtered_df[[
                "id", "timestamp", "filename", "category", "status", "anomaly_score", "threshold"
            ]].rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "filename": "Filename",
                "category": "Category",
                "status": "Verdict",
                "anomaly_score": "Score",
                "threshold": "Threshold",
            }),
            use_container_width=True,
            height=320,
        )

        # Image preview for an inspection record
        st.markdown("#### 🔍 Historical Inspection Evidence Viewer")
        selected_id = st.selectbox(
            "Select Inspection Record ID to Inspect Visual Evidence",
            options=filtered_df["id"].tolist(),
            format_func=lambda x: f"Inspection #{x}",
        )

        record_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f"**Original Optical Image:** `{record_row['image_path']}`")
            if os.path.exists(record_row["image_path"]):
                st.image(record_row["image_path"], use_container_width=True)
            else:
                st.warning("Original image file not found on disk.")
        with p_col2:
            st.markdown(f"**XAI Localization Heatmap:** `{record_row['heatmap_path']}`")
            if os.path.exists(record_row["heatmap_path"]):
                st.image(record_row["heatmap_path"], use_container_width=True)
            else:
                st.warning("Heatmap overlay image file not found on disk.")

    else:
        st.info("Inspection database is currently empty. Run a scan in Tab 1 to populate the audit trail.")
