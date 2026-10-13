"""
Custom styling for the VANET FRL Research Dashboard.
Provides a clean, professional, research-grade design system:
- High contrast typography and subtle borders
- Cohesive color scheme (Slate / Deep Navy / Electric Teal / Emerald / Crimson accents)
- Restrained visual hierarchy without distracting clutter
- Dedicated styles for 2D road simulation canvas and telemetry inspectors
"""

def get_custom_css() -> str:
    return """
    <style>
    /* Global Container */
    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1440px;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #0f172a;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Badge & Tag Helpers */
    .badge-primary {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Callout Card */
    .research-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 1.0rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .research-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .research-card-body {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Telemetry Panels */
    .telemetry-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .telemetry-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 3px;
        color: #1e293b;
    }
    .telemetry-val {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-weight: 600;
    }

    /* Road Simulation Container */
    .sim-canvas-container {
        background-color: #0b0f19;
        border-radius: 10px;
        padding: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        margin-bottom: 1rem;
    }

    /* Table adjustments */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }

    /* Tab navigation polish */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """
