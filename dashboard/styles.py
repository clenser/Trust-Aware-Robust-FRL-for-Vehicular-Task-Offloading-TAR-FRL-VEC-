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

    /* Metric Cards - Guaranteed High-Contrast Black Typography */
    div[data-testid="stMetric"],
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 4px 0 rgba(0, 0, 0, 0.06) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="stMetric"]:hover,
    [data-testid="stMetric"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px -1px rgba(0, 0, 0, 0.12) !important;
        border-color: #94a3b8 !important;
    }

    /* ALL Labels inside Metric Cards: Deep Black, Bold, High-Contrast */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span,
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"],
    label[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabelText"],
    div[data-testid="stMetricLabelText"] p,
    div[data-testid="stMetricLabelText"] span {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        opacity: 1 !important;
        visibility: visible !important;
        line-height: 1.3 !important;
    }

    /* ALL Values inside Metric Cards: Bold Black */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p,
    [data-testid="stMetricValue"] span,
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Delta Sub-Text inside Metric Cards: Dark Charcoal/Black */
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] *,
    [data-testid="stMetricDelta"] div,
    [data-testid="stMetricDelta"] p,
    [data-testid="stMetricDelta"] span,
    div[data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
        visibility: visible !important;
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
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 4px solid #2563eb !important;
        border-radius: 6px !important;
        padding: 14px 18px !important;
        margin-bottom: 1.0rem !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
    }
    .research-card,
    .research-card * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    .research-card-title {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        margin-bottom: 4px !important;
    }
    .research-card-body {
        font-size: 0.88rem !important;
        color: #1e293b !important;
        -webkit-text-fill-color: #1e293b !important;
        line-height: 1.5 !important;
    }

    /* Telemetry Panels */
    .telemetry-card {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        padding: 12px 14px !important;
        margin-bottom: 8px !important;
    }
    .telemetry-card,
    .telemetry-card * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    .telemetry-title {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        margin-bottom: 6px !important;
    }
    .telemetry-row {
        display: flex !important;
        justify-content: space-between !important;
        font-size: 0.85rem !important;
        margin-bottom: 3px !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    .telemetry-row span {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    .telemetry-val {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-weight: 700 !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
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
