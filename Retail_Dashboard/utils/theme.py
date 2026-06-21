import streamlit as st

def load_css():
    """Injects the custom CSS into the Streamlit app. Call once per page."""
    try:
        with open("assets/style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Make sure assets/style.css exists.")


SECTION_ICONS = {
    "default": "&#9670;",
    "model": "&#9881;",
    "forecast": "&#9650;",
    "segment": "&#9737;",
    "alert": "&#9888;",
    "search": "&#128269;",
}

def section_header(title, subtitle=None, icon="default"):
    icon_glyph = SECTION_ICONS.get(icon, SECTION_ICONS["default"])
    st.markdown(f"<h2><span class='section-icon'>{icon_glyph}</span>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color: #6B7785; margin-top: -10px; margin-left: 38px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)


def metric_card(label, value, status_color="none", icon="", bar_pct=None):
    """
    KPI card with a colored status dot, an optional icon, and an optional
    mini progress bar beneath the number.

    IMPORTANT: Streamlit's markdown HTML parser fails on 2+ levels of nested
    <div> tags in a single st.markdown() call (see streamlit/streamlit#3190).
    To work around this, the card is rendered with st.container(border=True)
    — a native widget that draws its own border/shadow — and each line of
    content inside is its own separate, shallow st.markdown call.
    """
    with st.container(border=True):
        dot_html = f'<span class="status-dot status-{status_color}"></span>' if status_color != "none" else ""
        icon_html = f'<span class="metric-icon">{icon}</span>' if icon else ""
        st.markdown(f'<div class="metric-label">{dot_html}{icon_html}{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)

        if bar_pct is not None:
            bar_color = status_color if status_color != "none" else "blue"
            pct = max(0, min(100, bar_pct))
            st.markdown(f'<div class="metric-bar-track"><div class="metric-bar-fill metric-bar-{bar_color}" style="width:{pct}%;"></div></div>', unsafe_allow_html=True)


def status_card(name, sub, stats, status="blue"):
    status_to_dot = {"good": "green", "alert": "red", "watch": "amber", "info": "blue"}
    border_color_map = {"good": "#2E6B4F", "alert": "#C8553D", "watch": "#B8853A", "info": "#2E5B8A"}
    dot_color = status_to_dot.get(status, "blue")
    border_color = border_color_map.get(status, "#2E5B8A")

    with st.container(border=True):
        st.markdown(
            f'<div style="border-left:3px solid {border_color}; padding-left:12px; margin:-1px 0;">'
            f'<span class="status-dot status-{dot_color}"></span>'
            f'<span style="font-weight:600; font-size:15px;">{name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<p style="font-size:12px; color:#6B7785; margin:6px 0 0 12px;">{sub}</p>', unsafe_allow_html=True)
        stat_html = "".join(f'<span>{label} <b>{value}</b></span>' for label, value in stats)
        st.markdown(f'<div class="stat-row" style="margin-left:12px;">{stat_html}</div>', unsafe_allow_html=True)


def pill(text, status="blue"):
    return f'<span class="status-pill pill-{status}">{text}</span>'


def show_image(filename, img_dir="Visualizations", caption=""):
    import os
    from PIL import Image
    path = os.path.join(img_dir, filename)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, width="stretch")
    else:
        st.warning(f"⚠ {filename} not found in {img_dir}")