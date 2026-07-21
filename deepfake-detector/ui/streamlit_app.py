"""
Obscuro Deepimage — Streamlit frontend (dark-themed, matches web UI aesthetic).
"""
import os
import json
import hashlib
from typing import Optional
import streamlit as st
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Obscuro Deepimage",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Pixel-perfect CSS override ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide all Streamlit chrome */
  #MainMenu, header, footer,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  [data-testid="collapsedControl"],
  .stDeployButton { display: none !important; }

  /* Dark background */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main, .block-container {
    background-color: #0d0d0d !important;
  }

  /* Remove default padding */
  .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
  }

  /* Base text */
  body, p, span, div, label {
    color: #aaaaaa;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
</style>
""", unsafe_allow_html=True)

# ── Navbar ────────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="16" cy="16" r="14" fill="#383838"/>
  <rect x="14.4" y="4.8" width="3.2" height="3.2" rx="0.6" fill="#909090"/>
  <rect x="22.4" y="7.6" width="3.2" height="3.2" rx="0.6" fill="#909090"
        transform="rotate(45 24 9.2)"/>
  <rect x="23.8" y="14.4" width="3.2" height="3.2" rx="0.6" fill="#909090"/>
  <rect x="22.4" y="21.2" width="3.2" height="3.2" rx="0.6" fill="#909090"
        transform="rotate(-45 24 22.8)"/>
  <rect x="14.4" y="24" width="3.2" height="3.2" rx="0.6" fill="#909090"/>
  <rect x="6.4" y="21.2" width="3.2" height="3.2" rx="0.6" fill="#909090"
        transform="rotate(45 8 22.8)"/>
  <rect x="5" y="14.4" width="3.2" height="3.2" rx="0.6" fill="#909090"/>
  <rect x="6.4" y="7.6" width="3.2" height="3.2" rx="0.6" fill="#909090"
        transform="rotate(-45 8 9.2)"/>
  <circle cx="16" cy="16" r="2.4" fill="#909090"/>
</svg>"""

st.markdown(f"""
<style>
  .obs-navbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 28px;
    background: #0d0d0d;
    position: sticky; top: 0; z-index: 100;
  }}
  .obs-brand {{ display: flex; align-items: center; gap: 10px; }}
  .obs-brand-name {{ color: #fff; font-size: 15px; font-weight: 500; }}
  .obs-hamburger {{ display: flex; flex-direction: column; gap: 5px; }}
  .obs-hamburger span {{ display: block; width: 22px; height: 2.5px; background: #888; border-radius: 1px; }}
</style>
<div class="obs-navbar">
  <div class="obs-brand">
    {LOGO_SVG}
    <span class="obs-brand-name">Obscuro</span>
  </div>
  <div class="obs-hamburger">
    <span></span><span></span><span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .obs-title {
    font-family: Georgia, "Times New Roman", Times, serif;
    font-size: 52px; font-weight: 400; color: #7d7d7d;
    text-align: center; margin: 28px 0 32px;
    line-height: 1.1;
  }
</style>
<h1 class="obs-title">Obscuro Deepimage</h1>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def call_analyse(file_bytes: bytes, filename: str) -> Optional[dict]:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{API_URL}/api/analyse",
                files={"file": (filename, file_bytes, "application/octet-stream")},
            )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text[:400]}")
        return None
    except httpx.ConnectError:
        st.error("Cannot connect to the detection backend (port 8000).")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None

def score_color(s: float) -> str:
    if s > 0.58: return "#f87171"
    if s < 0.42: return "#6ee7b7"
    return "#fbbf24"

def render_verdict(result: dict):
    verdict = result.get("verdict", "UNCERTAIN")
    prob    = result.get("deepfake_probability", 0.5)
    conf    = result.get("confidence_in_verdict", 0.0)
    media   = result.get("media_type", "image")
    pt      = result.get("processing_time_s", 0)

    vcolor  = {"LIKELY FAKE": "#f87171", "LIKELY REAL": "#6ee7b7"}.get(verdict, "#fbbf24")

    st.markdown(f"""
    <style>
      .obs-verdict-card {{
        background:#282828;border-radius:16px;padding:28px;
        text-align:center;margin-bottom:16px;
      }}
      .obs-verdict-label {{
        font-family:Georgia,serif;font-size:38px;font-weight:400;
        color:{vcolor};margin-bottom:8px;
      }}
      .obs-verdict-meta {{ color:#888;font-size:14px;line-height:1.8; }}
      .obs-verdict-meta b {{ color:#ccc; }}
      .obs-prob-track {{
        margin:16px auto 4px;max-width:400px;
        background:#1c1c1c;border-radius:6px;height:6px;overflow:hidden;
      }}
      .obs-prob-fill {{
        height:100%;border-radius:6px;background:{score_color(prob)};
        width:{prob*100:.1f}%;
      }}
      .obs-prob-legend {{ font-size:11px;color:#555;text-align:center; }}
    </style>
    <div class="obs-verdict-card">
      <div class="obs-verdict-label">{verdict}</div>
      <div class="obs-verdict-meta">
        Deepfake probability: <b>{prob:.1%}</b> &nbsp;·&nbsp;
        Verdict confidence: <b>{conf:.0%}</b> &nbsp;·&nbsp;
        {media} &nbsp;·&nbsp; {pt:.2f}s
      </div>
      <div class="obs-prob-track"><div class="obs-prob-fill"></div></div>
      <div class="obs-prob-legend">{prob*100:.1f}% deepfake probability</div>
    </div>
    """, unsafe_allow_html=True)

    for w in result.get("warnings", []):
        st.warning(w)

    # ── Agent breakdown ──────────────────────────────────────────────────────
    agents  = result.get("agent_results", [])
    weights = result.get("fusion_weights", {})

    if agents:
        st.markdown("""
        <style>
          .obs-section {{ background:#282828;border-radius:16px;padding:22px 24px;margin-bottom:14px; }}
          .obs-section-title {{ font-family:Georgia,serif;font-size:17px;color:#888;
                                margin-bottom:16px;font-weight:400; }}
          .obs-agent-row {{ display:flex;align-items:center;gap:12px;
                            padding:9px 0;border-bottom:1px solid #1e1e1e; }}
          .obs-agent-row:last-child {{ border-bottom:none; }}
          .obs-agent-name {{ flex:1;font-size:13px;color:#999; }}
          .obs-bar-wrap {{ width:120px;height:4px;background:#1c1c1c;
                           border-radius:2px;overflow:hidden; }}
          .obs-bar-fill {{ height:100%;border-radius:2px; }}
          .obs-score-val {{ font-size:12px;color:#666;width:44px;text-align:right; }}
        </style>
        """, unsafe_allow_html=True)

        rows_html = ""
        for a in agents:
            name  = a.get("agent_name", "Agent")
            ran   = a.get("ran", True)
            score = a.get("score", 0.5)
            color = score_color(score)
            if not ran:
                rows_html += f'<div class="obs-agent-row"><span class="obs-agent-name" style="opacity:.4;font-style:italic">{name} — skipped</span></div>'
            else:
                rows_html += (
                    f'<div class="obs-agent-row">'
                    f'<span class="obs-agent-name">{name}</span>'
                    f'<div class="obs-bar-wrap"><div class="obs-bar-fill" style="width:{score*100:.0f}%;background:{color}"></div></div>'
                    f'<span class="obs-score-val">{score*100:.1f}%</span>'
                    f'</div>'
                )

        weight_rows = ""
        for sig, w in weights.items():
            label = sig.replace("_score","").replace("_"," ").title()
            weight_rows += (
                f'<div class="obs-agent-row">'
                f'<span class="obs-agent-name">{label}</span>'
                f'<div class="obs-bar-wrap"><div class="obs-bar-fill" style="width:{w*100:.0f}%;background:#7d7d7d"></div></div>'
                f'<span class="obs-score-val">{w*100:.0f}%</span>'
                f'</div>'
            )

        st.markdown(f"""
        <div class="obs-section">
          <div class="obs-section-title">Agent Signal Breakdown</div>
          {rows_html}
        </div>
        <div class="obs-section">
          <div class="obs-section-title">Fusion Weights</div>
          <div style="font-size:11px;color:#555;margin-bottom:12px;">
            Quantum-inspired calibration (PennyLane — classical simulation only)
          </div>
          {weight_rows}
        </div>
        """, unsafe_allow_html=True)

    # ── Rationale ────────────────────────────────────────────────────────────
    rationale = result.get("rationale", "")
    if rationale:
        st.markdown(f"""
        <div class="obs-section">
          <div class="obs-section-title">Forensic Rationale</div>
          <div style="font-size:14px;color:#888;line-height:1.8;">{rationale.replace(chr(10),'<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Download ─────────────────────────────────────────────────────────────
    st.download_button(
        label="Download Report (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"obscuro_report.json",
        mime="application/json",
    )

    st.markdown("""
    <div style="background:#1a1610;border-left:3px solid #6b4a10;border-radius:8px;
                padding:12px 16px;font-size:12px;color:#777;line-height:1.7;margin-top:12px;">
      <b>Important:</b> Deepfake detection is probabilistic. This tool is for forensic research
      only and is <em>not</em> legally admissible evidence.
    </div>
    """, unsafe_allow_html=True)


# ── Upload card ───────────────────────────────────────────────────────────────
import base64
from pathlib import Path

cursor_path = Path(__file__).parent.parent / "app" / "static" / "cursor.png"
cursor_b64 = ""
if cursor_path.exists():
    cursor_b64 = base64.b64encode(cursor_path.read_bytes()).decode()

st.markdown(f"""
<style>
  /* Style the Streamlit file uploader to look like the design card */
  [data-testid="stFileUploader"] {{
    background: #282828;
    border-radius: 20px;
    padding: 28px 36px 24px;
    border: none;
    max-width: 318px;
    margin: 0 auto 32px;
  }}
  [data-testid="stFileUploader"] section {{
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
  }}
  [data-testid="stFileUploader"] section > div {{
    background: transparent !important;
    border: none !important;
  }}
  [data-testid="stFileUploaderDropzoneInstructions"] {{
    display: none !important;
  }}
  [data-testid="stFileDropzoneInstructions"] {{ display: none !important; }}
  /* Upload button */
  [data-testid="stFileUploader"] button,
  [data-testid="stBaseButton-secondary"] {{
    background: #1c1c1c !important;
    color: #aaa !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    width: 100% !important;
  }}
  [data-testid="stFileUploader"] button:hover {{
    background: #252525 !important;
    color: #ccc !important;
  }}
</style>

<!-- Custom card layout above the file uploader -->
<div style="display:flex;flex-direction:column;align-items:center;
            background:#282828;border-radius:20px;padding:28px 36px 16px;
            max-width:318px;margin:0 auto 0;">
  <div style="font-family:Georgia,'Times New Roman',serif;font-size:30px;
              font-weight:400;color:#fff;margin-bottom:14px;">Drag &amp; Drop</div>
  {"<img src='data:image/png;base64," + cursor_b64 + "' style='width:92px;height:92px;object-fit:contain;image-rendering:pixelated;margin-bottom:20px;display:block;' draggable='false'/>" if cursor_b64 else ""}
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed",
)

if uploaded is not None:
    file_bytes = uploaded.read()
    filename   = uploaded.name
    size_mb    = len(file_bytes) / 1024 / 1024

    is_image = any(filename.lower().endswith(e)
                   for e in [".jpg", ".jpeg", ".png", ".webp", ".bmp"])

    col_prev, col_meta = st.columns([1, 2])
    with col_prev:
        if is_image:
            st.image(file_bytes, use_container_width=True)
        else:
            st.video(file_bytes)
    with col_meta:
        st.markdown(f"""
        <div style="padding-top:8px;">
          <p style="font-size:13px;color:#888;line-height:1.9;">
            <b style="color:#bbb">File</b> {filename}<br>
            <b style="color:#bbb">Size</b> {size_mb:.2f} MB<br>
            <b style="color:#bbb">Type</b> {'Image' if is_image else 'Video'}
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    # Styled analyse button
    st.markdown("""
    <style>
      [data-testid="stBaseButton-primary"] {
        background: #282828 !important;
        color: #aaa !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
      }
      [data-testid="stBaseButton-primary"]:hover {
        background: #333 !important;
        color: #ddd !important;
      }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Run Forensic Analysis", type="primary", use_container_width=True):
        with st.spinner("Running multi-agent forensic analysis…"):
            result = call_analyse(file_bytes, filename)
        if result:
            render_verdict(result)
