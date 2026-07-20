"""Entry point for the Streamlit frontend."""
import subprocess
import sys
import os

if __name__ == "__main__":
    port = int(os.environ.get("UI_PORT", 8080))
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ])
