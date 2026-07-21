"""Entry point for the Obscuro Deepimage FastAPI backend + UI server."""
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("API_PORT", 8000)))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
