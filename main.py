"""Application entrypoint for local development.

Running `python main.py` will start uvicorn against `app.main:app`.
"""
import os
import uvicorn


def _should_reload() -> bool:
    value = os.getenv("UVICORN_RELOAD", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=_should_reload(),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()