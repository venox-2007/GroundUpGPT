"""Entrypoint script to start the GroundUpGPT FastAPI inference backend."""

from pathlib import Path
import sys
import uvicorn

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import get_settings


def main() -> None:
    settings = get_settings()
    print("=" * 65)
    print(" GroundUpGPT — Local Inference Backend")
    print(f" Binding to: http://{settings.host}:{settings.port}")
    print(f" Swagger UI: http://{settings.host}:{settings.port}/docs")
    print("=" * 65)

    uvicorn.run(
        "backend.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
