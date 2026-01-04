from fastapi import FastAPI
from api.route import router as api_router
from core.logging import setup_logging
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="Keystroke Stress Detection API",
        description="Stress-level detection using keystroke dynamics",
        version="1.0.0"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      
        allow_credentials=True,
        allow_methods=["*"],       
        allow_headers=["*"],
    )
    app.include_router(api_router)

    return app

app = create_app()
