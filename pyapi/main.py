import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Fly.io sets PORT automatically
    uvicorn.run("pyapi.api:app", host="0.0.0.0", port=port)
