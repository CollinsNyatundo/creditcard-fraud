from fastapi import Request
from slowapi import Limiter


def get_api_key(request: Request) -> str:
    """Extract API key for rate limiting."""
    return request.headers.get("X-API-Key", "")


limiter = Limiter(key_func=get_api_key)
