import asyncio
import logging

logger = logging.getLogger(__name__)


async def retry_call(func, retries: int = 3, delay: float = 2.0) -> dict:
    """
    Async retry wrapper.  Uses asyncio.sleep instead of time.sleep so the
    event loop is not blocked while waiting between attempts.

    Returns a dict with keys: story, retry_count, fallback_used.
    Raises Exception when all attempts are exhausted.
    """
    retry_count = 0

    for attempt in range(1, retries + 1):
        try:
            response = func()
            return {
                "story": response,
                "retry_count": retry_count,
                "fallback_used": False,
            }
        except Exception as e:
            retry_count += 1
            logger.warning("Attempt %d/%d failed: %s", attempt, retries, e)
            if attempt < retries:
                await asyncio.sleep(delay)

    raise Exception("Retry attempts exhausted.")