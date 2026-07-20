import time
import traceback


def retry_call(
    func,
    retries=3,
    delay=2
):

    retry_count = 0

    for attempt in range(1, retries + 1):

        try:

            response = func()

            return {

                "story": response,

                "retry_count": retry_count,

                "fallback_used": False

            }

        except Exception as e:

            retry_count += 1

            print(
                f"[Retry] Attempt {attempt}/{retries} failed: {e}"
            )

            traceback.print_exc()

            if attempt < retries:

                time.sleep(delay)

    raise Exception("Retry attempts exhausted.")