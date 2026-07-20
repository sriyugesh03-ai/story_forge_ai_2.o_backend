import time


class EvaluationService:

    def __init__(self):

        self.request_start = None
        self.retrieval_start = None
        self.llm_start = None
        self.retrieval_time = 0
        self.llm_time = 0

    # -------------------------
    # Request Timer
    # -------------------------

    def start_request(self):

        self.request_start = time.perf_counter()

    # -------------------------
    # Retrieval Timer
    # -------------------------

    def start_retrieval(self):

        self.retrieval_start = time.perf_counter()

    def end_retrieval(self):

        self.retrieval_time = (

            time.perf_counter()

            - self.retrieval_start

        )

    # -------------------------
    # LLM Timer
    # -------------------------

    def start_llm(self):

        self.llm_start = time.perf_counter()

    def end_llm(self):

        self.llm_time = (

            time.perf_counter()

            - self.llm_start

        )

    # -------------------------
    # Build Metrics
    # -------------------------

    def build_metrics(

        self,

        model_used,

        retrieved_chunks,

        retry_count,

        fallback_used,

        sources

    ):

        total_time = (

            time.perf_counter()

            - self.request_start

        )

        return {

            "model_used": model_used,

            "retrieved_chunks": retrieved_chunks,

            "retrieval_time_ms":

                round(self.retrieval_time * 1000, 2),

            "llm_generation_time_ms":

                round(self.llm_time * 1000, 2),

            "total_latency_ms":

                round(total_time * 1000, 2),

            "retry_count": retry_count,

            "fallback_used": fallback_used,

            "sources": sources

        }