# procurement_system/agents/mixins.py

import time
import logging


class TracingMixin:
    """Replaced by LangSmith auto-instrumentation. Kept for MRO compatibility."""
    pass


# class TracingMixin:
#     def trace(self, operation: str, **kwargs):
#         start = time.time()
#         logging.info(f"[TRACE] {self.__class__.__name__} | {operation} | input: {kwargs}")
#         return start

#     def trace_end(self, start: float, operation: str, result=None):
#         elapsed = time.time() - start
#         logging.info(f"[TRACE] {self.__class__.__name__} | {operation} | {elapsed:.2f}s | result: {result}")
