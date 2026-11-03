"""Solace-AI source libraries.

The shared packages under ``src/`` (``solace_common``, ``solace_events``, ``solace_security``,
``solace_infrastructure``, ``solace_testing``) are imported as top-level modules, for example
``from solace_events.config import KafkaSettings``, with ``src/`` on ``PYTHONPATH``. Do not import
them as ``src.solace_*``: that loads a second copy of each module, with its own classes and
singletons (MVP-270). This package therefore imports nothing.
"""
