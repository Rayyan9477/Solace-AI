"""
Solace-AI Orchestrator Service - Response Module.

Response generation, styling, and safety wrapping components were removed
as dead code — they were never wired into the LangGraph pipeline.
The graph handles response assembly directly via aggregator → safety_postcheck.
"""
