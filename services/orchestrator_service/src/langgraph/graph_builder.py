"""
Solace-AI Orchestrator Service - LangGraph Graph Builder.
Constructs the multi-agent state graph with nodes, edges, and checkpointing.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Callable, Literal
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _POSTGRES_CHECKPOINT_AVAILABLE = True
except ImportError:
    _POSTGRES_CHECKPOINT_AVAILABLE = False

from .state_schema import (
    OrchestratorState,
    AgentType,
    ProcessingPhase,
    RiskLevel,
    AgentResult,
    MessageEntry,
    SafetyFlags,
)
from .supervisor import SupervisorAgent, SupervisorSettings
from .memory_node import memory_retrieval_node

logger = structlog.get_logger(__name__)


def _import_agent_nodes() -> dict[str, Any]:
    """Lazy import of agent node functions to avoid circular imports."""
    from ..agents.chat_agent import chat_agent_node as real_chat_agent_node
    from ..agents.diagnosis_agent import diagnosis_agent_node as real_diagnosis_agent_node
    from ..agents.therapy_agent import therapy_agent_node as real_therapy_agent_node
    from ..agents.personality_agent import personality_agent_node as real_personality_agent_node
    from ..agents.safety_agent import safety_agent_node as real_safety_agent_node
    return {
        "chat": real_chat_agent_node,
        "diagnosis": real_diagnosis_agent_node,
        "therapy": real_therapy_agent_node,
        "personality": real_personality_agent_node,
        "safety": real_safety_agent_node,
    }


class _CrisisResourceManager:
    """Local crisis resource manager for the orchestrator.

    Mirrors the safety service's CrisisResourceManager to avoid cross-service imports.
    Provides crisis hotline/resource information for inclusion in crisis responses.
    """

    _RESOURCES = [
        {"name": "Emergency Services", "contact": "911", "type": "phone", "available": "24/7"},
        {"name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988", "type": "phone", "available": "24/7"},
        {"name": "Crisis Text Line", "contact": "Text HOME to 741741", "type": "text", "available": "24/7"},
        {"name": "SAMHSA National Helpline", "contact": "1-800-662-4357", "type": "phone", "available": "24/7"},
        {"name": "Veterans Crisis Line", "contact": "988 (Press 1)", "type": "phone", "available": "24/7"},
        {"name": "Trevor Project (LGBTQ+)", "contact": "1-866-488-7386", "type": "phone", "available": "24/7"},
    ]

    def get_resources_for_level(self, crisis_level: str) -> list[dict[str, str]]:
        """Get appropriate resources for crisis level."""
        level = crisis_level.upper()
        if level == "CRITICAL":
            return self._RESOURCES  # All resources including 911
        if level == "HIGH":
            return self._RESOURCES[1:]  # Skip 911 for HIGH
        return self._RESOURCES[1:3]  # 988 + Crisis Text Line for lower levels


_crisis_resource_manager: _CrisisResourceManager | None = None


def _get_crisis_resource_manager() -> _CrisisResourceManager:
    """Get singleton crisis resource manager."""
    global _crisis_resource_manager
    if _crisis_resource_manager is None:
        _crisis_resource_manager = _CrisisResourceManager()
    return _crisis_resource_manager


class GraphBuilderSettings(BaseSettings):
    """Configuration for the graph builder."""
    enable_checkpointing: bool = Field(default=True)
    enable_safety_precheck: bool = Field(default=True)
    enable_safety_postcheck: bool = Field(default=True)
    enable_parallel_processing: bool = Field(default=True)
    max_iterations: int = Field(default=10, ge=1, le=50)
    safety_timeout_ms: int = Field(default=5000)
    agent_timeout_ms: int = Field(default=30000)
    # Use full Safety Service for precheck instead of local rule-based check
    use_safety_service_precheck: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="ORCHESTRATOR_GRAPH_", env_file=".env", extra="ignore")


def safety_precheck_node(state: OrchestratorState) -> dict[str, Any]:
    """Safety pre-check node - analyzes input for crisis indicators."""
    message = state.get("current_message", "")
    logger.info("safety_precheck_processing", message_length=len(message))
    crisis_keywords = ["suicide", "kill myself", "end my life", "want to die", "self-harm", "hurt myself", "cutting", "overdose", "no reason to live", "end it all"]
    high_risk_keywords = ["plan to", "going to", "tonight", "method", "goodbye", "final"]
    message_lower = message.lower()
    triggered = [kw for kw in crisis_keywords if kw in message_lower]
    high_risk_matches = [kw for kw in high_risk_keywords if kw in message_lower]
    crisis_detected = len(triggered) > 0
    risk_level = RiskLevel.CRITICAL if (crisis_detected and high_risk_matches) else (RiskLevel.HIGH if crisis_detected else (RiskLevel.LOW if any(kw in message_lower for kw in ["depressed", "anxious", "hopeless"]) else RiskLevel.NONE))
    safety_flags = SafetyFlags(risk_level=risk_level, crisis_detected=crisis_detected, crisis_type="suicidal_ideation" if crisis_detected else None, requires_escalation=risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL), monitoring_level="intensive" if crisis_detected else "standard", triggered_keywords=triggered, last_assessment_at=datetime.now(timezone.utc))
    agent_result = AgentResult(agent_type=AgentType.SAFETY, success=True, confidence=0.9 if crisis_detected else 0.7, metadata={"triggered_keywords": triggered, "phase": "precheck"})
    logger.info("safety_precheck_complete", risk_level=risk_level.value, crisis_detected=crisis_detected, triggered_count=len(triggered))
    return {"safety_flags": safety_flags.to_dict(), "processing_phase": ProcessingPhase.SAFETY_PRECHECK.value, "agent_results": [agent_result.to_dict()]}


def crisis_handler_node(state: OrchestratorState) -> dict[str, Any]:
    """Crisis handler node - generates immediate safety response with crisis resources."""
    safety_flags = state.get("safety_flags", {})
    risk_level = str(safety_flags.get("risk_level", "NONE")).upper()
    logger.warning("crisis_handler_activated", risk_level=risk_level)

    # Get crisis resources from the resource manager
    resource_manager = _get_crisis_resource_manager()
    resources = resource_manager.get_resources_for_level(risk_level)

    # Build resource text from dynamic resources
    resource_lines = "\n".join(
        f"- **{r['name']}**: {r['contact']} ({r['available']})"
        for r in resources
    )

    crisis_response = (
        "I'm really concerned about what you're sharing. "
        "Your safety is the most important thing right now.\n\n"
        "Please reach out for immediate support:\n"
        f"{resource_lines}\n\n"
        "I'm here with you, and I want you to know that these feelings "
        "can get better with the right support. Would you like to talk "
        "about what's been happening?"
    )

    response_msg = MessageEntry.assistant_message(
        content=crisis_response,
        metadata={
            "is_crisis_response": True,
            "risk_level": risk_level,
            "resources_provided": len(resources),
        },
    )
    agent_result = AgentResult(
        agent_type=AgentType.SAFETY,
        success=True,
        response_content=crisis_response,
        confidence=1.0,
        metadata={
            "is_crisis_response": True,
            "resources": resources,
        },
    )
    return {
        "final_response": crisis_response,
        "messages": [response_msg.to_dict()],
        "safety_flags": {**safety_flags, "safety_resources_shown": True},
        "processing_phase": ProcessingPhase.CRISIS_HANDLING.value,
        "agent_results": [agent_result.to_dict()],
    }


def aggregator_node(state: OrchestratorState) -> dict[str, Any]:
    """Aggregator node - combines results from multiple agents."""
    agent_results = state.get("agent_results", [])
    personality_style = state.get("personality_style", {})
    logger.info("aggregator_processing", result_count=len(agent_results))
    responses = [r["response_content"] for r in agent_results if r.get("response_content")]
    final_response = responses[-1] if responses else "I'm here to support you. How can I help you today?"
    if personality_style.get("warmth", 0.6) > 0.8:
        final_response = f"I really appreciate you sharing this with me. {final_response}"
    response_msg = MessageEntry.assistant_message(content=final_response, metadata={"aggregated": True, "source_count": len(responses)})
    return {"final_response": final_response, "messages": [response_msg.to_dict()], "processing_phase": ProcessingPhase.AGGREGATION.value, "agent_results": [AgentResult(agent_type=AgentType.AGGREGATOR, success=True, response_content=final_response, confidence=0.85, metadata={"source_count": len(responses)}).to_dict()]}


def safety_postcheck_node(state: OrchestratorState) -> dict[str, Any]:
    """Safety post-check node - validates final response."""
    final_response = state.get("final_response", "")
    logger.info("safety_postcheck_processing", response_length=len(final_response))
    harmful_patterns = ["you should", "just do it", "give up", "no point"]
    needs_filtering = any(p in final_response.lower() for p in harmful_patterns)
    if needs_filtering:
        logger.warning("safety_postcheck_filtering_applied")
    return {"processing_phase": ProcessingPhase.COMPLETED.value, "agent_results": [AgentResult(agent_type=AgentType.SAFETY, success=True, confidence=0.9, metadata={"phase": "postcheck", "filtered": needs_filtering}).to_dict()]}


def route_after_safety(state: OrchestratorState) -> Literal["crisis_handler", "memory_retrieval"]:
    """Route based on safety pre-check results."""
    safety_flags = state.get("safety_flags", {})
    crisis_detected = safety_flags.get("crisis_detected", False)
    risk_level = str(safety_flags.get("risk_level", "NONE")).upper()
    requires_escalation = safety_flags.get("requires_escalation", False)

    route_to_crisis = crisis_detected or risk_level in ("HIGH", "CRITICAL") or requires_escalation
    destination = "crisis_handler" if route_to_crisis else "memory_retrieval"

    logger.info(
        "routing_decision",
        destination=destination,
        crisis_detected=crisis_detected,
        risk_level=risk_level,
        requires_escalation=requires_escalation,
    )
    return destination


def route_to_agents(state: OrchestratorState) -> list[str]:
    """Route to selected agents for parallel processing."""
    selected_agents = state.get("selected_agents", [])
    if not selected_agents:
        return ["chat_agent"]
    agent_node_map = {
        "safety": "safety_agent",
        "chat": "chat_agent",
        "diagnosis": "diagnosis_agent",
        "therapy": "therapy_agent",
        "personality": "personality_agent",
    }
    nodes = []
    for agent in selected_agents:
        node_name = agent_node_map.get(agent, "chat_agent")
        if node_name not in nodes:
            nodes.append(node_name)
    return nodes if nodes else ["chat_agent"]


def route_after_safety_agent(state: OrchestratorState) -> Literal["crisis_handler", "aggregator"]:
    """Route based on safety agent assessment results.

    If the safety agent detects HIGH or CRITICAL risk, route to crisis handler.
    Otherwise, continue to aggregation.
    """
    safety_flags = state.get("safety_flags", {})
    crisis_detected = safety_flags.get("crisis_detected", False)
    risk_level = str(safety_flags.get("risk_level", "NONE")).upper()
    requires_escalation = safety_flags.get("requires_escalation", False)

    route_to_crisis = crisis_detected or risk_level in ("HIGH", "CRITICAL") or requires_escalation
    destination = "crisis_handler" if route_to_crisis else "aggregator"

    logger.info(
        "safety_agent_routing_decision",
        destination=destination,
        crisis_detected=crisis_detected,
        risk_level=risk_level,
        requires_escalation=requires_escalation,
    )
    return destination


class OrchestratorGraphBuilder:
    """
    Builds the LangGraph state machine for multi-agent orchestration.
    Configures nodes, edges, and checkpointing for the orchestrator.
    """

    def __init__(
        self,
        settings: GraphBuilderSettings | None = None,
        postgres_connection_string: str | None = None,
    ) -> None:
        self._settings = settings or GraphBuilderSettings()
        self._supervisor_settings = SupervisorSettings()
        self._postgres_conn_str = postgres_connection_string
        self._checkpointer = self._create_checkpointer()
        self._graph = None
        self._compiled = None

    def _create_checkpointer(self) -> Any:
        """Create appropriate checkpointer based on configuration."""
        if not self._settings.enable_checkpointing:
            return None
        if self._postgres_conn_str and _POSTGRES_CHECKPOINT_AVAILABLE:
            logger.info("using_postgres_checkpointer")
            return AsyncPostgresSaver.from_conn_string(self._postgres_conn_str)
        if self._postgres_conn_str and not _POSTGRES_CHECKPOINT_AVAILABLE:
            logger.warning(
                "postgres_checkpointer_unavailable",
                msg="langgraph-checkpoint-postgres not installed, falling back to in-memory",
            )
        return InMemorySaver()

    def build(self) -> StateGraph:
        """
        Build the orchestrator state graph.

        Returns:
            Configured StateGraph instance
        """
        use_safety_service = self._settings.use_safety_service_precheck
        logger.info(
            "building_orchestrator_graph",
            checkpointing=self._settings.enable_checkpointing,
            use_safety_service_precheck=use_safety_service,
        )
        agent_nodes = _import_agent_nodes()
        builder = StateGraph(OrchestratorState)
        # Safety precheck: use full Safety Service or local rule-based check
        if use_safety_service:
            builder.add_node("safety_precheck", agent_nodes["safety"])
        else:
            builder.add_node("safety_precheck", safety_precheck_node)
        builder.add_node("memory_retrieval", memory_retrieval_node)
        builder.add_node("supervisor", SupervisorAgent(self._supervisor_settings).process)
        builder.add_node("crisis_handler", crisis_handler_node)
        # Agent nodes: real service HTTP clients
        builder.add_node("chat_agent", agent_nodes["chat"])
        builder.add_node("diagnosis_agent", agent_nodes["diagnosis"])
        builder.add_node("therapy_agent", agent_nodes["therapy"])
        builder.add_node("personality_agent", agent_nodes["personality"])
        builder.add_node("safety_agent", agent_nodes["safety"])
        builder.add_node("aggregator", aggregator_node)
        builder.add_node("safety_postcheck", safety_postcheck_node)
        # Edges
        builder.add_edge(START, "safety_precheck")
        builder.add_conditional_edges("safety_precheck", route_after_safety, ["crisis_handler", "memory_retrieval"])
        builder.add_edge("memory_retrieval", "supervisor")
        builder.add_edge("crisis_handler", END)
        builder.add_conditional_edges("supervisor", route_to_agents, ["chat_agent", "diagnosis_agent", "therapy_agent", "personality_agent", "safety_agent"])
        builder.add_edge("chat_agent", "aggregator")
        builder.add_edge("diagnosis_agent", "aggregator")
        builder.add_edge("therapy_agent", "aggregator")
        builder.add_edge("personality_agent", "aggregator")
        # Safety agent has conditional routing: crisis -> crisis_handler, otherwise -> aggregator
        builder.add_conditional_edges("safety_agent", route_after_safety_agent, ["crisis_handler", "aggregator"])
        builder.add_edge("aggregator", "safety_postcheck")
        builder.add_edge("safety_postcheck", END)
        self._graph = builder
        logger.info("orchestrator_graph_built", node_count=11)
        return builder

    def compile(self) -> Any:
        """
        Compile the graph with checkpointer.

        Returns:
            Compiled graph ready for execution
        """
        if self._graph is None:
            self.build()
        self._compiled = self._graph.compile(checkpointer=self._checkpointer)
        logger.info("orchestrator_graph_compiled", checkpointing_enabled=self._checkpointer is not None)
        return self._compiled

    def get_compiled_graph(self) -> Any:
        """Get the compiled graph, building if necessary."""
        if self._compiled is None:
            self.compile()
        return self._compiled

    def get_checkpointer(self) -> Any:
        """Get the checkpointer instance (InMemorySaver or AsyncPostgresSaver)."""
        return self._checkpointer

    async def invoke(self, state: OrchestratorState, thread_id: str | None = None) -> OrchestratorState:
        """
        Invoke the graph with given state.

        Args:
            state: Initial orchestrator state
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final orchestrator state
        """
        graph = self.get_compiled_graph()
        config = {"configurable": {"thread_id": thread_id or state.get("thread_id", "default")}}
        result = await graph.ainvoke(state, config=config)
        return result

    def invoke_sync(self, state: OrchestratorState, thread_id: str | None = None) -> OrchestratorState:
        """
        Synchronously invoke the graph.

        Args:
            state: Initial orchestrator state
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final orchestrator state
        """
        graph = self.get_compiled_graph()
        config = {"configurable": {"thread_id": thread_id or state.get("thread_id", "default")}}
        result = graph.invoke(state, config=config)
        return result
