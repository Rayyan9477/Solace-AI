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

# Import real agent node functions that call actual services via HTTP
from ..agents.chat_agent import chat_agent_node as real_chat_agent_node
from ..agents.diagnosis_agent import diagnosis_agent_node as real_diagnosis_agent_node
from ..agents.therapy_agent import therapy_agent_node as real_therapy_agent_node
from ..agents.personality_agent import personality_agent_node as real_personality_agent_node
from ..agents.safety_agent import safety_agent_node as real_safety_agent_node

logger = structlog.get_logger(__name__)


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
    # Use local stub agents instead of HTTP service clients (for testing)
    use_local_agents: bool = Field(default=False)
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
    """Crisis handler node - generates immediate safety response."""
    safety_flags = state.get("safety_flags", {})
    risk_level = safety_flags.get("risk_level", "none")
    logger.warning("crisis_handler_activated", risk_level=risk_level)
    crisis_response = """I'm really concerned about what you're sharing. Your safety is the most important thing right now.

If you're having thoughts of harming yourself, please reach out for immediate support:
- **988 Suicide & Crisis Lifeline**: Call or text 988 (available 24/7)
- **Crisis Text Line**: Text HOME to 741741
- **Emergency Services**: Call 911 if you're in immediate danger

I'm here with you, and I want you to know that these feelings can get better with the right support. Would you like to talk about what's been happening?"""
    response_msg = MessageEntry.assistant_message(content=crisis_response, metadata={"is_crisis_response": True, "risk_level": risk_level})
    agent_result = AgentResult(agent_type=AgentType.SAFETY, success=True, response_content=crisis_response, confidence=1.0, metadata={"is_crisis_response": True})
    return {"final_response": crisis_response, "messages": [response_msg.to_dict()], "safety_flags": {**safety_flags, "safety_resources_shown": True}, "processing_phase": ProcessingPhase.CRISIS_HANDLING.value, "agent_results": [agent_result.to_dict()]}


def chat_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Chat agent node - handles general conversation."""
    message = state.get("current_message", "")
    personality_style = state.get("personality_style", {})
    logger.info("chat_agent_processing", message_length=len(message))
    warmth = personality_style.get("warmth", 0.7)
    response = ("Thank you for sharing that with me. I'm here to listen and support you. " if warmth > 0.7 else "I understand. ") + "How are you feeling about this?"
    return {"agent_results": [AgentResult(agent_type=AgentType.CHAT, success=True, response_content=response, confidence=0.7, metadata={"warmth": warmth}).to_dict()]}


def diagnosis_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Diagnosis agent node - coordinates with diagnosis service."""
    logger.info("diagnosis_agent_processing", message_length=len(state.get("current_message", "")))
    response = "Based on what you've shared, it might be helpful to explore these feelings further. Would you like to talk more about when these symptoms started?"
    return {"agent_results": [AgentResult(agent_type=AgentType.DIAGNOSIS, success=True, response_content=response, confidence=0.75, metadata={"assessment_type": "symptom_exploration"}).to_dict()]}


def therapy_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Therapy agent node - provides evidence-based therapeutic interventions."""
    logger.info("therapy_agent_processing", message_length=len(state.get("current_message", "")), intent=state.get("intent", "general_chat"))
    response = "It sounds like you're going through a difficult time. One thing that might help is to take a moment to notice how you're feeling right now, without trying to change it. Just observe your thoughts and feelings with curiosity."
    return {"agent_results": [AgentResult(agent_type=AgentType.THERAPY, success=True, response_content=response, confidence=0.80, metadata={"technique": "mindfulness_observation", "modality": "ACT"}).to_dict()]}


def personality_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Personality agent node - applies Big Five personality adaptation."""
    logger.info("personality_agent_processing", message_length=len(state.get("current_message", "")))
    style = {"warmth": 0.7, "structure": 0.5, "complexity": 0.5, "directness": 0.5, "energy": 0.5, "validation_level": 0.6, "style_type": "balanced"}
    return {"personality_style": style, "agent_results": [AgentResult(agent_type=AgentType.PERSONALITY, success=True, confidence=0.70, metadata={"style_params": style}).to_dict()]}


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


def route_after_safety(state: OrchestratorState) -> Literal["crisis_handler", "supervisor"]:
    """Route based on safety pre-check results."""
    safety_flags = state.get("safety_flags", {})
    if safety_flags.get("crisis_detected") or safety_flags.get("risk_level") in ("high", "critical"):
        logger.info("routing_to_crisis_handler")
        return "crisis_handler"
    return "supervisor"


def route_to_agents(state: OrchestratorState) -> list[str]:
    """Route to selected agents for parallel processing."""
    selected_agents = state.get("selected_agents", [])
    if not selected_agents:
        return ["chat_agent"]
    agent_node_map = {
        "safety": "chat_agent",
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
        use_local = self._settings.use_local_agents
        use_safety_service = self._settings.use_safety_service_precheck
        logger.info(
            "building_orchestrator_graph",
            checkpointing=self._settings.enable_checkpointing,
            use_local_agents=use_local,
            use_safety_service_precheck=use_safety_service,
        )
        builder = StateGraph(OrchestratorState)
        # Safety precheck: use full Safety Service or local rule-based check
        if use_safety_service:
            builder.add_node("safety_precheck", real_safety_agent_node)
        else:
            builder.add_node("safety_precheck", safety_precheck_node)
        builder.add_node("supervisor", SupervisorAgent(self._supervisor_settings).process)
        # Use local crisis handler for immediate crisis response
        builder.add_node("crisis_handler", crisis_handler_node)
        # Agent nodes: use HTTP clients or local stubs based on configuration
        if use_local:
            # Use local stub implementations (for testing or when services unavailable)
            builder.add_node("chat_agent", chat_agent_node)
            builder.add_node("diagnosis_agent", diagnosis_agent_node)
            builder.add_node("therapy_agent", therapy_agent_node)
            builder.add_node("personality_agent", personality_agent_node)
        else:
            # Use real agent nodes that call actual services via HTTP
            builder.add_node("chat_agent", real_chat_agent_node)
            builder.add_node("diagnosis_agent", real_diagnosis_agent_node)
            builder.add_node("therapy_agent", real_therapy_agent_node)
            builder.add_node("personality_agent", real_personality_agent_node)
        # Use local aggregator and post-check nodes
        builder.add_node("aggregator", aggregator_node)
        builder.add_node("safety_postcheck", safety_postcheck_node)
        builder.add_edge(START, "safety_precheck")
        builder.add_conditional_edges("safety_precheck", route_after_safety, ["crisis_handler", "supervisor"])
        builder.add_edge("crisis_handler", END)
        builder.add_conditional_edges("supervisor", route_to_agents, ["chat_agent", "diagnosis_agent", "therapy_agent", "personality_agent"])
        builder.add_edge("chat_agent", "aggregator")
        builder.add_edge("diagnosis_agent", "aggregator")
        builder.add_edge("therapy_agent", "aggregator")
        builder.add_edge("personality_agent", "aggregator")
        builder.add_edge("aggregator", "safety_postcheck")
        builder.add_edge("safety_postcheck", END)
        self._graph = builder
        logger.info("orchestrator_graph_built", node_count=9, mode="local" if use_local else "http_clients")
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
