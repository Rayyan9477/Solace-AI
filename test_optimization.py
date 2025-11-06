"""
Test script for verifying optimization module integration.

This script tests the optimization components to ensure they are properly
integrated with the main application.
"""

import asyncio
import time
from typing import Dict, Any

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

async def test_optimization_integration():
    """Test optimization module integration"""
    print("=" * 60)
    print("Testing Multi-Agent System Optimization")
    print("=" * 60)

    # Test 1: Check if optimization module is available
    print("\n1. Checking optimization module availability...")
    try:
        from src.optimization import (
            AgentPerformanceProfiler,
            OptimizedAgentOrchestrator,
            ContextWindowManager,
            AgentResultCache
        )
        print("[OK] Optimization module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import optimization module: {e}")
        return

    # Test 2: Check if optimization is enabled in orchestration
    print("\n2. Checking orchestration optimization...")
    try:
        from src.agents.orchestration import OPTIMIZATION_ENABLED
        if OPTIMIZATION_ENABLED:
            print("[OK] Optimization enabled in orchestration module")
        else:
            print("[WARN] Optimization available but not enabled")
    except ImportError:
        print("[FAIL] Could not check orchestration optimization status")

    # Test 3: Check optimization configuration
    print("\n3. Checking optimization configuration...")
    try:
        from src.config import OptimizationConfig, OPTIMIZATION_CONFIG_AVAILABLE
        if OPTIMIZATION_CONFIG_AVAILABLE:
            summary = OptimizationConfig.get_optimization_summary()
            print("[OK] Optimization configuration loaded:")
            print(f"   - Optimization: {'Enabled' if summary['optimization_enabled'] else 'Disabled'}")
            print(f"   - Parallel execution: {'Yes' if summary['features']['parallel_execution'] else 'No'}")
            print(f"   - Caching: {'Yes' if summary['features']['caching'] else 'No'}")
            print(f"   - Context compression: {'Yes' if summary['features']['context_compression'] else 'No'}")
            print(f"   - Performance monitoring: {'Yes' if summary['features']['performance_monitoring'] else 'No'}")
            print(f"   - Parallel workers: {summary['parallel_workers']}")
            print(f"   - Cache TTL: {summary['cache_ttl']}s")
        else:
            print("[WARN] Optimization configuration not available")
    except ImportError as e:
        print(f"[FAIL] Could not load optimization config: {e}")

    # Test 4: Test performance profiler
    print("\n4. Testing performance profiler...")
    try:
        profiler = AgentPerformanceProfiler()

        # Simulate agent execution
        test_agent = "test_agent"
        test_input = {"message": "Test message"}
        test_output = {"response": "Test response"}

        # Profile a simulated execution
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulate work
        execution_time = time.time() - start_time

        # Record metrics
        profiler.record_execution(
            agent_name=test_agent,
            execution_time=execution_time,
            token_count=100,
            cache_hit=False,
            context_size=50
        )

        # Get recommendations
        recommendations = profiler.get_optimization_recommendations()
        print("[OK] Performance profiler working")
        if recommendations:
            print(f"   Recommendations: {len(recommendations)} suggestions")
    except Exception as e:
        print(f"[FAIL] Performance profiler test failed: {e}")

    # Test 5: Test context optimization
    print("\n5. Testing context optimization...")
    try:
        context_manager = ContextWindowManager()

        test_context = {
            "memory": [{"content": "Previous conversation " * 100}],
            "emotion": {"primary_emotion": "anxious", "intensity": 0.7},
            "safety": {"risk_level": "low", "safe": True}
        }

        optimized = context_manager.get_optimized_context(
            agent_name="chat_agent",
            full_context=test_context,
            query="How are you feeling?"
        )

        original_size = len(str(test_context))
        optimized_size = len(str(optimized))
        compression_ratio = (1 - optimized_size / original_size) * 100

        print("[OK] Context optimization working")
        print(f"   Original size: {original_size} chars")
        print(f"   Optimized size: {optimized_size} chars")
        print(f"   Compression: {compression_ratio:.1f}%")
    except Exception as e:
        print(f"[FAIL] Context optimization test failed: {e}")

    # Test 6: Test caching
    print("\n6. Testing result caching...")
    try:
        cache = AgentResultCache()

        # Test cache operations
        test_key_hash = cache.compute_input_hash(
            {"message": "test"},
            {"context": "test"}
        )

        # Set cache
        cache.set("test_agent", test_key_hash, {"result": "cached"})

        # Get from cache
        cached_result = cache.get("test_agent", test_key_hash)

        if cached_result:
            print("[OK] Caching system working")
            stats = cache.get_cache_stats()
            print(f"   Cache stats: {stats['hit_count']} hits, {stats['miss_count']} misses")
        else:
            print("[WARN] Cache set but retrieval failed")
    except Exception as e:
        print(f"[FAIL] Caching test failed: {e}")

    # Test 7: Check parallel execution groups
    print("\n7. Testing parallel execution configuration...")
    try:
        from src.config import OptimizationConfig
        parallel_groups = OptimizationConfig.get_parallel_groups()

        if parallel_groups:
            print("[OK] Parallel execution groups configured:")
            for group_name, agents in parallel_groups.items():
                print(f"   - {group_name}: {', '.join(agents)}")
        else:
            print("[WARN] No parallel execution groups configured")
    except Exception as e:
        print(f"[FAIL] Could not check parallel groups: {e}")

    # Test 8: Test optimized model configuration
    print("\n8. Testing optimized model configuration...")
    try:
        from src.config.settings import AppConfig

        if hasattr(AppConfig, 'get_optimized_model_config'):
            # Test different agent configurations
            agents_to_test = ["chat_agent", "emotion_agent", "search_agent"]

            for agent in agents_to_test:
                config = AppConfig.get_optimized_model_config(agent)
                print(f"[OK] {agent} config:")
                print(f"   - Model: {config.get('model', 'default')}")
                print(f"   - Temperature: {config.get('temperature', 0.7)}")
                print(f"   - Max tokens: {config.get('max_tokens', 1000)}")
        else:
            print("[WARN] Optimized model configuration not available")
    except Exception as e:
        print(f"[FAIL] Model configuration test failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Optimization Integration Test Complete")
    print("=" * 60)


async def test_workflow_performance():
    """Test workflow execution with optimization"""
    print("\n" + "=" * 60)
    print("Testing Workflow Performance")
    print("=" * 60)

    try:
        # Create a minimal test setup
        test_agents = {}

        # Create mock agent modules
        class MockAgent:
            async def process(self, input_data: Dict[str, Any], context: Dict[str, Any] = None):
                # Simulate some processing time
                await asyncio.sleep(0.1)
                return {
                    "response": f"Processed by {self.__class__.__name__}",
                    "status": "success"
                }

        # Create test agents
        for agent_name in ["emotion_agent", "personality_agent", "safety_agent"]:
            test_agents[agent_name] = MockAgent()

        # Test with OptimizedAgentOrchestrator
        from src.optimization.optimized_orchestrator import OptimizedAgentOrchestrator

        orchestrator = OptimizedAgentOrchestrator(test_agents)

        # Execute a test workflow
        print("\nExecuting optimized workflow...")
        start_time = time.time()

        result = await orchestrator.execute_optimized_workflow(
            workflow_id="test_workflow",
            input_data="Test input",
            context={},
            session_id="test_session"
        )

        execution_time = time.time() - start_time

        print(f"[OK] Workflow executed in {execution_time:.2f}s")
        print(f"   Status: {result['status']}")
        if 'optimization_report' in result:
            report = result['optimization_report']
            print(f"   Speedup: {report['execution_metrics'].get('actual_speedup', 1):.1f}x")
            print(f"   Cache hit rate: {report['cache_performance'].get('hit_rate', 0):.1%}")

    except Exception as e:
        print(f"[FAIL] Workflow performance test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting optimization integration tests...\n")

    # Run tests
    asyncio.run(test_optimization_integration())
    asyncio.run(test_workflow_performance())

    print("\nTests complete!")