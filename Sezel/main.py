"""Main entry point for Sezel Phase 1."""

# When this module is executed directly (python Sezel\main.py) the
# package context for relative imports is not set which raises
# "attempted relative import with no known parent package". Set up
# sys.path and __package__ so relative imports work in both
# `python -m Sezel.main` and `python Sezel\main.py` execution modes.
import sys
import pathlib

from networkx.algorithms import dominance

if globals().get("__package__") is None and __name__ == "__main__":
    # Insert the repository root (parent of the package dir) into sys.path
    package_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    # Set package name so relative imports behave correctly
    __package__ = package_dir.name

import asyncio
from pathlib import Path

from .core.bus import EventBus
from .core.type import Affect
from .core.config import Config
from .orchestrator.fsm import FSM
from .orchestrator.loop import Orchestrator
from .orchestrator.router import Router
from .memory.working import WorkingMemory
from .memory.episodic import EpisodicStore
from .memory.semantic import SemanticStore
from .memory.consolidate import Consolidate
from .cognition.llm_cloud import ClaudeLLM
from .cognition.llm_local import OllamaLLM
from .cognition.embedder import BGEmbedder
from .interface.cli import CliInterface
from .emotion.affect import AffectiveState
from .emotion.appraisal import Appraiser
from .emotion.detector import TextEmotionDetector


async def main():
    """Boot up all components and start the event loop."""

    print("=" * 60)
    print("Sezel — Phase 1 Persistent AI Assistant")
    print("=" * 60)
    print()

    # Step 1: Load configuration
    print("[1/12] Loading configuration...")
    config_dir = Path(__file__).parent / "config"
    try:
        # Load models.yaml from config directory
        config = Config.load_models(config_dir)
        model_name = config.get("models", {}).get("primary", {}).get("name", "qwen2.5:7b")
        model_host = config.get("models", {}).get("primary", {}).get("host", "http://localhost:11434")
        episodic_db = config.get("memory", {}).get("episodic", {}).get("database", "sezel.db")

        # Load routing config
        routing_config = Config.load(config_dir/ "routing.yaml")
        routing = routing_config.get("routing", {})
        complexity_threshold = routing.get("complexity_threshold", 0.6)
        cloud_model = routing.get("cloud_model", "claude-sonnet-4-6")

        print(f"   Model: {model_name}")
        print(f"   Host: {model_host}")
        print(f"   Cloud model: {cloud_model}")
        print(f"   Complexity threshold: {complexity_threshold}")
    except Exception as e:
        print(f"   Warning: Config loading failed ({e}), using defaults")
        model_name = "qwen2.5:7b"
        model_host = "http://localhost:11434"
        episodic_db = "sezel.db"
        complexity_threshold = 0.6
        cloud_model = "claude-sonnet-4-6"

    # Step 2: Create infrastructure

    print("[2/12] Creating event bus...")
    bus = EventBus()

    print("[3/12] Creating FSM...")
    fsm = FSM()

    print("[4/12] Creating working memory...")
    working = WorkingMemory(size=20)

    print("[5/12] Creating episodic store...")
    episodic = EpisodicStore(episodic_db)

    # Step 3: Rehydrate working memory from episodic store
    print("[6/12] Rehydrating from episodic memory...")
    try:
        recent_turns = await episodic.recent(10)
        for turn in recent_turns:
            working.append(turn)
        if recent_turns:
            print(f"   Restored {len(recent_turns)} turns from previous session")
        else:
            print("   Starting fresh (no previous history)")
    except Exception as e:
        print(f"   Warning: Could not rehydrate episodic memory ({e})")

    # STEP 4: Create Local LLM (Ollama)
    print("[7/12] Initializing local LLM (Ollama)...")
    try:
        # Use explicit name `local_llm` to match later references
        local_llm = OllamaLLM(model=model_name, host=model_host)
        print(f"   LLM ready: {model_name}")
    except Exception as e:
        print(f"   ERROR: Failed to initialize LLM: {e}")
        return

    # STEP 5: Create Embedder + Semantic Store

    print("[8/12] Initializing embedder (text → vectors)...")
    embedder = BGEmbedder()
    # Embedder loads lazily on first use, so this is instant

    print("[9/12] Creating semantic memory store...")
    # Use variable name `semantic` (not `semantic_store`) to match later use
    semantic = SemanticStore(embedder=embedder)
    try:
        memory_count = await semantic.count()
        print(f"   Semantic store ready ({memory_count} existing memories)")
    except Exception as e:
        print(f"   Warning: Semantic store init issue ({e})")

    # STEP 6: Create Cloud LLM (Claude)

    print("[10/12] Initializing cloud LLM (Claude)...")
    cloud_llm = None
    try:
        cloud_llm = ClaudeLLM(model=cloud_model)
        print(f"   Cloud LLM ready: {cloud_model}")
    except Exception as e:
        # Cloud LLM is optional — continue with local LLM only
        cloud_llm = None
        print(f"   Cloud LLM not available: {e}")
        print("   (Sezel will use local LLM only. Set ANTHROPIC_API_KEY for cloud)")

    # STEP 7: Create Router + Consolidator

    print("[11/12] Creating router...")
    router = Router(
        local_llm=local_llm,
        complexity_threshold=complexity_threshold
    )

    # Create Consolidator (only if cloud is available)
    consolidator = None
    if cloud_llm:
        # Consolidate class in memory.consolidate is named `Consolidate`
        consolidator = Consolidate(
            cloud_llm=cloud_llm,
            episodic_store=episodic,
            semantic_store=semantic
        )
        print("   Consolidator ready (background memory processor)")
    else:
        print("   Consolidator disabled (needs cloud LLM)")

    affective_state= AffectiveState(
        baseline= Affect(valence= 0.1, arousal= 0.0, dominance= 0.1),
        decay_per_sec= 0.95,
        persist_path= "sezel_mood.json"
    )

    initial_mood = affective_state.current()
    print(f"   Affective state loaded: {initial_mood.as_prompt()}")

    text_emotion= TextEmotionDetector()
    print("   Text emotion detector ready (model loads on first use)")

    appraiser= Appraiser(affective_state)
    print("   Appraiser ready")

    # STEP 8: Wire the Orchestrator
    print("[12/12] Wiring orchestrator...")

    # Wire up orchestrator
    orchestrator = Orchestrator(
        bus=bus,
        local_llm=local_llm,
        cloud_llm=cloud_llm or local_llm,
        router=router,
        working=working,
        episodic=episodic,
        semantic=semantic,
        fsm=fsm,
        affective_state=affective_state,
        text_emotion_detector=text_emotion,
        decay_interval=30.0,
    )

    cli = CliInterface(bus)
    orchestrator.cli = cli

    print()
    print("=" * 60)
    print("Starting orchestrator loop...")
    print("=" * 60)
    print()

    # STEP 9: Start concurrent tasks
    try:
        await asyncio.gather(
            cli.run(),
            orchestrator.run(),
            return_exceptions=False
        )
    except KeyboardInterrupt:
        print("\n[Shutting down...]")
    finally:
        print("[Closing connections...]")
        affective_state.persist()
        print("[Mood persisted to disk]")
        try:
            await local_llm.close()
            if cloud_llm:
                await cloud_llm.close()
            await episodic.close()
            await bus.close()
        except Exception as e:
            print(f"Warning during shutdown: {e}")
        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())

