"""Main entry point for Sezel Phase 1."""

# When this module is executed directly (python Sezel\main.py) the
# package context for relative imports is not set which raises
# "attempted relative import with no known parent package". Set up
# sys.path so the Sezel package is importable, then use absolute imports.
import sys
import pathlib

from networkx.algorithms import dominance

# Unconditionally add the parent directory to sys.path so Sezel is always importable
package_dir = pathlib.Path(__file__).resolve().parent
if str(package_dir.parent) not in sys.path:
    sys.path.insert(0, str(package_dir.parent))

import asyncio
from pathlib import Path

from Sezel.core.bus import EventBus
from Sezel.core.type import Affect
from Sezel.core.config import Config
from Sezel.orchestrator.fsm import FSM
from Sezel.orchestrator.loop import Orchestrator
from Sezel.orchestrator.router import Router
from Sezel.memory.working import WorkingMemory
from Sezel.memory.episodic import EpisodicStore
from Sezel.memory.semantic import SemanticStore
from Sezel.memory.consolidate import Consolidate
from Sezel.cognition.llm_cloud import ClaudeLLM
from Sezel.cognition.llm_local import OllamaLLM
from Sezel.cognition.embedder import BGEmbedder
from Sezel.cognition.vlm import OllamaVLM
from Sezel.cognition.model_manager import ModelManager, VRAMbudget
from Sezel.interface.cli import CliInterface
from Sezel.emotion.affect import AffectiveState
from Sezel.emotion.appraisal import Appraiser
from Sezel.emotion.detector import TextEmotionDetector
from Sezel.vision.ocr import OCREngine
from Sezel.vision.vlm_pipeline import VisionPipeline
from Sezel.vision.ally import UIAccessibility
from Sezel.vision.capture import ScreenCapture
from Sezel.memory.visual import VisualStore


async def main():
    """Boot up all components and start the event loop."""

    print("=" * 60)
    print("Sezel — Phase 1 Persistent AI Assistant")
    print("=" * 60)
    print()

    # Step 1: Load configuration
    print("[1/14] Loading configuration...")
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
        
        # Vision Config
        vision_config = routing_config.get("vision", {})
        vlm_model = vision_config.get("vlm_model", "llava")
        vision_enabled = vision_config.get("enabled", True)

        print(f"   Model: {model_name}")
        print(f"   Host: {model_host}")
        print(f"   Cloud model: {cloud_model}")
        print(f"   Complexity threshold: {complexity_threshold}")
        print(f"   Vision enabled: {vision_enabled}")
    except Exception as e:
        print(f"   Warning: Config loading failed ({e}), using defaults")
        model_name = "qwen2.5:7b"
        model_host = "http://localhost:11434"
        episodic_db = "sezel.db"
        complexity_threshold = 0.6
        cloud_model = "claude-sonnet-4-6"
        vlm_model = "llava"
        vision_enabled = True

    # Step 2: Create infrastructure

    print("[2/14] Creating event bus...")
    bus = EventBus()

    print("[3/14] Creating FSM...")
    fsm = FSM()

    print("[4/14] Creating working memory...")
    working = WorkingMemory(size=20)

    print("[5/14] Creating episodic store...")
    episodic = EpisodicStore(episodic_db)

    # Step 3: Rehydrate working memory from episodic store
    print("[6/14] Rehydrating from episodic memory...")
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
    print("[7/14] Initializing local LLM (Ollama)...")
    try:
        # Use explicit name `local_llm` to match later references
        local_llm = OllamaLLM(model=model_name, host=model_host)
        print(f"   LLM ready: {model_name}")
    except Exception as e:
        print(f"   ERROR: Failed to initialize LLM: {e}")
        return

    # STEP 5: Create Embedder + Semantic Store

    print("[8/14] Initializing embedder (text → vectors)...")
    embedder = BGEmbedder()
    # Embedder loads lazily on first use, so this is instant

    print("[9/14] Creating semantic memory store...")
    # Use variable name `semantic` (not `semantic_store`) to match later use
    semantic = SemanticStore(embedder=embedder)
    try:
        memory_count = await semantic.count()
        print(f"   Semantic store ready ({memory_count} existing memories)")
    except Exception as e:
        print(f"   Warning: Semantic store init issue ({e})")

    # STEP 6: Create Cloud LLM (Claude)

    print("[10/14] Initializing cloud LLM (Claude)...")
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

    print("[11/14] Creating router...")
    router = Router(
        local_llm=local_llm,
        complexity_threshold=complexity_threshold,
        vision_enabled = vision_enabled,
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

    # Step 8: Emotion Engine
    print("[12/14] Setting up emotion engine...")
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

    # STEP 9: Create vision components
    print("[13/14] Setting up vision System...")
    vision_pipeline = None
    if vision_enabled:
        try:
            capture = ScreenCapture()
            ocr = OCREngine()
            ally = UIAccessibility()
            visual_store = VisualStore(
                storage_dir="visual_memory",
                embedder=embedder,
                semantic_store=semantic,
            )

            # Create VLM (via ollama)
            vlm = None
            vlm_model_available = vision_config.get("vlm_model", "llava")
            try:
                vlm = OllamaVLM(model = vlm_model_available, host = model_host)
                print(f"   VLM ready: {vlm_model_available}")
            except Exception as e:
                print(f"   VLM not available: {e}")
                print("   (Sezel will use OCR + UI tree without VLM captioning)")

            # Create VRAM model manager
            vram_budget = VRAMbudget(
                total_gb=vision_config.get("total_gb", 6.0),
                reserved_gb= 0.5
            )
            model_manager = ModelManager(vram_budget = vram_budget)
            if vlm:
                model_manager.register("vlm", vram_gb = 4.0)

            vision_pipeline = VisionPipeline(
                capture = capture,
                ocr = ocr,
                ally = ally,
                vlm = vlm,
                model_manager = model_manager,
                visual_store = visual_store,
            )
            print("   Vision pipeline ready")
            print(f"   Visual memory directory: visual_memory/")
        except Exception as e:
            print(f"   Warning: Vision setup failed ({e})")
            print("   Sezel will continue without vision capabilities")
            vision_pipeline = None
    else:
        print("   Vision disabled (configured in routing.yaml)")

    # STEP 10: Wire the Orchestrator
    # Wire up orchestrator
    print("[14/14] Wiring orchestrator...")

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
        vision_pipeline=vision_pipeline,
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

