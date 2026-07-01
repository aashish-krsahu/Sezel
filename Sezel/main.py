"""Main entry point for Sezel Phase 1."""

import asyncio
import sys
from pathlib import Path

from .core.bus import EventBus
from .core.config import Config
from .core.type import Turn
from .orchestrator.fsm import FSM
from .orchestrator.loop import Orchestrator
from .memory.working import WorkingMemory
from .memory.episodic import EpisodicStore
from .cognition.llm_local import OllamaLLM
from .interface.cli import CliInterface


async def main():
    """Boot up all components and start the event loop."""

    print("=" * 60)
    print("Sezel — Phase 1 Persistent AI Assistant")
    print("=" * 60)
    print()

    # Step 1: Load configuration
    print("[1/7] Loading configuration...")
    config_dir = Path(__file__).parent / "config"
    try:
        config = Config.load_models(config_dir)
        model_name = config.get("models", {}).get("primary", {}).get("name", "qwen2.5:7b")
        model_host = config.get("models", {}).get("primary", {}).get("host", "http://localhost:11434")
        episodic_db = config.get("memory", {}).get("episodic", {}).get("database", "sezel.db")
        print(f"   Model: {model_name}")
        print(f"   Host: {model_host}")
        print(f"   Episodic DB: {episodic_db}")
    except Exception as e:
        print(f"   Warning: Config loading failed ({e}), using defaults")
        model_name = "llama3"
        model_host = "http://localhost:11434"
        episodic_db = "sezel.db"

    # Step 2: Create infrastructure
    print("[2/7] Creating event bus...")
    bus = EventBus()

    print("[3/7] Creating FSM...")
    fsm = FSM()

    print("[4/7] Creating working memory...")
    working = WorkingMemory(size=20)

    print("[5/7] Creating episodic store...")
    episodic = EpisodicStore(episodic_db)

    # Step 6: Rehydrate working memory from episodic store
    print("[6/7] Rehydrating from episodic memory...")
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

    # Step 7: Create LLM and CLI
    print("[7/7] Initializing components...")
    try:
        llm = OllamaLLM(model=model_name, host=model_host)
        print(f"   LLM ready: {model_name}")
    except Exception as e:
        print(f"   ERROR: Failed to initialize LLM: {e}")
        return

    cli = CliInterface(bus)

    # Wire up orchestrator
    orchestrator = Orchestrator(
        bus=bus,
        llm=llm,
        working=working,
        episodic=episodic,
        fsm=fsm,
        cli=cli
    )

    print()
    print("=" * 60)
    print("Starting orchestrator loop...")
    print("=" * 60)
    print()

    # Start concurrent tasks: CLI input loop + Orchestrator
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
        try:
            await llm.close()
            await episodic.close()
            await bus.close()
        except Exception as e:
            print(f"Warning during shutdown: {e}")
        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())

