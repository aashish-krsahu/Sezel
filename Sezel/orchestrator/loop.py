"""Orchestrator: the heart of Phase 1 — manages the event loop and state transitions."""

import asyncio
from ..core.bus import EventBus
from ..core.type import Event, Perception, Context, Affect, Turn, Plan
from ..core.protocols import LLM
from ..orchestrator.fsm import FSM, State
from ..memory.working import WorkingMemory
from ..memory.episodic import EpisodicStore
from ..interface.cli import CliInterface


class Orchestrator:
    """Main orchestrator managing the full event loop and state transitions."""

    def __init__(
        self,
        bus: EventBus,
        llm: LLM,
        working: WorkingMemory,
        episodic: EpisodicStore,
        fsm: FSM,
        cli: CliInterface | None = None
    ):
        self.bus = bus
        self.llm = llm
        self.working = working
        self.episodic = episodic
        self.fsm = fsm
        self.cli = cli

    async def run(self) -> None:
        """Main orchestrator loop: process events one at a time through all states."""
        try:
            while True:
                # ── IDLE: Wait for an event ──
                # FSM: IDLE ──► PERCEIVING
                self.fsm.to(State.PERCEIVING)

                ev = await self.bus.next()

                # ── PERCEIVE: Parse raw event into structured perception ──
                perc = self._perceive(ev)
                # FSM: PERCEIVING ──► ASSEMBLING
                self.fsm.to(State.ASSEMBLING)

                # ── ASSEMBLE: Build context from perception + memory ──
                ctx = self._assemble(perc)
                # FSM: ASSEMBLING ──► REASONING
                self.fsm.to(State.REASONING)

                # ── REASON: Get response from LLM ──
                try:
                    plan = await self.llm.complete(ctx)
                except Exception as e:
                    print(f"\n[Error during LLM reasoning: {e}]")
                    plan = Plan(text="I'm having trouble thinking right now. Try again?")

                # FSM: REASONING ──► RESPONDING
                self.fsm.to(State.RESPONDING)

                # ── RESPOND: Emit the response ──
                await self._emit(plan.text)

                # FSM: RESPONDING ──► CONSOLIDATING
                self.fsm.to(State.CONSOLIDATING)

                # ── CONSOLIDATE: Save to working + episodic memory ──
                await self._consolidate(perc, plan)

                # FSM: CONSOLIDATING ──► IDLE
                self.fsm.to(State.IDLE)

        except KeyboardInterrupt:
            print("\n[Orchestrator stopped]")
        except Exception as e:
            print(f"\n[Unexpected error in orchestrator: {e}]")

    def _perceive(self, event: Event) -> Perception:
        """Extract perception from raw event."""
        text = event.payload.get("text", "")
        return Perception(text=text)

    def _assemble(self, perc: Perception) -> Context:
        """Assemble context from perception + memory."""
        working_turns = self.working.recent(10)
        return Context(
            working=working_turns,
            retrieved=[],  # Phase 1: no retrieval
            mood=Affect(),  # Phase 1: neutral mood
            perception=perc
        )

    async def _emit(self, text: str) -> None:
        """Emit response to user."""
        if self.cli:
            await self.cli.emit(text)
        else:
            print(f"\nSezel: {text}\n")

    async def _consolidate(self, perc: Perception, plan: Plan) -> None:
        """Save turns to both working and episodic memory."""
        # Create Turn objects
        user_turn = Turn(role="user", content=perc.text)
        asst_turn = Turn(role="assistant", content=plan.text)

        # Add to working memory (in-RAM ring buffer)
        self.working.append(user_turn)
        self.working.append(asst_turn)

        # Write to episodic store (persistent SQLite)
        await self.episodic.write(user_turn, Affect())
        await self.episodic.write(asst_turn, Affect())

