"""
    now with routing + semantic memory retrieval.

    The loop flow
        perceive → assemble (WITH memory) → ROUTE → reason (local OR cloud) → respond → consolidate

"""

import asyncio

from .router import Router
from ..core.bus import EventBus
from ..core.type import Event, Perception, Context, Affect, Turn, Plan
from ..core.protocols import LLM
from ..orchestrator.fsm import FSM, State
from ..memory.working import WorkingMemory
from ..memory.episodic import EpisodicStore
from ..memory.semantic import SemanticStore
from ..interface.cli import CliInterface


class Orchestrator:
    """Main orchestrator managing the full event loop and state transitions."""

    def __init__(
        self,
        bus: EventBus,
        local_llm: LLM,
        cloud_llm: LLM,
        router: Router,
        working: WorkingMemory,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        fsm: FSM,
        cli: CliInterface | None = None
    ):
        self.bus = bus
        self.local_llm = local_llm
        self.cloud_llm = cloud_llm
        self.router = router
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.fsm = fsm
        self.cli = cli

    async def run(self) -> None:
        """Main Orchestrator loop."""

        try:
            while True:

                # IDLE: Wait for an event (blocks here)

                self.fsm.to(State.PERCEIVING)

                ev = await self.bus.next()

                #  PERCEIVE: Parse raw event into perception

                perc = self._perceive(ev)
                self.fsm.to(State.ASSEMBLING)

                #  ASSEMBLE: Build context from perception +
                #            working memory + semantic retrieval

                ctx = await self._assemble(perc)
                self.fsm.to(State.ROUTING)

                # ROUTE: Decide which LLM to use

                route = await self.router.decide(ctx)
                llm = self.cloud_llm if route == "cloud" else self.local_llm

                self.fsm.to(State.REASONING)

                # REASON: Get response from the chosen LLM

                try:
                    plan = await llm.complete(ctx)
                except Exception as e:
                    print(f"\n[Error during LLM reasoning ({route}): {e}]")
                    plan = Plan(text="I'm having trouble thinking right now. Try again?")

                self.fsm.to(State.RESPONDING)

                # RESPOND: Emit the response

                await self._emit(plan.text)

                self.fsm.to(State.CONSOLIDATING)

                # CONSOLIDATE: Save important facts to memory
                # CONSOLIDATE: Save to working + episodic memory

                await self._consolidate(perc, plan)

                self.fsm.to(State.IDLE)

        except KeyboardInterrupt:
            print("\n[Orchestrator stopped]")
        except Exception as e:
            print(f"\n[Unexpected error in orchestrator: {e}]")

    async def _perceive(self, perc: Perception) -> Context:
        """
        Assemble context from perception + working memory + semantic retrieval.
        This now searches semantic memory for relevant past facts and
        includes them in the context. This is how Sezel
        "remembers" things from old conversations!
        """

        # Step 1: Get recent conversation history from working memory
        working_turn = self.working.recent(10)

        # Step 2: Search semantic memory for relevant past information
        retrieved_turn = []
        if perc.text:
            try:
                # Search for memories related to the current question

                hits = await self.semantic.recall(perc.text, k=5)

                for hit in hits:
                    if hit.score > 0.3:
                        retrieved_turn.append(Turn(
                            role="system",
                            content=f"[past memory] {hit.text}",
                            ts=0.0
                        ))
            except Exception as e:
                print(f"[Error during semantic memory retrieval: {e}]")

        # Step 3: Estimate token count for routing decisions
        # (Rough estimate: 1 token ≈ 4 characters)
        total_chars = sum(len(t.content) for t in working_turn)
        total_chars += sum(len(t.content) for t in retrieved_turn)
        total_chars += len(perc.text or "")
        token_estimate = total_chars // 4

        # Step 4: Build the full context
        return Context(
            working= working_turn,
            retrieved= retrieved_turn,
            mood=Affect(),
            perception=perc,
            token_estimate=token_estimate
        )

    async def _emit(self, text:str)-> None:
        """Emit response to user."""
        if self.cli:
            await self.cli.emit(text)
        else:
            await print(f"\nSezel: {text}\n")

    async def _consolidate(self, perc: Perception, plan: Plan) -> None:
        """Save turns to both working and episodic memory."""
        user_turn = Turn(role="user", content=perc.text or "")
        asst_turn = Turn(role="assistant", content=plan.text)

        # Save to working memory (in-RAM)
        self.working.append(user_turn)
        self.working.append(asst_turn)

        # Save to episodic store (persistent SQLite)
        await self.episodic.write(user_turn, Affect())
        await self.episodic.write(asst_turn, Affect())

