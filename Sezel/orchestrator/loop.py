"""
    now with routing + semantic memory retrieval.

    The loop flow
        perceive → [detect emotion] -> assemble -> [appraise mood] → route → reason (local OR cloud) → respond → consolidate

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
from ..emotion.affect import AffectiveState
from ..emotion.appraisal import Appraiser
from ..emotion.detector import TextEmotionDetector

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
        cli: CliInterface | None = None,
        affective_state : AffectiveState | None = None,
        text_emotion_detector : TextEmotionDetector | None = None,
        decay_interval: float = 30,
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
        self.mood = affective_state or AffectiveState()
        self.text_emotion = text_emotion_detector or TextEmotionDetector()
        self.appraiser = Appraiser(self.mood)
        self.decay_interval = decay_interval
        self._decay_task: asyncio.Task | None = None

    async def run(self) -> None:
        """Main Orchestrator loop."""

        try:
            while True:

                # IDLE: Wait for an event (blocks here)

                self.fsm.to(State.PERCEIVING)

                ev = await self.bus.next()

                #  PERCEIVE: Parse raw event into perception

                perc = self._perceive(ev)

                if perc.text:
                    detected = self.text_emotion(perc.text)
                    if detected is not None:
                        perc.user_affect = detected

                self.fsm.to(State.ASSEMBLING)

                #  ASSEMBLE: Build context from perception +
                #            working memory + semantic retrieval

                ctx = await self._assemble(perc)
                self.fsm.to(State.ROUTING)

                current_mood = self.appraiser.appraise(perc, ctx)
                ctx.mood = current_mood

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

    def _perceive(self, event: Event):

        text = event.payload.get("text", "")
        return Perception(text=text)

    async def _assemble(self, perc: Perception) -> Context:
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
            mood=self.mood.current(),
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

        user_affect = perc.user_affect or Affect()
        current_mood = self.mood.current()

        user_turn = Turn(role="user", content=perc.text or "")
        asst_turn = Turn(role="assistant", content=plan.text)

        # Save to working memory (in-RAM)
        self.working.append(user_turn)
        self.working.append(asst_turn)

        # Save to episodic store (persistent SQLite)
        await self.episodic.write(user_turn, user_affect)
        await self.episodic.write(asst_turn, current_mood)

    async def _decay_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.decay_interval)
                self.mood.decay_tick()
                # Persist after decay so mood survives crashes
                self.mood.persist()
        except asyncio.CancelledError:
            pass

