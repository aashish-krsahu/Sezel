from core.type import Plan
from ..core.type import Turn,Plan
from ..core.protocols import LLM
from ..memory.episodic import EpisodicStore
from ..memory.semantic import SemanticStore

class Consolidate:
    """Background worker that converts episodic memories into semantic facts.

        Episodic = "what happened" (raw conversation)
        Semantic = "what matters" (important facts)

        Usage:
            consolidator = Consolidator(cloud_llm, episodic_store, semantic_store)
            await consolidator.tick()  # Process one batch of new memories
    """

    def __init__(
        self,
        cloud_llm: LLM,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        max_per_tick: int = 10
    ):
        self.cloud_llm = cloud_llm
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.max_per_tick = max_per_tick

    async def tick(self) -> int:

        # Step 1: Get conversations that haven't been consolidated yet
        recent = await self.episodic.unconsolidated(self.max_per_tick)
        if not recent:
            return 0

        # Step 2: Build a summary prompt
        conversation_text = self._format_conversation(recent)
        summary_prompt = (
            "Review this conversation and extract IMPORTANT, DURABLE facts.\n"
            "Durable facts are things that stay true over time:\n"
            "- User's name, preferences, habits, personal info\n"
            "- Knowledge the user shared (facts, instructions)\n"
            "- Topics the user cares about\n\n"
            "Do NOT include:\n"
            "- Greetings or small talk\n"
            "- Transient things (current mood, temporary states)\n"
            "- Information already in the conversation context\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            "Output ONLY the facts, one per line, starting with '- '\n"
            "Example:\n"
            "- User's name is Alice\n"
            "- User lives in New York\n"
            "- User has a dog named Max\n"
            "If nothing important was said, output: NOTHING_IMPORTANT"
        )

        # Step 3: Create a mini-Context for the summarization prompt
        from ..core.type import Context, Perception, Affect
        mini_ctx = Context(
            working=[],
            retrieved=[],
            mood=Affect(),
            personal=Perception(text=summary_prompt),
        )

        # Step 4: Ask Cluade to extract facts
        try:
            plan = await self.cloud_llm.plan(mini_ctx)
        except Exception as e:
            print(f"   [consolidator error: {e}]")
            return 0

        # Step 5: Parse the response into individual facts
        facts = self._parse_facts(plan.text)

        if not facts:
            # Nothing important - still marked as consolidated so we don't
            # keep re-processing it
            await self.episodic.mark_consolidated(recent)
            return 0

        # Step 6: Save each facts to Semantic memory
        saved_count = 0
        for fact in facts:
            await self.semantic_store.upsert(
                text=fact,
                meta= {"Source": "Consolidation", "type": "fact"}
            )
            saved_count += 1

        # Step 7: Mark these conversation as processed
        await self.episodic._mark_consolidated(recent)

        print(f"   [Consolidator: saved {saved_count} new facts]")
        return saved_count

    def _format_conversation(self, turns: list[Turn]):
        """
        Turns raw Turn objects into something Claude can read:
            User: hello
            Assistant: hi there
            User: my name is Alice
        """

        lines = []
        for turn in Turn:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f" {label}: {turn.text}")

        return "\n".join(lines)

    async def _parse_facts(self, text:str):
        """Parse Claude's response into individual facts.

            Expects format:
                - Fact one
                - Fact two
                NOTHING_IMPORTANT
        """

        if "NOTHING_IMPORTANT" in text.upper():
            return []

        facts = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                facts.append(line[2:])

        return facts
