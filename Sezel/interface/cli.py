"""Command-line interface for user input/output."""

import sys
import asyncio
import time
from ..core.bus import EventBus
from ..core.type import Event


class CliInterface:
    """Read from stdin, publish events; receive outputs and print."""

    def __init__(self, bus: EventBus):
        self.bus = bus

    async def run(self) -> None:
        """Main loop: read stdin line by line, publish as Events."""
        loop = asyncio.get_event_loop()

        print("Sezel: Hello! I'm ready to chat. Type your message and press Enter.")
        print("(Press Ctrl+C to exit)\n")

        try:
            while True:
                # Run blocking stdin read in an executor to avoid blocking event loop
                line = await loop.run_in_executor(None, sys.stdin.readline)

                if not line:  # EOF (Ctrl+D on Unix, Ctrl+Z+Enter on Windows)
                    print("Sezel: Goodbye!")
                    break

                text = line.strip()
                if not text:  # Skip empty lines
                    continue

                # Publish as event
                event = Event(
                    kind="user_text",
                    payload={"text": text},
                    ts=time.time()
                )
                await self.bus.publish(event)
        except KeyboardInterrupt:
            print("\n\nSezel: Goodbye!")

    async def emit(self, text: str) -> None:
        """Emit a response to the user."""
        print(f"\nSezel: {text}\n")

