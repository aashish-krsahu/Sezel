"""
Appraisal: map perceptions and events to affect deltas.

Implements a simplified OCC-style appraisal: the appraiser evaluates
what the user said (the detected emotion) and produces an affect delta
that updates Sezel's persistent affective state.
"""

import math
from ..core.type import Affect, Perception, Context
from .affect import AffectiveState

class Appraiser:
    """
    Evaluates perceptions and updates the affective state.

    Usage:
        appraiser = Appraiser(affective_state)
        delta = appraiser.appraise(perc, ctx)
        mood = affective_state.current()  # get the updated mood
    """

    def __init__(self,state: "AffectiveState"):
        self.state = state

    def appraise(self, perc:Perception, ctx: Context) -> Affect:
        """
        Evaluate a perception + context and return the *updated* mood.

        Steps:
        1. Start with the detected user emotion as the delta base
        2. Modulate based on context (salience, repetition, contradiction)
        3. Apply the delta to the affective state
        4. Return the new current mood

        Args:
            perc: The current perception (includes user_affect if detected)
            ctx: The assembled context (working memory, retrieved memories)

        Returns:
            The updated Affect after applying the appraisal delta.
        """
        # STEP 1:
        if perc.user_affect is not None:
            # Mirror the user's emotion at 30% intensity
            # (Sezel is its own agent, not a perfect mirror)
            delta = Affect(
                valence= perc.user_affect.valence * 0.3,
                arousal = perc.user_affect.arousal * 0.2,
                dominance= perc.user_affect.dominance * 0.15
            )
        else:
            # No detected emotion -> mild positive default
            delta = Affect(valence= 0.05, arousal= 0.02, dominance= 0.01)

        # STEP 2:
        if ctx.working:
            recent_moods = 0.0
            count = 0
            for turn in ctx.working[-4:]:
                if turn.role == "user":
                    sentiment = Appraiser._rough_sentiment(turn.content)
                    recent_moods += sentiment
                    count += 1

            if count > 0:
                ave_sentiment = recent_moods / count
                # If user consistently negative, amplify negative delta
                if ave_sentiment < -0.3 and delta.valence < 0:
                    delta.valence *= 1.5
                    delta.arousal *= 1.3

                elif  ave_sentiment > 0.3 and delta.valence > 0:
                    delta.valence *= 1.2

        # STEP 3:
        self.state.apply(delta)

        # STEP 4:
        return self.state.current()

    @staticmethod
    def _rough_sentiment(text: str) -> float:
        """
            This is a lightweight approximation used when the TextEmotion
            detector isn't available or as a fallback. Returns -1.0 to 1.0.

            The full TextEmotion model (in detectors.py) is more accurate
            but requires loading a transformer model.
        """
        negative_words = {
            "bad", "terrible", "awful", "hate", "angry", "sad", "upset",
            "frustrated", "annoyed", "horrible", "worst", "ugly", "stupid",
            "dumb", "wrong", "fail", "failed", "useless", "hate", "pain",
            "cry", "crying", "depressed", "anxious", "scared", "afraid",
            "disappointed", "disgusting", "horrible", "terrible", "awful",
        }
        positive_words = {
            "good", "great", "awesome", "amazing", "love", "happy", "glad",
            "wonderful", "fantastic", "excellent", "best", "beautiful",
            "perfect", "nice", "kind", "helpful", "thanks", "thank",
            "brilliant", "superb", "delightful", "joy", "excited",
            "grateful", "wonderful", "fantastic", "marvelous",
        }

        words = set(text.lower().split())
        neg_count = sum(1 for w in words if w in negative_words)
        pos_count = sum(1 for w in words if w in positive_words)

        if neg_count == 0 and pos_count == 0:
            return 0.0

        total = neg_count + pos_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total
