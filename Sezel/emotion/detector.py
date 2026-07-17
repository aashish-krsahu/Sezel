"""Text emotion detectors — classify user text into PAD affect."""
from networkx.algorithms import dominance
from sqlalchemy.ext.asyncio import result

from ..core.type import Affect

EMOTION_TO_PAD: dict[str, tuple[float, float, float]] = {
    "anger":     (-0.51,  0.59,  0.25),
    "disgust":   (-0.60,  0.35,  0.11),
    "fear":      (-0.64,  0.60, -0.43),
    "joy":       ( 0.76,  0.48,  0.35),
    "neutral":   ( 0.00,  0.00,  0.00),
    "sadness":   (-0.63, -0.27, -0.33),
    "surprise":  ( 0.40,  0.67, -0.13)
}

class TextEmotionDetector:
    """
    Detect user emotion from text using a transformer model.

    Uses 'j-hartmann/emotion-english-distilroberta' which classifies
    text into 7 emotion classes. Maps the result to PAD affect.
    """

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base") -> None:
        self.model_name = model_name
        self._classifier = None

    def __call__(self, text: str) -> Affect | None:
        return self.detect(text)

    def _load(self) -> None:
        """
            Lazy-load the transformer pipeline.
            loads when detect() is called for the first time.
            keeps CLI startup fast.
        """

        if self._classifier is None:
            try:
                from transformers import pipeline
                self._classifier = pipeline(
                    "text-classification",
                    model = self.model_name,
                    top_k = None
                )
            except ImportError:
                print("  [TextEmotionDetector: transformers not installed, using fallback]")
                self._classifier = "fallback"
            except Exception as e:
                print(f"  [TextEmotionDetector: failed to load model ({e}), using fallback]")
                self._classifier = "fallback"

    def detect(self, text: str) -> Affect | None:

        if not text or not text.strip():
            return None

        self._load()

        if self._classifier is None:
            return None

        if self._classifier == "fallback":
            # Use rough sentiment as fallback
            from .appraisal import _rough_sentiment
            sent = _rough_sentiment(text)
            return Affect(
                valence=sent * 0.5,
                arousal= sent * 0.3,
                dominance= sent * 0.2
            )

        # Run the transformer classifier
        try:
            results= self._classifier(text)
        except Exception as e:
            print(f"  [TextEmotionDetector: inference error ({e}), using fallback]")
            from .appraisal import _rough_sentiment
            sent = _rough_sentiment(text)
            return Affect(
                valence=sent * 0.5,
                arousal=sent * 0.3,
                dominance=sent * 0.2
            )

        # Find the highest-scoring emotion
        if isinstance(results, list) and results:
            scores = results[0] if results and isinstance(results[0], list) else results
            if scores:
                top = max(scores, key=lambda x: x.get("score", 0))
                label = top.get("label", "neutral").lower()
                pad = EMOTION_TO_PAD.get(label, EMOTION_TO_PAD["neutral"])
                return Affect(
                    valence=pad[0],
                    arousal=pad[1],
                    dominance=pad[2]
                )

        return None