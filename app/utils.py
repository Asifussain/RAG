import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean extracted PDF text by normalizing whitespace and removing artifacts.

    Handles:
    - Excessive whitespace / blank lines
    - Leading/trailing whitespace per line
    - Non-breaking spaces and other Unicode whitespace
    - Common PDF extraction artifacts (e.g. form-feed characters)
    """
    # Replace non-breaking spaces and other Unicode whitespace with regular space
    text = re.sub(r"[\xa0\u2000-\u200b\u202f\u205f\u3000]", " ", text)

    # Remove form-feed and other control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse multiple spaces/tabs into a single space (within a line)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove brochure artifacts: standalone page numbers, very short orphan lines
    cleaned_lines = []
    for line in lines:
        # Skip lines that are just numbers (page numbers like "49", "50")
        if re.fullmatch(r"\d{1,4}", line):
            continue
        # Skip very short orphan lines (1-2 chars) that are layout noise
        if len(line) <= 2 and not line.isalpha():
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Strip leading/trailing whitespace from the whole text
    return text.strip()


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """Compute a fast character-level Jaccard similarity between two texts.

    Uses word-level sets so it's fast enough for deduplication during chunking.
    Returns a float in [0, 1] where 1 means identical word sets.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
