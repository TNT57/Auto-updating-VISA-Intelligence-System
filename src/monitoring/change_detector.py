"""
Change detection engine — Phase 2 (Weeks 3-4).

Will implement:
- File-level hash comparison (fast detection)
- Content-level semantic diff
- Structured field extraction and comparison
- Severity classification (Critical / Important / Minor)
"""


class ChangeDetector:
    """Detects and classifies changes between document versions."""

    def __init__(self):
        raise NotImplementedError("Phase 2 — Coming in Weeks 3-4")

    def detect_changes(self, old_content: str, new_content: str) -> list[dict]:
        """Compare old and new content, return list of changes."""
        raise NotImplementedError

    def classify_severity(self, change: dict) -> str:
        """Classify change as CRITICAL, IMPORTANT, or MINOR."""
        raise NotImplementedError