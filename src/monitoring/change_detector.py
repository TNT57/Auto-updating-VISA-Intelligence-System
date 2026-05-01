"""
Change detection engine — Phase 2 (improved).

Implements three-level change detection:
  Level 1 — File hash comparison (fast "has anything changed?")
  Level 2 — Text diff via difflib ("what changed?")
  Level 3 — Severity classification ("how important is it?")
            Now uses LLM-based classification with keyword fallback.

Detected changes are recorded in SQLite via DatabaseManager.
"""

import difflib
from datetime import datetime

from loguru import logger

from src.utils.db_manager import DatabaseManager


# ------------------------------------------------------------------
# Severity classification keywords
# ------------------------------------------------------------------
CRITICAL_KEYWORDS = [
    "processing time",
    "processing times",
    "eligibility",
    "eligible",
    "requirement",
    "requirements",
    "criteria",
    "must",
    "mandatory",
    "refuse",
    "refusal",
    "cancel",
    "cancellation",
    "visa validity",
    "stay period",
    "work rights",
    "study rights",
    "condition",
    "conditions",
]

IMPORTANT_KEYWORDS = [
    "fee",
    "fees",
    "charge",
    "charges",
    "cost",
    "price",
    "document",
    "documents",
    "evidence",
    "form",
    "application",
    "apply",
    "lodge",
    "health",
    "insurance",
    "oshc",
    "english",
    "ielts",
    "pte",
    "toefl",
    "language",
]

# If none of the above match → MINOR (formatting, contacts, etc.)


class ChangeDetector:
    """Detects and classifies changes between document versions."""

    def __init__(self, db: DatabaseManager | None = None, use_llm: bool = True):
        self.db = db or DatabaseManager()
        self.use_llm = use_llm
        self._llm_client = None
        # Ensure tables exist
        self.db.create_tables()

    def _get_llm_client(self):
        """Lazy-initialize LLM client for severity classification."""
        if self._llm_client is None and self.use_llm:
            try:
                from src.generation.llm_client import LLMClient
                self._llm_client = LLMClient()
            except Exception as e:
                logger.warning("LLM client unavailable for classification: {}", e)
                self.use_llm = False
        return self._llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        old_content: str | None,
        new_content: str,
        source_url: str | None = None,
    ) -> list[dict]:
        """
        Compare old and new content, return a list of detected changes.

        Each change dict has: severity, change_type, old_value, new_value, summary.
        If old_content is None this is treated as a first-time snapshot (no diff).
        """
        # Level 1 — Hash comparison
        if old_content is None:
            logger.info("No previous snapshot for {} — storing baseline", source_url)
            return []

        old_hash = self._hash(old_content)
        new_hash = self._hash(new_content)

        if old_hash == new_hash:
            logger.info("Content unchanged: {}", source_url)
            return []

        logger.info(
            "Content changed for {} ({} → {})",
            source_url,
            old_hash[:12],
            new_hash[:12],
        )

        # Level 2 — Text diff
        diffs = self._compute_diff(old_content, new_content)

        if not diffs:
            logger.info("Hash differs but no meaningful text diffs — whitespace/formatting")
            return [{
                "severity": "MINOR",
                "change_type": "formatting",
                "old_value": None,
                "new_value": None,
                "summary": "Content hash changed but no significant text differences detected (possible whitespace/formatting change).",
                "source_url": source_url,
            }]

        # Level 3 — Classify each diff
        changes = []
        for diff in diffs:
            diff["severity"] = self.classify_severity(diff["old_value"] or "" + diff["new_value"] or "")
            diff["source_url"] = source_url
            changes.append(diff)

        logger.info(
            "Detected {} change(s) for {}",
            len(changes),
            source_url,
        )
        return changes

    def compare_and_record(
        self,
        old_content: str | None,
        new_content: str,
        source_url: str | None = None,
    ) -> list[dict]:
        """Compare content and record all detected changes in the database."""
        changes = self.compare(old_content, new_content, source_url)

        for change in changes:
            self.db.record_change(
                severity=change["severity"],
                change_type=change["change_type"],
                old_value=change.get("old_value"),
                new_value=change.get("new_value"),
                summary=change.get("summary"),
                source_url=change.get("source_url"),
            )

        return changes

    def classify_severity(self, text: str, summary: str | None = None) -> str:
        """
        Classify a change as CRITICAL, IMPORTANT, or MINOR.

        Tries LLM-based classification first (Improvement #5).
        Falls back to keyword matching if LLM is unavailable.
        """
        # Try LLM-based classification
        if self.use_llm:
            llm = self._get_llm_client()
            if llm:
                try:
                    return self._classify_severity_llm(llm, text, summary)
                except Exception as e:
                    logger.warning("LLM classification failed, using keywords: {}", e)

        # Keyword fallback
        return self._classify_severity_keywords(text)

    def _classify_severity_llm(
        self, llm, text: str, summary: str | None = None
    ) -> str:
        """Use LLM to classify change severity (Improvement #5)."""
        from src.generation.prompt_templates import SEVERITY_CLASSIFICATION_PROMPT

        prompt = SEVERITY_CLASSIFICATION_PROMPT.format(
            changed_text=text[:1500],  # Truncate to fit token limits
            change_summary=summary or "Text change detected",
        )
        raw = llm.generate(
            prompt=prompt,
            system_prompt="You are a concise classifier. Respond with exactly one severity word and one reason sentence.",
            max_tokens=100,
            temperature=0.0,
        )
        # Parse the verdict
        raw_upper = raw.upper().strip()
        if "CRITICAL" in raw_upper:
            return "CRITICAL"
        elif "IMPORTANT" in raw_upper:
            return "IMPORTANT"
        else:
            return "MINOR"

    @staticmethod
    def _classify_severity_keywords(text: str) -> str:
        """Keyword-based severity classification (fallback)."""
        text_lower = text.lower()

        for kw in CRITICAL_KEYWORDS:
            if kw in text_lower:
                return "CRITICAL"

        for kw in IMPORTANT_KEYWORDS:
            if kw in text_lower:
                return "IMPORTANT"

        return "MINOR"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(content: str) -> str:
        """SHA-256 hash of normalised content."""
        import hashlib
        # Normalise: strip whitespace and collapse multiple spaces
        normalised = " ".join(content.split())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_diff(old: str, new: str) -> list[dict]:
        """
        Compute line-level unified diff between old and new content.

        Returns a list of dicts, each representing a meaningful change block.
        """
        old_lines = old.splitlines()
        new_lines = new.splitlines()

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        changes: list[dict] = []

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                continue

            old_block = "\n".join(old_lines[i1:i2])
            new_block = "\n".join(new_lines[j1:j2])

            # Generate a human-readable summary
            if op == "replace":
                change_type = "content_update"
                summary = f"Content updated ({i2 - i1} line(s) changed)"
            elif op == "insert":
                change_type = "content_added"
                summary = f"New content added ({j2 - j1} line(s))"
            elif op == "delete":
                change_type = "content_removed"
                summary = f"Content removed ({i2 - i1} line(s))"
            else:
                continue  # skip 'replace' that's really equal

            changes.append({
                "change_type": change_type,
                "old_value": old_block or None,
                "new_value": new_block or None,
                "summary": summary,
            })

        return changes