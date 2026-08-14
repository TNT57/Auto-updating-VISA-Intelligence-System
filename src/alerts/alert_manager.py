"""
Alert delivery for detected policy changes.

Sends a Discord webhook when the daily pipeline finds changes at or above the
configured severity. Delivery attempts are written to the `alert_logs` table
and successfully-alerted changes are marked `notified=2` so a re-run never
alerts on the same change twice.
"""

import httpx
from loguru import logger

from src.utils.config import settings
from src.utils.db_manager import DatabaseManager

# Ordered least → most severe. Used to resolve `alert_min_severity`.
SEVERITY_ORDER = ("MINOR", "IMPORTANT", "CRITICAL")

SEVERITY_EMOJI = {"CRITICAL": "🔴", "IMPORTANT": "🟡", "MINOR": "🟢"}

# Discord embed colours (decimal RGB).
SEVERITY_COLOR = {"CRITICAL": 0xE01E5A, "IMPORTANT": 0xECB22E, "MINOR": 0x2EB67D}

# Discord caps embed description length; leave headroom for our own framing.
_MAX_FIELD_CHARS = 900


class AlertManager:
    """Delivers change alerts to Discord (and logs every attempt)."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        webhook_url: str | None = None,
        timeout: float = 15.0,
    ):
        self.db = db or DatabaseManager()
        self.webhook_url = (
            webhook_url
            if webhook_url is not None
            else settings.discord_webhook_url
        )
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether a Discord webhook is configured."""
        return bool(self.webhook_url)

    def severities_to_alert(self) -> tuple[str, ...]:
        """Resolve `alert_min_severity` into the set of severities to send."""
        minimum = (settings.alert_min_severity or "IMPORTANT").upper()
        if minimum not in SEVERITY_ORDER:
            logger.warning(
                "Unknown ALERT_MIN_SEVERITY '{}' — defaulting to IMPORTANT",
                minimum,
            )
            minimum = "IMPORTANT"
        cutoff = SEVERITY_ORDER.index(minimum)
        return tuple(SEVERITY_ORDER[cutoff:])

    def send_pending_alerts(self) -> int:
        """
        Alert on every un-notified change at or above the minimum severity.

        Returns the number of changes included in a successfully sent alert.
        """
        if not self.enabled:
            logger.info("No DISCORD_WEBHOOK_URL configured — skipping alerts")
            return 0

        changes = self.db.get_unnotified_changes(
            severities=self.severities_to_alert()
        )
        if not changes:
            logger.info("No pending changes to alert on")
            return 0

        # One message per run rather than one per diff block — a page-wide
        # rewrite would otherwise spam the channel and hit Discord's rate limit.
        payload = self._build_payload(changes)
        change_ids = [c.id for c in changes]

        try:
            self._post(payload)
        except Exception as exc:
            logger.error("Discord alert failed: {}", exc)
            self.db.log_alert(
                channel="discord",
                status="failed",
                change_id=change_ids[0] if change_ids else None,
                error_message=str(exc),
            )
            return 0

        for change_id in change_ids:
            self.db.log_alert(
                channel="discord", status="sent", change_id=change_id
            )
        self.db.mark_changes_notified(change_ids)
        logger.info("Discord alert sent for {} change(s)", len(change_ids))
        return len(change_ids)

    def send_alert(self, change: dict, channel: str = "discord") -> bool:
        """
        Send a single ad-hoc alert. Returns True on success.

        `send_pending_alerts` is the normal entry point; this exists for
        manual triggering and testing.
        """
        if channel != "discord":
            raise ValueError(f"Unsupported alert channel: {channel}")
        if not self.enabled:
            logger.info("No DISCORD_WEBHOOK_URL configured — skipping alert")
            return False

        try:
            self._post(self._build_payload([_DictChange(change)]))
        except Exception as exc:
            logger.error("Discord alert failed: {}", exc)
            self.db.log_alert(
                channel="discord",
                status="failed",
                change_id=change.get("id"),
                error_message=str(exc),
            )
            return False

        self.db.log_alert(
            channel="discord", status="sent", change_id=change.get("id")
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, payload: dict) -> None:
        """POST to the Discord webhook, raising on a non-2xx response."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.webhook_url, json=payload)
            resp.raise_for_status()

    def _build_payload(self, changes) -> dict:
        """Build a Discord webhook payload with one embed per change."""
        counts: dict[str, int] = {}
        for change in changes:
            counts[change.severity] = counts.get(change.severity, 0) + 1

        headline = ", ".join(
            f"{SEVERITY_EMOJI.get(sev, '⚪')} {counts[sev]} {sev.lower()}"
            for sev in SEVERITY_ORDER[::-1]
            if sev in counts
        )

        # Discord allows at most 10 embeds per message.
        embeds = [self._build_embed(c) for c in list(changes)[:10]]
        if len(changes) > 10:
            embeds.append({
                "description": f"…and {len(changes) - 10} more change(s). "
                               "See the Changes page for the full list.",
                "color": SEVERITY_COLOR["MINOR"],
            })

        return {
            "username": "485 Visa Intelligence",
            "content": f"**Policy change detected** — {headline}",
            "embeds": embeds,
        }

    @staticmethod
    def _build_embed(change) -> dict:
        """Format a single change as a Discord embed."""
        severity = change.severity
        fields = []

        if change.old_value:
            fields.append({
                "name": "Before",
                "value": _truncate(change.old_value),
                "inline": False,
            })
        if change.new_value:
            fields.append({
                "name": "After",
                "value": _truncate(change.new_value),
                "inline": False,
            })

        embed = {
            "title": f"{SEVERITY_EMOJI.get(severity, '⚪')} {severity} — "
                     f"{change.change_type}",
            "description": change.summary or "No summary available.",
            "color": SEVERITY_COLOR.get(severity, 0x95A5A6),
            "fields": fields,
        }
        if change.source_url:
            embed["url"] = change.source_url
            embed["footer"] = {"text": change.source_url}
        return embed


def _truncate(text: str, limit: int = _MAX_FIELD_CHARS) -> str:
    """Trim text to fit a Discord embed field, wrapped as a code block."""
    text = text.strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n…"
    return f"```\n{text}\n```"


class _DictChange:
    """Adapts a plain change dict to the attribute access `_build_embed` uses."""

    def __init__(self, data: dict):
        self.id = data.get("id")
        self.severity = data.get("severity", "MINOR")
        self.change_type = data.get("change_type", "unknown")
        self.old_value = data.get("old_value")
        self.new_value = data.get("new_value")
        self.summary = data.get("summary")
        self.source_url = data.get("source_url")
