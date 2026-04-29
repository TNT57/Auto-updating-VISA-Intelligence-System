"""Alert manager — Phase 3 (Weeks 5-6). Will handle Discord webhooks and email alerts."""


class AlertManager:
    """Manages alert delivery via multiple channels."""

    def __init__(self):
        raise NotImplementedError("Phase 3 — Coming in Weeks 5-6")

    def send_alert(self, change: dict, channel: str = "discord") -> bool:
        """Send an alert notification."""
        raise NotImplementedError