"""
Tests for pipeline state persistence, alerting, and severity classification.

These cover the behaviours that make the "auto-updating" claim true:
snapshots surviving between runs, changes being alerted exactly once, and the
severity classifier actually seeing the new text.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def db(tmp_path):
    """A DatabaseManager backed by a throwaway SQLite file."""
    from src.utils.db_manager import DatabaseManager

    manager = DatabaseManager(db_path=str(tmp_path / "test.db"))
    manager.create_tables()
    return manager


# ══════════════════════════════════════════════════════════════════════
# Page snapshot persistence
# ══════════════════════════════════════════════════════════════════════

class TestPageSnapshots:
    """The state that makes day-over-day change detection possible."""

    def test_no_snapshot_returns_none(self, db):
        assert db.get_latest_page_snapshot("https://example.com/visa") is None

    def test_snapshot_round_trip(self, db):
        db.save_page_snapshot(
            url="https://example.com/visa",
            content="The fee is $1,895.",
            content_hash="abc123",
            title="Visa fees",
        )
        snapshot = db.get_latest_page_snapshot("https://example.com/visa")

        assert snapshot is not None
        assert snapshot["content"] == "The fee is $1,895."
        assert snapshot["content_hash"] == "abc123"
        assert snapshot["title"] == "Visa fees"

    def test_latest_snapshot_wins(self, db):
        url = "https://example.com/visa"
        db.save_page_snapshot(url=url, content="old", content_hash="h1")
        db.save_page_snapshot(url=url, content="new", content_hash="h2")

        assert db.get_latest_page_snapshot(url)["content"] == "new"

    def test_snapshots_are_isolated_per_url(self, db):
        db.save_page_snapshot(url="https://a.test", content="A", content_hash="a")
        db.save_page_snapshot(url="https://b.test", content="B", content_hash="b")

        assert db.get_latest_page_snapshot("https://a.test")["content"] == "A"
        assert db.get_latest_page_snapshot("https://b.test")["content"] == "B"

    def test_prune_keeps_most_recent(self, db):
        url = "https://example.com/visa"
        for i in range(8):
            db.save_page_snapshot(url=url, content=f"v{i}", content_hash=f"h{i}")

        deleted = db.prune_page_snapshots(url, keep=3)

        assert deleted == 5
        # The newest snapshot must survive pruning.
        assert db.get_latest_page_snapshot(url)["content"] == "v7"


class TestLoadPreviousContent:
    """
    Regression guard for the bug that made the scheduled pipeline useless.

    Previously this read gitignored HTML files off disk, so on an ephemeral CI
    runner the second run re-baselined instead of detecting a change.
    """

    def test_second_run_sees_the_first_runs_content(self, db):
        from scripts.daily_update import load_previous_content

        url = "https://immi.homeaffairs.gov.au/visas/temporary-graduate-485"

        # Run 1: nothing stored yet, so this is a baseline.
        assert load_previous_content(url, db) is None
        db.save_page_snapshot(url=url, content="Fee: $1,895", content_hash="h1")

        # Run 2: must see run 1's content rather than re-baselining.
        assert load_previous_content(url, db) == "Fee: $1,895"

    def test_detects_a_real_change_across_two_runs(self, db):
        from scripts.daily_update import load_previous_content
        from src.monitoring.change_detector import ChangeDetector

        detector = ChangeDetector(db=db, use_llm=False)
        url = "https://immi.homeaffairs.gov.au/visas/visa-fees"

        # Run 1 — baseline, no changes expected.
        first = detector.compare_and_record(
            old_content=load_previous_content(url, db),
            new_content="The visa fee is $1,895.",
            source_url=url,
        )
        db.save_page_snapshot(
            url=url, content="The visa fee is $1,895.", content_hash="h1"
        )
        assert first == []

        # Run 2 — the fee changed, so this must be detected and recorded.
        second = detector.compare_and_record(
            old_content=load_previous_content(url, db),
            new_content="The visa fee is $2,000.",
            source_url=url,
        )

        assert len(second) > 0
        assert len(db.get_recent_changes(limit=10)) == len(second)


# ══════════════════════════════════════════════════════════════════════
# Severity classification
# ══════════════════════════════════════════════════════════════════════

class TestSeverityClassification:
    """The classifier must see the new text, not only the old."""

    def test_new_text_drives_severity(self, db):
        """
        Regression: `old or "" + new or ""` parsed as `old or ("" + new)`,
        so for a replace the new text never reached the classifier.
        """
        from src.monitoring.change_detector import ChangeDetector

        detector = ChangeDetector(db=db, use_llm=False)

        # "Contact us" is MINOR; the replacement introduces an eligibility
        # requirement, which is CRITICAL. Only the new text carries the signal.
        changes = detector.compare(
            old_content="Contact us on our website.",
            new_content="You must meet the eligibility criteria.",
            source_url="https://example.com",
        )

        assert changes
        assert any(c["severity"] == "CRITICAL" for c in changes)

    def test_llm_classification_is_capped(self, db):
        """A page-wide rewrite must not fire one LLM call per diff block."""
        from src.monitoring.change_detector import ChangeDetector

        detector = ChangeDetector(db=db, use_llm=True)
        fake_llm = MagicMock()
        fake_llm.generate.return_value = "MINOR — formatting only"
        detector._llm_client = fake_llm

        old = "\n".join(f"old line {i}" for i in range(40))
        new = "\n".join(f"new line {i}" for i in range(40))

        with patch("src.monitoring.change_detector.settings") as mock_settings:
            mock_settings.max_llm_classifications = 3
            detector.compare(old, new, source_url="https://example.com")

        assert fake_llm.generate.call_count <= 3

    def test_keyword_fallback_when_llm_unavailable(self, db):
        from src.monitoring.change_detector import ChangeDetector

        detector = ChangeDetector(db=db, use_llm=False)
        assert detector.classify_severity("processing time updated") == "CRITICAL"
        assert detector.classify_severity("the application fee changed") == "IMPORTANT"
        assert detector.classify_severity("footer colour tweak") == "MINOR"


# ══════════════════════════════════════════════════════════════════════
# Run accounting
# ══════════════════════════════════════════════════════════════════════

class TestRunAccounting:
    """
    A run where every URL is blocked must not look successful.

    Regression: `pages_scraped = len(results)` counted failed fetches, so a
    run where all four URLs returned 403 reported "4 pages scraped" and
    skipped the non-zero exit.
    """

    def _blocked(self, url):
        return {"url": url, "error": "403 Forbidden", "content": None}

    def _ok(self, url):
        return {"url": url, "content": "Some page text", "content_hash": "h"}

    def test_all_blocked_counts_zero_scraped(self, db):
        import scripts.daily_update as du

        scraper = MagicMock()
        scraper.run_daily_scrape.return_value = [
            self._blocked(f"https://immi.test/{i}") for i in range(4)
        ]

        with patch.object(du, "DatabaseManager", return_value=db), \
             patch.object(du, "HomeAffairsScraper", return_value=scraper), \
             patch.object(du, "reingest_updated_pdfs", return_value=0), \
             patch.object(du, "ingest_scraped_pages", return_value=0):
            summary = du.run_daily_update()

        assert summary["pages_scraped"] == 0
        assert summary["pages_failed"] == 4
        assert len(summary["errors"]) == 4

    def test_partial_success_is_counted_accurately(self, db):
        import scripts.daily_update as du

        scraper = MagicMock()
        scraper.run_daily_scrape.return_value = [
            self._ok("https://immi.test/a"),
            self._blocked("https://immi.test/b"),
            self._ok("https://immi.test/c"),
        ]

        with patch.object(du, "DatabaseManager", return_value=db), \
             patch.object(du, "HomeAffairsScraper", return_value=scraper), \
             patch.object(du, "reingest_updated_pdfs", return_value=0), \
             patch.object(du, "ingest_scraped_pages", return_value=0):
            summary = du.run_daily_update()

        assert summary["pages_scraped"] == 2
        assert summary["pages_failed"] == 1

    def test_main_exits_nonzero_when_nothing_scraped(self):
        import scripts.daily_update as du

        blocked = {
            "pages_scraped": 0,
            "pages_failed": 4,
            "changes_detected": 0,
            "chunks_ingested": 0,
            "alerts_sent": 0,
            "errors": ["403"] * 4,
        }
        with patch.object(du, "run_daily_update", return_value=blocked):
            with pytest.raises(SystemExit) as exc_info:
                du.main([])

        assert exc_info.value.code == 1

    def test_main_exits_zero_on_a_healthy_run(self):
        import scripts.daily_update as du

        healthy = {
            "pages_scraped": 4,
            "pages_failed": 0,
            "changes_detected": 1,
            "chunks_ingested": 12,
            "alerts_sent": 1,
            "errors": [],
        }
        with patch.object(du, "run_daily_update", return_value=healthy):
            du.main([])  # must not raise SystemExit


class TestAlertsToggle:
    """
    Change notification is parked via ALERTS_ENABLED. Detection must keep
    working — only the outbound push is switched off.
    """

    def _scraper_returning(self, content):
        scraper = MagicMock()
        scraper.run_daily_scrape.return_value = [
            {"url": "https://immi.test/fees", "content": content,
             "content_hash": "h1", "title": "Fees"}
        ]
        return scraper

    def _run(self, db, alerts_enabled):
        import scripts.daily_update as du

        with patch.object(du, "DatabaseManager", return_value=db), \
             patch.object(du, "HomeAffairsScraper",
                          return_value=self._scraper_returning("Fee is $2,500.")), \
             patch.object(du, "reingest_updated_pdfs", return_value=0), \
             patch.object(du, "ingest_scraped_pages", return_value=0), \
             patch.object(du.settings, "alerts_enabled", alerts_enabled), \
             patch("src.alerts.alert_manager.AlertManager.send_pending_alerts",
                   return_value=3) as mock_send:
            summary = du.run_daily_update()
        return summary, mock_send

    def test_alerts_skipped_when_disabled(self, db):
        summary, mock_send = self._run(db, alerts_enabled=False)

        mock_send.assert_not_called()
        assert summary["alerts_sent"] == 0
        assert not summary["errors"]

    def test_alerts_sent_when_enabled(self, db):
        summary, mock_send = self._run(db, alerts_enabled=True)

        mock_send.assert_called_once()
        assert summary["alerts_sent"] == 3

    def test_change_detection_still_records_while_alerts_are_off(self, db):
        """The parked feature must not disable detection itself."""
        db.save_page_snapshot(
            url="https://immi.test/fees", content="Fee is $2,235.",
            content_hash="h0",
        )
        summary, _ = self._run(db, alerts_enabled=False)

        assert summary["changes_detected"] > 0
        assert len(db.get_recent_changes(limit=10)) > 0

    def test_alerts_default_to_off(self):
        from src.utils.config import Settings

        assert Settings().alerts_enabled is False


class TestDetectOnlyAndCostControl:
    """
    Re-indexing needs PyTorch and a 420MB model. The monitored pages change
    rarely, so a run that finds nothing new must not pay that cost — the CI
    workflow branches on the `changed` output to decide whether to install
    the ML dependencies at all.
    """

    def _run(self, db, content, detect_only=False, previous=None):
        import scripts.daily_update as du

        if previous is not None:
            db.save_page_snapshot(url="https://immi.test/fees",
                                  content=previous, content_hash="h0")

        scraper = MagicMock()
        scraper.run_daily_scrape.return_value = [{
            "url": "https://immi.test/fees", "content": content,
            "content_hash": "h1", "title": "Fees",
        }]

        with patch.object(du, "DatabaseManager", return_value=db), \
             patch.object(du, "HomeAffairsScraper", return_value=scraper), \
             patch.object(du, "reingest_updated_pdfs", return_value=7) as pdfs, \
             patch.object(du, "ingest_scraped_pages", return_value=9) as pages:
            summary = du.run_daily_update(detect_only=detect_only)
        return summary, pdfs, pages

    def test_no_change_skips_reindexing(self, db):
        summary, pdfs, pages = self._run(
            db, content="Fee is $2,235.", previous="Fee is $2,235."
        )

        assert summary["changes_detected"] == 0
        pdfs.assert_not_called()
        pages.assert_not_called()
        assert summary["chunks_ingested"] == 0

    def test_a_real_change_triggers_reindexing(self, db):
        summary, pdfs, pages = self._run(
            db, content="Fee is $2,500.", previous="Fee is $2,235."
        )

        assert summary["changes_detected"] > 0
        pdfs.assert_called_once()
        pages.assert_called_once()
        assert summary["chunks_ingested"] == 16

    def test_detect_only_never_reindexes(self, db):
        """Even with a change, --detect-only leaves embedding to a later step."""
        summary, pdfs, pages = self._run(
            db, content="Fee is $2,500.", previous="Fee is $2,235.",
            detect_only=True,
        )

        assert summary["changes_detected"] > 0
        pdfs.assert_not_called()
        pages.assert_not_called()

    def test_github_output_reports_whether_anything_changed(self, tmp_path,
                                                            monkeypatch):
        import scripts.daily_update as du

        out = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))

        du._emit_github_output({"changes_detected": 3, "pages_scraped": 4})
        assert "changed=true" in out.read_text()

        out.write_text("")
        du._emit_github_output({"changes_detected": 0, "pages_scraped": 4})
        assert "changed=false" in out.read_text()

    def test_github_output_is_skipped_outside_ci(self, monkeypatch):
        import scripts.daily_update as du

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        du._emit_github_output({"changes_detected": 1, "pages_scraped": 1})


class TestUserAgent:
    """
    The WAF returns 403 for any User-Agent containing "Bot" or "Crawler".
    That single rule — not headers, not the source IP — caused four months of
    failed scheduled runs. Verified by trial against the live site 2026-08-13:
    "Visa485IntelligenceBot/1.0" and "MyCrawler/1.0" both 403 on every attempt,
    while "VisaWatch/1.0", curl and python-httpx all return 200.
    """

    FORBIDDEN = ("bot", "crawler", "spider", "scraper")

    def test_default_user_agent_avoids_blocked_keywords(self):
        from src.utils.config import Settings

        ua = Settings().user_agent.lower()
        for word in self.FORBIDDEN:
            assert word not in ua, (
                f"default User-Agent contains {word!r}; the Home Affairs WAF "
                f"returns 403 for those. Got: {ua}"
            )

    def test_default_user_agent_still_identifies_the_client(self):
        """Dodging the keyword filter must not mean impersonating a browser."""
        from src.utils.config import Settings

        ua = Settings().user_agent
        assert "Visa485Intelligence" in ua
        assert "github.com" in ua
        assert "Mozilla" not in ua

    def test_standard_headers_are_sent(self):
        from src.scraping.homeaffairs_scraper import HomeAffairsScraper

        with patch("httpx.Client") as mock_client:
            HomeAffairsScraper(fetcher=MagicMock(name="fetcher"))

        headers = mock_client.call_args.kwargs["headers"]
        assert "text/html" in headers["Accept"]
        assert "Accept-Language" in headers
        assert headers["User-Agent"]


# ══════════════════════════════════════════════════════════════════════
# Scraped page ingestion
# ══════════════════════════════════════════════════════════════════════

class TestIngestScrapedPages:
    """Monitored pages must be searchable, not just diffed."""

    def _result(self, url="https://immi.test/fees", content="The fee is $2,235."):
        return {
            "url": url,
            "title": "Visa fees",
            "content": content,
            "content_hash": "h1",
            "timestamp": "2026-01-01T00:00:00",
        }

    def test_pages_are_chunked_and_added(self):
        from scripts.daily_update import ingest_scraped_pages

        fake_store = MagicMock()
        with patch("src.ingestion.vectorstore_manager.VectorStoreManager",
                   return_value=fake_store):
            added = ingest_scraped_pages([self._result()])

        assert added > 0
        fake_store.add_chunks.assert_called_once()

        chunks = fake_store.add_chunks.call_args[0][0]
        assert all(c.doc_type == "webpage" for c in chunks)
        # source must be the URL so deletes stay stable across title changes
        assert all(c.source == "https://immi.test/fees" for c in chunks)
        assert chunks[0].metadata["title"] == "Visa fees"

    def test_stale_chunks_are_deleted_before_reinsert(self):
        from scripts.daily_update import ingest_scraped_pages

        fake_store = MagicMock()
        with patch("src.ingestion.vectorstore_manager.VectorStoreManager",
                   return_value=fake_store):
            ingest_scraped_pages([self._result()])

        fake_store.delete_document.assert_called_once_with(
            "https://immi.test/fees"
        )

    def test_failed_and_empty_scrapes_are_skipped(self):
        from scripts.daily_update import ingest_scraped_pages

        fake_store = MagicMock()
        results = [
            {"url": "https://a.test", "error": "timeout", "content": None},
            {"url": "https://b.test", "content": "   "},
        ]
        with patch("src.ingestion.vectorstore_manager.VectorStoreManager",
                   return_value=fake_store):
            assert ingest_scraped_pages(results) == 0

        fake_store.add_chunks.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# Alerting
# ══════════════════════════════════════════════════════════════════════

class TestAlertManager:
    """Discord delivery, deduplication, and failure logging."""

    def _record(self, db, severity="CRITICAL"):
        return db.record_change(
            severity=severity,
            change_type="content_update",
            old_value="fee is $1,895",
            new_value="fee is $2,000",
            summary="Visa fee increased",
            source_url="https://immi.homeaffairs.gov.au/visa-fees",
        )

    def test_disabled_without_webhook(self, db):
        from src.alerts.alert_manager import AlertManager

        manager = AlertManager(db=db, webhook_url="")
        assert manager.enabled is False
        assert manager.send_pending_alerts() == 0

    def test_sends_and_marks_notified(self, db):
        from src.alerts.alert_manager import AlertManager

        self._record(db)
        manager = AlertManager(db=db, webhook_url="https://discord.test/hook")

        with patch.object(manager, "_post") as mock_post:
            sent = manager.send_pending_alerts()

        assert sent == 1
        mock_post.assert_called_once()

        # Second run must not re-alert on the same change.
        with patch.object(manager, "_post") as mock_post:
            assert manager.send_pending_alerts() == 0
            mock_post.assert_not_called()

    def test_minor_changes_are_not_alerted_by_default(self, db):
        from src.alerts.alert_manager import AlertManager

        self._record(db, severity="MINOR")
        manager = AlertManager(db=db, webhook_url="https://discord.test/hook")

        with patch.object(manager, "_post") as mock_post:
            assert manager.send_pending_alerts() == 0
            mock_post.assert_not_called()

    def test_failure_is_logged_and_change_stays_pending(self, db):
        from src.alerts.alert_manager import AlertManager
        from src.utils.db_manager import AlertLog

        self._record(db)
        manager = AlertManager(db=db, webhook_url="https://discord.test/hook")

        with patch.object(manager, "_post", side_effect=RuntimeError("503")):
            assert manager.send_pending_alerts() == 0

        with db.get_session() as session:
            logs = session.query(AlertLog).all()
            assert len(logs) == 1
            assert logs[0].status == "failed"
            assert "503" in logs[0].error_message

        # A failed send must leave the change eligible for the next run.
        assert len(db.get_unnotified_changes()) == 1

    def test_payload_shape(self, db):
        from src.alerts.alert_manager import AlertManager

        self._record(db)
        manager = AlertManager(db=db, webhook_url="https://discord.test/hook")
        payload = manager._build_payload(db.get_unnotified_changes())

        assert "content" in payload
        assert len(payload["embeds"]) == 1
        embed = payload["embeds"][0]
        assert "CRITICAL" in embed["title"]
        assert embed["description"] == "Visa fee increased"
        field_text = " ".join(f["value"] for f in embed["fields"])
        assert "$1,895" in field_text
        assert "$2,000" in field_text

    def test_payload_caps_embeds_at_discord_limit(self, db):
        from src.alerts.alert_manager import AlertManager

        for _ in range(14):
            self._record(db)
        manager = AlertManager(db=db, webhook_url="https://discord.test/hook")
        payload = manager._build_payload(db.get_unnotified_changes())

        # 10 change embeds plus one "and N more" overflow embed.
        assert len(payload["embeds"]) == 11

    def test_severity_threshold_is_configurable(self, db):
        from src.alerts.alert_manager import AlertManager

        manager = AlertManager(db=db, webhook_url="https://discord.test/hook")

        with patch("src.alerts.alert_manager.settings") as mock_settings:
            mock_settings.alert_min_severity = "CRITICAL"
            assert manager.severities_to_alert() == ("CRITICAL",)

            mock_settings.alert_min_severity = "MINOR"
            assert set(manager.severities_to_alert()) == {
                "MINOR", "IMPORTANT", "CRITICAL"
            }
