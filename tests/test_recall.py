"""Tests for when a stored identity may replace the identify step.

Run: python -m unittest discover -s tests -v

This is the first place memory changes what the agent does, so the rules are
pinned down here rather than left to a reading of the code.

Three things these tests exist to protect:

1. The live identify step is SKIPPED only for a strong key with a recent
   record. pipeline.identify_company is replaced by a counter in every test
   below, so "did the agent make the call" is asserted directly rather than
   inferred.

2. A weak key gets no memory at all, not even as a hint. A name-based key
   can name a different company next month, and feeding the model a past
   guess would bias it toward repeating a misidentification on exactly the
   ambiguous names this pipeline is supposed to stop on.

3. A reused identity is handed back with its date, so the terminal, the web
   UI and the report can say the fact was not established today.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

# recall imports pipeline, which imports config, which reads these at import
# time. Dummies keep the tests hermetic: no real keys, no network.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

import memory  # noqa: E402
import recall  # noqa: E402

NOW = datetime.now(timezone.utc)


def stamp(days_ago):
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Establish(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "memory.jsonl")

        self.live_calls = []
        patcher = patch.object(recall.pipeline, "identify_company",
                               side_effect=self.fake_identify)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_identify(self, company_name, company_url):
        """Stand-in for the LLM step: records that it was reached."""
        self.live_calls.append((company_name, company_url))
        return SimpleNamespace(description="freshly identified",
                               location=None, website=None,
                               confident_match=True, ambiguity_note=None)

    def remember(self, days_ago=2, name="Notion", website="https://notion.so",
                 description="Workspace and notes tool", location="US"):
        memory.record_run(
            path=self.path, company_name=name, run_at=stamp(days_ago),
            identity=SimpleNamespace(description=description, location=location,
                                     website=website),
        )

    def establish(self, name="Notion", url="https://notion.so"):
        return recall.establish_identity(name, url, path=self.path)

    # --- the shortcut -----------------------------------------------------

    def test_a_recent_run_replaces_the_live_step(self):
        self.remember(days_ago=2)
        identity, recalled = self.establish()

        self.assertEqual(self.live_calls, [], "the LLM step should be skipped")
        self.assertEqual(identity.description, "Workspace and notes tool")
        self.assertEqual(identity.location, "US")
        self.assertEqual(identity.website, "https://notion.so")
        self.assertIsNotNone(recalled)

    def test_a_reused_identity_is_confident(self):
        """Only confident runs reach the ledger: both UIs stop before writing
        a report when identity is ambiguous."""
        self.remember()
        identity, _ = self.establish()
        self.assertTrue(identity.confident_match)
        self.assertIsNone(identity.ambiguity_note)

    def test_the_date_comes_back_so_it_can_be_labelled(self):
        self.remember(days_ago=5)
        _, recalled = self.establish()
        self.assertEqual(recalled["run_at"], stamp(5))
        self.assertAlmostEqual(recalled["age_days"], 5.0, places=1)

    def test_a_bare_domain_still_finds_the_history(self):
        """The key is canonical, so the URL need not be typed the same way."""
        self.remember()
        for url in ("notion.so", "https://www.notion.so/product",
                    "HTTPS://NOTION.SO/?utm=x"):
            with self.subTest(url=url):
                _, recalled = self.establish(url=url)
                self.assertIsNotNone(recalled)
        self.assertEqual(self.live_calls, [])

    # --- when the shortcut must not be taken ------------------------------

    def test_nothing_remembered_means_the_live_step_runs(self):
        identity, recalled = self.establish()
        self.assertEqual(len(self.live_calls), 1)
        self.assertIsNone(recalled)
        self.assertEqual(identity.description, "freshly identified")

    def test_a_stale_record_is_re_established(self):
        self.remember(days_ago=memory.RECALL_MAX_AGE_DAYS + 1)
        _, recalled = self.establish()
        self.assertEqual(len(self.live_calls), 1)
        self.assertIsNone(recalled)

    def test_a_weak_key_never_uses_memory(self):
        """No URL: the key is the name, which can mean a different company
        next month. This repo holds two different "Sprout Design" reports."""
        memory.record_run(path=self.path, company_name="Sprout Design",
                          run_at=stamp(1),
                          identity=SimpleNamespace(
                              description="A design agency", location="IN",
                              website=None))

        identity, recalled = recall.establish_identity(
            "Sprout Design", "", path=self.path)

        self.assertEqual(len(self.live_calls), 1)
        self.assertIsNone(recalled)
        self.assertEqual(identity.description, "freshly identified")

    def test_a_directory_url_never_uses_memory(self):
        """A LinkedIn URL cannot key a company, so it cannot recall one."""
        self.remember()
        _, recalled = self.establish(
            url="https://www.linkedin.com/company/notion")
        self.assertEqual(len(self.live_calls), 1)
        self.assertIsNone(recalled)

    def test_another_companys_history_is_not_reused(self):
        self.remember(name="Duolingo", website="https://duolingo.com")
        _, recalled = self.establish(name="Notion", url="https://notion.so")
        self.assertEqual(len(self.live_calls), 1)
        self.assertIsNone(recalled)

    def test_an_unreadable_ledger_costs_the_shortcut_not_the_run(self):
        with patch.object(recall.memory, "recall_identity",
                          side_effect=OSError("permission denied")):
            identity, recalled = self.establish()

        self.assertEqual(len(self.live_calls), 1)
        self.assertIsNone(recalled)
        self.assertEqual(identity.description, "freshly identified")

    def test_a_failed_live_step_still_returns_none(self):
        """Unchanged behaviour: the UIs stop when identity cannot be
        established, and memory must not paper over that."""
        with patch.object(recall.pipeline, "identify_company",
                          return_value=None):
            identity, recalled = self.establish()
        self.assertIsNone(identity)
        self.assertIsNone(recalled)


if __name__ == "__main__":
    unittest.main()
