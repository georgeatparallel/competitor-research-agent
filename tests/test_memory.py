"""Tests for the memory key: which company a stored run belongs to.

Run: python -m unittest discover -s tests -v

No API keys are set up here, unlike the other test modules: memory.py is
deliberately standard-library only, so it needs no client and no network.

Three things these tests exist to protect:

1. One company produces ONE key. Every spelling a founder might type --
   bare domain, trailing slash, capitals, tracking query, stray spaces --
   has to collapse to the same string. A key that drifts does not raise an
   error; it silently splits one company's history in two, and the Stage 3
   diff then compares a company against nothing.

2. A key derived from a name is marked weak, and one that could not be
   derived at all is empty. Later stages are allowed to skip the identify
   call on a strong key, so "I know this domain" and "I guessed from a
   name" must never be the same shape. app.py already invites a LinkedIn
   URL, and "linkedin.com" as a key would file every company researched
   that way into one shared bucket.

3. Free text in the URL box cannot become a strong key. Both UIs accept
   whatever is typed, and "hello" parses into a non-empty netloc.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import memory

NOTION = ("notion.so", True)


def identity(website=None, description="A workspace company", location="US"):
    """Stand-in for a CompanyIdentity.

    memory.py reads these fields with getattr and never imports schemas.py,
    so a plain namespace is enough -- no pydantic, no API key.
    """
    return SimpleNamespace(description=description, location=location,
                           website=website)


def competitor(name, positioning="Cheaper and simpler", pricing="$10/mo",
               sources=("https://example.com",)):
    return SimpleNamespace(name=name, positioning=positioning, pricing=pricing,
                           sources=list(sources))


class StrongKeys(unittest.TestCase):
    def test_every_spelling_of_one_domain_collapses(self):
        spellings = (
            "https://www.notion.so",
            "https://www.notion.so/",
            "https://notion.so/product",
            "http://notion.so",
            "notion.so",
            "www.notion.so/product?utm_source=x",
            "HTTPS://WWW.NOTION.SO/?utm=x",
            "  https://notion.so  ",
        )
        for website in spellings:
            with self.subTest(website=website):
                self.assertEqual(memory.canonical_key(website, "Notion"), NOTION)

    def test_the_name_does_not_affect_a_strong_key(self):
        """Drifting names must not produce drifting keys -- the whole point."""
        for name in ("Notion", "Notion Labs", "notion.so", "NOTION, Inc."):
            with self.subTest(name=name):
                self.assertEqual(
                    memory.canonical_key("https://www.notion.so", name), NOTION)

    def test_credentials_and_port_are_stripped(self):
        self.assertEqual(
            memory.canonical_key("https://user:pw@notion.so:8443/x", "Notion"),
            NOTION)

    def test_www_is_only_stripped_as_a_prefix(self):
        """"wwwfoo.com" is a different company from "foo.com"."""
        self.assertEqual(memory.canonical_key("https://wwwfoo.com", "Foo"),
                         ("wwwfoo.com", True))

    def test_multi_label_tld_is_kept_whole(self):
        """Accepted limitation: no public-suffix list, so co.uk stays intact.

        Correct for "foo.co.uk" itself. It would only matter for a
        subdomain of one, and the raw website is stored in the record, so
        the ledger can be re-keyed later without re-running anything.
        """
        self.assertEqual(memory.canonical_key("https://foo.co.uk", "Foo"),
                         ("foo.co.uk", True))

    def test_subdomains_other_than_www_are_kept(self):
        """Also an accepted limitation, documented so a change is deliberate."""
        self.assertEqual(memory.canonical_key("https://blog.notion.so", "Notion"),
                         ("blog.notion.so", True))


class WeakKeys(unittest.TestCase):
    def test_no_website_falls_back_to_the_name(self):
        for website in (None, "", "   "):
            with self.subTest(website=website):
                self.assertEqual(
                    memory.canonical_key(website, "Sprout Design Pvt Ltd"),
                    ("name:sprout-design-pvt-ltd", False))

    def test_name_slug_matches_the_report_filename_scheme(self):
        """Same scheme as save_report() in report.py, asserted literally here
        rather than by importing it: report.py pulls in config and its API
        keys, and this module is meant to stay free of that."""
        self.assertEqual(memory.canonical_key("", "Sprout Design Pvt. Ltd")[0],
                         "name:sprout-design-pvt-ltd")

    def test_free_text_is_not_a_domain(self):
        """"hello" has a non-empty netloc once a scheme is added, so without
        a sanity check it would become a strong key resting on a typo."""
        for website in ("hello", "not a url", ".", "..", "http://"):
            with self.subTest(website=website):
                self.assertEqual(memory.canonical_key(website, "Notion"),
                                 ("name:notion", False))

    def test_an_unidentifiable_company_has_no_key(self):
        """Empty, not "name:company": a shared bucket for every nameless run
        would look like one company with an incoherent history. record_run
        refuses to file a record with an empty key instead."""
        self.assertEqual(memory.canonical_key(None, ""), ("", False))
        self.assertEqual(memory.canonical_key("", "   "), ("", False))
        self.assertEqual(memory.canonical_key(None, "!!!"), ("", False))


class DirectoryHosts(unittest.TestCase):
    def test_every_known_directory_host_is_refused(self):
        for host in memory.DIRECTORY_HOSTS:
            with self.subTest(host=host):
                key, strong = memory.canonical_key(f"https://{host}/company/x",
                                                   "Notion")
                self.assertEqual((key, strong), ("name:notion", False))

    def test_www_and_regional_subdomains_are_refused_too(self):
        for website in ("https://www.linkedin.com/company/notion",
                        "https://uk.linkedin.com/company/notion",
                        "uk.linkedin.com/company/notion"):
            with self.subTest(website=website):
                self.assertEqual(memory.canonical_key(website, "Notion"),
                                 ("name:notion", False))

    def test_a_company_domain_that_merely_resembles_one_is_kept(self):
        """Suffix matching must not swallow a real company's domain."""
        for website, expected in (("https://mylinkedin.com", "mylinkedin.com"),
                                  ("https://linkedin.com.au", "linkedin.com.au")):
            with self.subTest(website=website):
                self.assertEqual(memory.canonical_key(website, "Thing"),
                                 (expected, True))


class KeyShapes(unittest.TestCase):
    """The three outcomes have to be tellable apart by shape alone."""

    def test_a_strong_key_is_never_prefixed(self):
        key, strong = memory.canonical_key("https://notion.so", "Notion")
        self.assertTrue(strong)
        self.assertFalse(key.startswith("name:"))
        self.assertTrue(key)

    def test_a_weak_key_is_always_prefixed(self):
        key, strong = memory.canonical_key("", "Notion")
        self.assertFalse(strong)
        self.assertTrue(key.startswith("name:"))

    def test_strong_and_weak_keys_for_one_company_never_collide(self):
        strong_key, _ = memory.canonical_key("https://notion.so", "Notion")
        weak_key, _ = memory.canonical_key("", "Notion")
        self.assertNotEqual(strong_key, weak_key)


class Records(unittest.TestCase):
    """build_record is pure, so these need no files and no clock."""

    def record(self, **overrides):
        fields = dict(company_name="Notion", company_url="",
                      identity=identity("https://www.notion.so"),
                      run_at="2026-10-30T09:00:00Z")
        fields.update(overrides)
        return memory.build_record(**fields)

    def test_the_identity_website_outranks_the_typed_url(self):
        """identify_company anchors identity to a real page; that resolved
        website is better evidence than whatever was typed in the box."""
        record = self.record(company_url="https://www.linkedin.com/company/notion")
        self.assertEqual((record["key"], record["key_is_strong"]), NOTION)

    def test_the_typed_url_is_used_when_identity_has_no_website(self):
        record = self.record(identity=identity(None),
                             company_url="https://notion.so")
        self.assertEqual((record["key"], record["key_is_strong"]), NOTION)

    def test_exclusions_stay_split_by_who_decided(self):
        """Stage 4 reads excluded_by_human. If the two were merged it would
        have to parse "Removed by human reviewer." out of English prose."""
        record = self.record(
            excluded_auto=[("Acme", "Too small to compete at this scale.")],
            removed_by_human=["Obsidian"],
        )
        self.assertEqual(record["excluded_auto"],
                         [["Acme", "Too small to compete at this scale."]])
        self.assertEqual(record["excluded_by_human"], ["Obsidian"])

    def test_competitor_fields_a_diff_will_need_are_kept(self):
        record = self.record(competitors=[competitor("Coda")])
        self.assertEqual(record["competitors"], [{
            "name": "Coda",
            "positioning": "Cheaper and simpler",
            "pricing": "$10/mo",
            "sources": ["https://example.com"],
        }])

    def test_a_missing_pricing_survives_as_null(self):
        """Competitor.pricing is Optional, so None has to serialise."""
        record = self.record(competitors=[competitor("Coda", pricing=None)])
        self.assertIsNone(record["competitors"][0]["pricing"])
        self.assertIn('"pricing": null', json.dumps(record, sort_keys=True))

    def test_provenance_is_recorded(self):
        """Which model and which retrieval backend produced this run.

        Without it, a future diff cannot distinguish "the market changed"
        from "I switched Tavily for Firecrawl"."""
        record = self.record(model="gemini-flash-lite-latest",
                             retrieval_provider="tavily")
        self.assertEqual(record["model"], "gemini-flash-lite-latest")
        self.assertEqual(record["retrieval_provider"], "tavily")

    def test_every_record_is_versioned(self):
        """Stage 3 will change this shape; old lines must stay readable."""
        self.assertEqual(self.record()["schema_version"], memory.SCHEMA_VERSION)

    def test_the_timestamp_is_utc_when_not_supplied(self):
        record = memory.build_record(company_name="Notion",
                                     identity=identity("https://notion.so"))
        self.assertTrue(record["run_at"].endswith("Z"), record["run_at"])
        self.assertEqual(len(record["run_at"]), 20, record["run_at"])

    def test_a_record_is_json_serialisable(self):
        """It is about to be written as one line; anything exotic fails here
        rather than at the end of a 90-second run."""
        json.dumps(self.record(competitors=[competitor("Coda")]))


class Ledger(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "memory.jsonl")

    def write(self, name="Notion", website="https://notion.so", **overrides):
        return memory.record_run(path=self.path, company_name=name,
                                 identity=identity(website), **overrides)

    def test_a_run_round_trips(self):
        self.write(run_at="2026-10-30T09:00:00Z")
        runs = memory.read_runs("notion.so", self.path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["company_name"], "Notion")

    def test_each_run_is_one_line(self):
        self.write(run_at="2026-10-30T09:00:00Z")
        self.write(run_at="2026-10-31T09:00:00Z")
        with open(self.path, encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            json.loads(line)  # each line stands alone

    def test_history_comes_back_newest_first(self):
        self.write(run_at="2026-09-01T09:00:00Z")
        self.write(run_at="2026-10-30T09:00:00Z")
        runs = memory.read_runs("notion.so", self.path)
        self.assertEqual([run["run_at"] for run in runs],
                         ["2026-10-30T09:00:00Z", "2026-09-01T09:00:00Z"])

    def test_other_companies_are_not_returned(self):
        self.write("Notion", "https://notion.so")
        self.write("Duolingo", "https://duolingo.com")
        self.assertEqual(len(memory.read_runs("notion.so", self.path)), 1)
        self.assertEqual(memory.read_runs("coda.io", self.path), [])

    def test_no_ledger_yet_is_not_an_error(self):
        missing = os.path.join(self.directory.name, "nothing.jsonl")
        self.assertEqual(memory.read_runs("notion.so", missing), [])

    def test_a_truncated_last_line_does_not_lose_the_history(self):
        """An append-only file interrupted mid-write leaves a partial line.
        Losing every past run to it would be the worse failure."""
        self.write(run_at="2026-10-30T09:00:00Z")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"key": "notion.so", "run_at": "2026-10-31')
        self.assertEqual(len(memory.read_runs("notion.so", self.path)), 1)

    def test_an_unidentifiable_run_is_not_filed(self):
        result = memory.record_run(path=self.path, company_name="",
                                   identity=identity(None))
        self.assertIsNone(result)
        self.assertFalse(os.path.exists(self.path))

    def test_an_empty_key_reads_back_nothing(self):
        self.write()
        self.assertEqual(memory.read_runs("", self.path), [])

    def test_a_missing_directory_is_created(self):
        nested = os.path.join(self.directory.name, "state", "memory.jsonl")
        memory.record_run(path=nested, company_name="Notion",
                          identity=identity("https://notion.so"))
        self.assertTrue(os.path.exists(nested))

    def test_non_ascii_names_stay_readable_in_the_file(self):
        self.write("Café Söder", "https://cafesoder.se")
        with open(self.path, encoding="utf-8") as handle:
            self.assertIn("Café Söder", handle.read())

    def test_memory_path_env_var_is_honoured(self):
        previous = os.environ.get(memory.LEDGER_PATH_ENV)
        os.environ[memory.LEDGER_PATH_ENV] = self.path
        try:
            memory.record_run(company_name="Notion",
                              identity=identity("https://notion.so"))
        finally:
            if previous is None:
                del os.environ[memory.LEDGER_PATH_ENV]
            else:
                os.environ[memory.LEDGER_PATH_ENV] = previous
        self.assertEqual(len(memory.read_runs("notion.so", self.path)), 1)


NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)


def stamp(days_ago):
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Recall(unittest.TestCase):
    """Stage 2 reads the ledger, so these are the first tests of memory
    actually influencing a run. The clock is injected; a test that depends
    on the real one would start failing on its own in a month."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "memory.jsonl")

    def write(self, days_ago=1, website="https://notion.so", name="Notion",
              description="Workspace and notes tool"):
        return memory.record_run(
            path=self.path, company_name=name, run_at=stamp(days_ago),
            identity=identity(website, description=description),
        )

    def recall(self, key="notion.so", **kwargs):
        return memory.recall_identity(key, self.path, now=NOW, **kwargs)

    def test_nothing_remembered_is_not_an_error(self):
        self.assertIsNone(self.recall())

    def test_a_recent_identity_comes_back(self):
        self.write(days_ago=3)
        recalled = self.recall()
        self.assertEqual(recalled["description"], "Workspace and notes tool")
        self.assertEqual(recalled["website"], "https://notion.so")
        self.assertAlmostEqual(recalled["age_days"], 3.0, places=3)

    def test_a_stale_identity_is_refused(self):
        """Older than RECALL_MAX_AGE_DAYS: companies rebrand, move and get
        acquired, and every later step is told which company this is."""
        self.write(days_ago=memory.RECALL_MAX_AGE_DAYS + 1)
        self.assertIsNone(self.recall())

    def test_the_age_limit_is_adjustable(self):
        self.write(days_ago=60)
        self.assertIsNone(self.recall())
        self.assertIsNotNone(self.recall(max_age_days=90))

    def test_the_newest_run_wins(self):
        self.write(days_ago=20, description="Old description")
        self.write(days_ago=2, description="Current description")
        self.assertEqual(self.recall()["description"], "Current description")

    def test_an_unreadable_timestamp_is_not_reused(self):
        """Unknown age is not the same as fresh."""
        memory.append_record(
            memory.build_record(company_name="Notion", run_at="last Tuesday",
                                identity=identity("https://notion.so")),
            self.path,
        )
        self.assertIsNone(self.recall())

    def test_an_unreadable_timestamp_does_not_hide_a_good_record(self):
        self.write(days_ago=2)
        memory.append_record(
            memory.build_record(company_name="Notion", run_at="whenever",
                                identity=identity("https://notion.so")),
            self.path,
        )
        self.assertIsNotNone(self.recall())

    def test_a_record_with_no_description_is_skipped(self):
        self.write(days_ago=5, description="")
        self.assertIsNone(self.recall())

    def test_another_companys_history_is_not_returned(self):
        self.write(days_ago=1, website="https://duolingo.com", name="Duolingo")
        self.assertIsNone(self.recall("notion.so"))

    def test_an_empty_key_recalls_nothing(self):
        self.write(days_ago=1)
        self.assertIsNone(self.recall(""))

    def test_the_provenance_of_the_key_is_reported(self):
        """A caller has to know whether this came from a domain or a guess
        at a name before deciding how much to trust it."""
        self.write(days_ago=1)
        self.assertTrue(self.recall()["key_is_strong"])

        weak = memory.record_run(path=self.path, company_name="Sprout Design",
                                 run_at=stamp(1), identity=identity(None))
        self.assertFalse(
            memory.recall_identity(weak["key"], self.path, now=NOW)["key_is_strong"])

    def test_the_run_date_comes_back_for_the_report(self):
        """A reused identity has to be labelled with its date, or a stale
        fact reads as a fresh one."""
        self.write(days_ago=4)
        self.assertEqual(self.recall()["run_at"], stamp(4))


class Ages(unittest.TestCase):
    def test_age_is_measured_in_days(self):
        record = {"run_at": stamp(2.5)}
        self.assertAlmostEqual(memory.record_age_days(record, NOW), 2.5, places=3)

    def test_an_unparseable_age_is_none(self):
        for value in ("", None, "2026-10-30", "last week", 17):
            with self.subTest(value=value):
                self.assertIsNone(memory.record_age_days({"run_at": value}, NOW))

    def test_a_record_written_now_is_about_zero_days_old(self):
        record = memory.build_record(company_name="Notion",
                                     identity=identity("https://notion.so"))
        self.assertLess(memory.record_age_days(record), 1.0)


if __name__ == "__main__":
    unittest.main()
