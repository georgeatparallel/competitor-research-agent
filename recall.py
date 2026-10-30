"""Where memory meets the pipeline: establish identity, reusing a recent run.

identify_company costs one LLM call and up to two searches. For a company
already researched under a known domain, that work has been done, and a
founder tracking the same competitor every month pays for it again every
time. This module decides when to reuse it instead.

It is deliberately NOT part of pipeline.py. pipeline.py stays memory-free,
so run_eval.py -- which calls it directly -- exercises the agent with no
stored history influencing the result. What keeps evaluation honest is the
layering, not a flag someone has to remember to set.

Two rules decide everything here.

STRONG KEYS ONLY. A key is strong only when it came from a domain. Reuse
needs that: "Sprout Design" naming one company today and a different one
next month is a collision this repo already contains twice. A weak key gets
no memory at all, not even as a hint. Offering the model a past run's guess
as a candidate would bias it toward repeating a misidentification, and an
ambiguous name is exactly where this pipeline is meant to stop rather than
guess.

FRESH ONLY, AND SAID OUT LOUD. The age limit lives in memory.py. A reused
identity is reported with the date it was established, in the terminal, in
the web UI and in the report, because a cached claim otherwise reads exactly
like a freshly established one.
"""

import memory
import pipeline
from schemas import CompanyIdentity


def _identity_from(recalled: dict) -> CompanyIdentity:
    """Rebuild a CompanyIdentity from a stored one.

    confident_match is True because only confident runs reach the ledger:
    both UIs stop before writing a report when identity is ambiguous, so a
    stored identity is one that already passed that gate.
    """
    return CompanyIdentity(
        description=recalled["description"],
        location=recalled.get("location"),
        website=recalled.get("website"),
        confident_match=True,
        ambiguity_note=None,
    )


def establish_identity(
    company_name: str,
    company_url: str,
    path: str | None = None,
) -> tuple[CompanyIdentity | None, dict | None]:
    """Identify the company, reusing a recent run when one can be trusted.

    Returns (identity, recalled). `recalled` is the stored record's payload
    when memory was used and None when the live step ran, so callers can
    label a reused identity with its date. `identity` is None only when the
    live step failed to produce one, exactly as before.
    """
    key, key_is_strong = memory.canonical_key(company_url, company_name)

    if key_is_strong:
        # Reads are best-effort, like the writes: an unreadable ledger must
        # cost the user a shortcut, not their run.
        try:
            recalled = memory.recall_identity(key, path)
        except OSError:
            recalled = None

        if recalled:
            return _identity_from(recalled), recalled

    return pipeline.identify_company(company_name, company_url), None
