"""30-day sliding corpus retention helpers."""

from __future__ import annotations

import time

from CloudService.api.state import ACTIVE_STATUSES, RETENTION_SECONDS, STORE, Corpus, Job


def extend_corpus_on_reuse(corpus: Corpus) -> None:
    """Reset the 30-day sliding clock when a corpus is reused for a new job."""
    corpus.last_used_at = time.time()


def corpus_is_expired(corpus: Corpus, now: float | None = None) -> bool:
    """Return True when ``last_used_at`` is older than the retention window."""
    stamp = now if now is not None else time.time()
    return (stamp - corpus.last_used_at) > RETENTION_SECONDS


def corpus_pinned_by_active_job(corpus_id: str) -> bool:
    """Return True when an active job still references ``corpus_id``."""
    with STORE.lock:
        for job in STORE.jobs.values():
            if job.corpus_id == corpus_id and job.status in ACTIVE_STATUSES:
                return True
    return False


def sweep_expired_corpora(now: float | None = None) -> list[str]:
    """Delete expired corpora that are not pinned by an active job.

    Returns the corpus ids that were removed.
    """
    stamp = now if now is not None else time.time()
    removed: list[str] = []
    with STORE.lock:
        for corpus_id, corpus in list(STORE.corpora.items()):
            if not corpus_is_expired(corpus, stamp):
                continue
            pinned = any(
                job.corpus_id == corpus_id and job.status in ACTIVE_STATUSES
                for job in STORE.jobs.values()
            )
            if pinned:
                continue
            del STORE.corpora[corpus_id]
            removed.append(corpus_id)
        STORE.persist_hint()
    return removed


def require_live_corpus(corpus_id: str, customer_id: str) -> Corpus:
    """Return a retained corpus owned by ``customer_id``.

    Raises KeyError when missing and TimeoutError when expired.
    """
    with STORE.lock:
        corpus = STORE.corpora.get(corpus_id)
        if corpus is None or corpus.customer_id != customer_id:
            raise KeyError(corpus_id)
        if corpus_is_expired(corpus) and not corpus_pinned_by_active_job(corpus_id):
            raise TimeoutError(corpus_id)
        return corpus


def pin_job_corpus(job: Job) -> None:
    """Mark the job's corpus as used now (create or reuse)."""
    with STORE.lock:
        corpus = STORE.corpora.get(job.corpus_id)
        if corpus is not None:
            extend_corpus_on_reuse(corpus)
