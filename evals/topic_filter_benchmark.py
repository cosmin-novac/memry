"""Compare legacy JSON category filtering with normalized indexed topic links.

Offline and deterministic. The benchmark measures the same exact-match filter
against one generated SQLite dataset; it does not call an LLM or a network API.

Run: python evals/topic_filter_benchmark.py --memories 50000 --queries 300
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import time


def build_db(memory_count: int, topic_count: int, seed: int) -> sqlite3.Connection:
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE memories (id INTEGER PRIMARY KEY, categories TEXT NOT NULL);
        CREATE TABLE topics (id INTEGER PRIMARY KEY, normalized TEXT NOT NULL UNIQUE);
        CREATE TABLE memory_topics (
            memory_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            PRIMARY KEY (memory_id, topic_id)
        );
        CREATE INDEX idx_memory_topics_topic ON memory_topics(topic_id, memory_id);
        """
    )
    db.executemany(
        "INSERT INTO topics (id, normalized) VALUES (?, ?)",
        ((index, f"topic-{index}") for index in range(topic_count)),
    )
    memories: list[tuple[int, str]] = []
    links: list[tuple[int, int]] = []
    for memory_id in range(memory_count):
        topic_ids = rng.sample(range(topic_count), k=rng.randint(1, 5))
        categories = [f"topic-{topic_id}" for topic_id in topic_ids]
        memories.append((memory_id, json.dumps(categories)))
        links.extend((memory_id, topic_id) for topic_id in topic_ids)
    db.executemany("INSERT INTO memories (id, categories) VALUES (?, ?)", memories)
    db.executemany(
        "INSERT INTO memory_topics (memory_id, topic_id) VALUES (?, ?)", links
    )
    db.commit()
    return db


LEGACY_SQL = """
SELECT COUNT(*) FROM memories m
WHERE EXISTS (
    SELECT 1 FROM json_each(m.categories) item
    WHERE lower(trim(CAST(item.value AS TEXT))) = ?
)
"""

INDEXED_SQL = """
SELECT COUNT(*) FROM topics t
JOIN memory_topics mt ON mt.topic_id = t.id
WHERE t.normalized = ?
"""


def measure(db: sqlite3.Connection, sql: str, topics: list[str]) -> list[float]:
    timings: list[float] = []
    for topic in topics:
        started = time.perf_counter_ns()
        db.execute(sql, (topic,)).fetchone()
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", type=int, default=50_000)
    parser.add_argument("--topics", type=int, default=1_000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    db = build_db(args.memories, args.topics, args.seed)
    rng = random.Random(args.seed + 1)
    queries = [f"topic-{rng.randrange(args.topics)}" for _ in range(args.queries)]
    for sql in (LEGACY_SQL, INDEXED_SQL):
        measure(db, sql, queries[:20])
    legacy = measure(db, LEGACY_SQL, queries)
    indexed = measure(db, INDEXED_SQL, queries)

    legacy_median = statistics.median(legacy)
    indexed_median = statistics.median(indexed)
    result = {
        "memories": args.memories,
        "topics": args.topics,
        "queries": args.queries,
        "legacy_json_median_ms": round(legacy_median, 4),
        "indexed_links_median_ms": round(indexed_median, 4),
        "median_speedup": round(legacy_median / max(indexed_median, 1e-9), 2),
        "legacy_mean_ms": round(statistics.mean(legacy), 4),
        "indexed_mean_ms": round(statistics.mean(indexed), 4),
    }
    print(json.dumps(result, indent=2))
    db.close()


if __name__ == "__main__":
    main()