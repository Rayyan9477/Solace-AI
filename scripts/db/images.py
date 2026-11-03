"""Pinned container image references for the data platform (IF-03).

``POSTGRES_IMAGE`` is the one PostgreSQL 16 reference used by every real-Postgres test, the compose
files (WS-03.17) and the backup image. It is pinned by digest, so a moved tag cannot change what runs.
To bump it, pull the new tag and paste the digest printed by
``docker image inspect --format "{{index .RepoDigests 0}}" postgres:16-alpine``.
"""
from __future__ import annotations

from typing import Final

POSTGRES_IMAGE: Final[str] = (
    "postgres:16-alpine@sha256:16bc17c64a573ef34162af9298258d1aec548232985b33ed7b1eac33ba35c229"
)

__all__ = ["POSTGRES_IMAGE"]
