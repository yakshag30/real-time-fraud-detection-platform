"""Storage abstraction: SQLite for local work, PostgreSQL in Docker/production."""

from .repository import SQLRepository, build_repository

__all__ = ["SQLRepository", "build_repository"]
