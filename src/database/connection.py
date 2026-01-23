"""Database connection management.

Provides:
- Connection pool management
- Session factory
- Async support via asyncio
- SQLite for development, PostgreSQL for production
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, AsyncGenerator, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..utils.logging import get_logger
from .models import Base

logger = get_logger(__name__)


class DatabaseConfig:
    """Database configuration."""

    def __init__(
        self,
        url: str = "sqlite:///polymarketbot.db",
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
    ):
        """Initialize database configuration.

        Args:
            url: Database URL (SQLite or PostgreSQL)
            echo: Echo SQL statements
            pool_size: Connection pool size
            max_overflow: Max connections above pool_size
            pool_timeout: Timeout waiting for connection
            pool_recycle: Recycle connections after seconds
        """
        self.url = url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL."""
        return self.url.startswith("postgresql")


class DatabaseManager:
    """Manages database connections and sessions.

    Example:
        db = DatabaseManager(config)
        db.initialize()

        with db.session() as session:
            account = session.query(TrackedAccount).first()

        db.close()
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize database manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._initialized = False

    @property
    def engine(self) -> Engine:
        """Get the database engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine

    def initialize(self) -> None:
        """Initialize database engine and create tables."""
        if self._initialized:
            return

        # Create engine with appropriate settings
        if self.config.is_sqlite:
            # SQLite specific settings
            self._engine = create_engine(
                self.config.url,
                echo=self.config.echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )

            # Enable foreign keys for SQLite
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

        else:
            # PostgreSQL settings
            self._engine = create_engine(
                self.config.url,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
            )

        # Create session factory
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

        # Create all tables
        Base.metadata.create_all(self._engine)

        self._initialized = True

        logger.info(
            "database_initialized",
            url=self._mask_url(self.config.url),
            is_sqlite=self.config.is_sqlite,
        )

    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of database URL."""
        if "@" in url:
            # Mask password in PostgreSQL URL
            parts = url.split("@")
            prefix = parts[0].rsplit(":", 1)[0]
            return f"{prefix}:***@{parts[1]}"
        return url

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session context manager.

        Yields:
            Database session

        Example:
            with db.session() as session:
                account = session.query(TrackedAccount).first()
        """
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new database session.

        Note: Caller is responsible for committing and closing.

        Returns:
            Database session
        """
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory()

    def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False

            logger.info("database_closed")

    def drop_all(self) -> None:
        """Drop all tables. USE WITH CAUTION."""
        if self._engine:
            Base.metadata.drop_all(self._engine)
            logger.warning("all_tables_dropped")

    def execute_raw(self, sql: str) -> None:
        """Execute raw SQL.

        Args:
            sql: SQL statement to execute
        """
        with self.engine.connect() as conn:
            conn.execute(sql)
            conn.commit()


# Global database instance
_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global database manager.

    Returns:
        DatabaseManager instance

    Raises:
        RuntimeError: If database not configured
    """
    if _db_manager is None:
        raise RuntimeError("Database not configured. Call configure_database() first.")
    return _db_manager


def configure_database(config: DatabaseConfig) -> DatabaseManager:
    """Configure and initialize the global database.

    Args:
        config: Database configuration

    Returns:
        DatabaseManager instance
    """
    global _db_manager

    if _db_manager is not None:
        _db_manager.close()

    _db_manager = DatabaseManager(config)
    _db_manager.initialize()

    return _db_manager


def close_database() -> None:
    """Close the global database connection."""
    global _db_manager

    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None
