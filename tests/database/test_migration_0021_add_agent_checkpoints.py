"""Tests for migration 0021: encrypted agent checkpoint tables."""

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from local_deep_research.database.alembic_runner import (
    get_head_revision,
    get_migrations_dir,
    run_migrations,
)


def test_0021_chains_after_0020():
    config = Config()
    config.set_main_option("script_location", str(get_migrations_dir()))
    script = ScriptDirectory.from_config(config)
    assert script.get_revision("0021").down_revision == "0020"


def test_head_revision_is_0021():
    assert get_head_revision() == "0021"


def test_upgrade_creates_agent_checkpoint_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'checkpoint.db'}")
    run_migrations(engine)
    tables = set(inspect(engine).get_table_names())
    assert {
        "agent_checkpoints",
        "agent_checkpoint_blobs",
        "agent_checkpoint_writes",
    }.issubset(tables)
    engine.dispose()
