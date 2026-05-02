"""expand job_metadata inventory columns

Revision ID: expand_job_metadata_inventory
Revises: add_name_to_users
Create Date: 2026-04-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "expand_job_metadata_inventory"
down_revision = "add_name_to_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("job_metadata", sa.Column("text_full", sa.Text(), nullable=True))
    op.add_column(
        "job_metadata",
        sa.Column("skills_found", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "job_metadata",
        sa.Column("must_have_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "job_metadata",
        sa.Column("nice_to_have_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "job_metadata",
        sa.Column("core_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "job_metadata",
        sa.Column("role_tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("job_metadata", sa.Column("source_keyword", sa.String(), nullable=True))
    op.add_column("job_metadata", sa.Column("scraped_at", sa.DateTime(), nullable=True))
    op.add_column(
        "job_metadata",
        sa.Column("extraction_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        op.f("ix_job_metadata_scraped_at"),
        "job_metadata",
        ["scraped_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_job_metadata_scraped_at"), table_name="job_metadata")
    op.drop_column("job_metadata", "extraction_metadata")
    op.drop_column("job_metadata", "scraped_at")
    op.drop_column("job_metadata", "source_keyword")
    op.drop_column("job_metadata", "role_tags")
    op.drop_column("job_metadata", "core_skills")
    op.drop_column("job_metadata", "nice_to_have_skills")
    op.drop_column("job_metadata", "must_have_skills")
    op.drop_column("job_metadata", "skills_found")
    op.drop_column("job_metadata", "text_full")
