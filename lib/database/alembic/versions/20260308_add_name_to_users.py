"""add name to users table

Revision ID: add_name_to_users
Revises: 98d0799e5eb6
Create Date: 2026-03-08 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_name_to_users'
down_revision = '98d0799e5eb6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # add nullable name column so existing rows are unaffected
    op.add_column('users', sa.Column('name', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'name')
