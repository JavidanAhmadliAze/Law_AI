"""inline transcript on conversations, drop messages table

Messages were only ever read as a whole conversation, never queried on their
own, so the child table is folded into a JSONB column. Existing rows are carried
across in append order before the table is dropped.

Revision ID: 7c1f2a9d40b3
Revises: 48a696bc414d
Create Date: 2026-09-08 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7c1f2a9d40b3"
down_revision: str | None = "48a696bc414d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "messages",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    # carry the existing transcripts over, oldest first
    op.execute(
        """
        UPDATE conversations c
        SET messages = COALESCE(
            (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'id', m.id::text,
                        'role', m.role,
                        'content', m.content,
                        'created_at', to_char(
                            m.created_at AT TIME ZONE 'UTC',
                            'YYYY-MM-DD"T"HH24:MI:SS.US+00:00'
                        )
                    )
                    ORDER BY m.created_at
                )
                FROM messages m
                WHERE m.conversation_id = c.id
            ),
            '[]'::jsonb
        )
        """
    )
    op.drop_index(op.f("ix_messages_conversation_id"), table_name="messages")
    op.drop_table("messages")


def downgrade() -> None:
    op.create_table(
        "messages",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_conversation_id"), "messages", ["conversation_id"])
    # unpack the JSONB array back into rows
    op.execute(
        """
        INSERT INTO messages (id, conversation_id, role, content, created_at)
        SELECT (e->>'id')::uuid, c.id, e->>'role', e->>'content',
               (e->>'created_at')::timestamptz
        FROM conversations c, jsonb_array_elements(c.messages) AS e
        """
    )
    op.drop_column("conversations", "messages")
