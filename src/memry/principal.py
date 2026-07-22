"""Who a request acts as.

Every authenticated request resolves to exactly one Principal, and every
namespace decision is made from it rather than from anything the caller sent.
That distinction is the whole point: an MCP tool argument or a REST body field
is attacker-controlled, so a named principal must never be able to name a
namespace outside its own.

The admin principal (``name`` None) is unconfined - it is the single-user
default the server has always had, and the operator holding ``MEMRY_API_KEY``.
A named principal (a config tenant today, an account once multiuser lands) is
transparently confined to ``<name>::<user>``.
"""

from __future__ import annotations

from pydantic import BaseModel


class Principal(BaseModel):
    """An authenticated identity plus the namespace it is confined to."""

    model_config = {"frozen": True}

    name: str | None = None  # None = admin / open mode: no confinement
    default_user: str = "default"

    @property
    def is_admin(self) -> bool:
        return self.name is None

    @property
    def prefix(self) -> str | None:
        """Namespace prefix every owned record must start with, or None for
        admin. This is what gets handed to the store's ownership gate."""
        return None if self.name is None else f"{self.name}::"

    def namespace(self, user_id: str | None) -> str | None:
        """Map a caller-supplied user id into this principal's space.

        For admin that is the id itself (including None, meaning "all
        namespaces"). For anyone else the id is only ever a *sub*-namespace:
        passing "victim" yields "alice::victim", never "victim".
        """
        if self.name is None:
            return user_id
        return f"{self.name}::{user_id or self.default_user}"

    def owns(self, user_id: str | None) -> bool:
        prefix = self.prefix
        if prefix is None:
            return True
        return bool(user_id) and user_id.startswith(prefix)


ADMIN = Principal()
