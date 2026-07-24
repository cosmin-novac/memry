"""Who a request acts as.

Every authenticated request resolves to exactly one Principal, and every
namespace decision is made from it rather than from anything the caller sent.
That distinction is the whole point: an MCP tool argument or a REST body field
is attacker-controlled, so a named principal must never be able to name a
namespace outside its own.

Administrator permission and memory ownership are independent. The operator
holding ``MEMRY_API_KEY`` is the only unconfined principal. Runtime accounts,
including the first account, are confined to one fixed memory space; configured
tenants may retain ``<name>::<user>`` sub-namespaces for programmatic clients.
"""

from __future__ import annotations

from pydantic import BaseModel


class Principal(BaseModel):
    """An authenticated identity plus the namespace it is confined to."""

    model_config = {"frozen": True}

    name: str | None = None
    default_user: str = "default"
    admin: bool = False
    fixed_user: str | None = None

    @property
    def is_admin(self) -> bool:
        """Whether this identity may perform administrative operations."""
        return self.name is None or self.admin

    @property
    def prefix(self) -> str | None:
        """Ownership selector handed to the store.

        ``None`` means unconfined operator access. A value ending in ``::`` is
        a tenant prefix; every other value is one exact account namespace.
        """
        if self.name is None:
            return None
        return self.fixed_user or f"{self.name}::"

    def namespace(self, user_id: str | None) -> str | None:
        """Map caller input into this principal's server-enforced space."""
        if self.name is None:
            return user_id
        if self.fixed_user is not None:
            return self.fixed_user
        return f"{self.name}::{user_id or self.default_user}"

    def owns(self, user_id: str | None) -> bool:
        selector = self.prefix
        if selector is None:
            return True
        if not user_id:
            return False
        return (
            user_id.startswith(selector)
            if selector.endswith("::")
            else user_id == selector
        )


ADMIN = Principal()
