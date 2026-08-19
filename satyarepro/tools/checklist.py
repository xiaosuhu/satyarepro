from __future__ import annotations

import inspect
import json
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

from .base import Tool


@dataclass
class CheckResult:
    check_id: str
    layer: int  # 1 or 2; a failed check (see ChecklistTool.execute) uses 0 as an "unknown" sentinel
    finding: str
    evidence: str
    confidence: str  # "high" | "medium" | "low"


# A single check: takes whatever kwargs the concrete tool's execute() was
# called with (field names are the subclass's business, not this base
# layer's) and returns a CheckResult, either directly or via await.
Check = Callable[..., "CheckResult | Awaitable[CheckResult]"]


class ChecklistTool(Tool):
    """Base class for tools built from a list of independent, parallel checks.

    Subclasses provide `checks` (and `schema`, still abstract per Tool).
    execute() runs every check — sync or async — collects a CheckResult from
    each, and never lets one check's exception abort the others.
    """

    @property
    @abstractmethod
    def checks(self) -> list[Check]: ...

    @property
    def caveats(self) -> str:
        return ""

    def _summarize(self, results: list[CheckResult]) -> str:
        """Default summary: counts checks whose finding isn't the literal "ok".

        Subclasses that use a different finding convention should override this.
        """
        flagged = sum(1 for r in results if r.finding != "ok")
        return f"{flagged}/{len(results)} checks flagged potential issues."

    async def _run_check(self, check: Check, kwargs: dict[str, Any]) -> CheckResult:
        check_id = getattr(check, "__name__", repr(check))
        try:
            outcome = check(**kwargs)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if not isinstance(outcome, CheckResult):
                raise TypeError(
                    f"check {check_id!r} returned {type(outcome).__name__}, expected CheckResult"
                )
            return outcome
        except Exception as exc:
            return CheckResult(
                check_id=check_id,
                layer=0,
                finding="check_failed",
                evidence=str(exc),
                confidence="low",
            )

    async def execute(self, **kwargs: Any) -> str:
        results = [await self._run_check(check, kwargs) for check in self.checks]
        return json.dumps(
            {
                "checks_run": [asdict(r) for r in results],
                "summary": self._summarize(results),
                "caveats": self.caveats,
            },
            indent=2,
        )
