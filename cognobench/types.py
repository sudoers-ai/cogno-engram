from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    case_id: str
    name: str
    expected: str
    actual: str
    passed: bool


@dataclass
class DimensionResult:
    name: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def pct(self) -> float:
        return 100.0 * self.passed / self.total if self.total else 0.0

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]
