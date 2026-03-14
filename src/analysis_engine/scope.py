from dataclasses import dataclass, field

@dataclass
class AnalysisScope:
    states: list[str] = field(default_factory=list)
    orgs: list[str] = field(default_factory=list)
    lobs: list[str] = field(default_factory=list)
    months: list[str] = field(default_factory=list)
    limit: int = 20
