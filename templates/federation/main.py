#!/usr/bin/env python3
"""
Federation Pattern Example

Demonstrates cross-project pattern sharing and learning for
multi-project AI agent orchestration.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Pattern:
    """A learned pattern that can be shared across projects."""
    pattern_id: str
    pattern_type: str  # "success", "failure", "optimization"
    signature: str
    frequency: int = 1
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "signature": self.signature,
            "frequency": self.frequency,
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Pattern":
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            signature=data["signature"],
            frequency=data.get("frequency", 1),
            last_seen=datetime.fromisoformat(data.get("last_seen", datetime.now().isoformat())),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FederatedPatternStore:
    """
    Store and share patterns across projects.

    Enables cross-project learning by:
    - Recording patterns locally
    - Sharing patterns with other projects
    - Subscribing to pattern updates
    """
    project_id: str
    storage_path: Optional[Path] = None
    _patterns: Dict[str, Pattern] = field(default_factory=dict)
    _subscriptions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize storage."""
        if self.storage_path is None:
            self.storage_path = Path(f".patterns_{self.project_id}.json")
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for p_data in data.get("patterns", []):
                    p = Pattern.from_dict(p_data)
                    self._patterns[p.pattern_id] = p
                self._subscriptions = data.get("subscriptions", [])
            except Exception:
                pass

    def _save_patterns(self) -> None:
        """Save patterns to storage."""
        data = {
            "project_id": self.project_id,
            "patterns": [p.to_dict() for p in self._patterns.values()],
            "subscriptions": self._subscriptions,
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def record_pattern(self, pattern: Pattern) -> None:
        """Record or update a pattern."""
        if pattern.pattern_id in self._patterns:
            existing = self._patterns[pattern.pattern_id]
            existing.frequency += 1
            existing.last_seen = datetime.now()
        else:
            self._patterns[pattern.pattern_id] = pattern
        self._save_patterns()

    def get_patterns(
        self,
        pattern_type: Optional[str] = None,
    ) -> List[Pattern]:
        """Get all patterns, optionally filtered by type."""
        patterns = list(self._patterns.values())
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    def subscribe(self, project_id: str) -> None:
        """Subscribe to patterns from another project."""
        if project_id not in self._subscriptions:
            self._subscriptions.append(project_id)
            self._save_patterns()

    def export_patterns(self) -> List[Dict]:
        """Export patterns for sharing."""
        return [p.to_dict() for p in self._patterns.values()]

    def import_patterns(self, patterns: List[Dict], source_project: str) -> int:
        """Import patterns from another project."""
        imported = 0
        for p_data in patterns:
            p_data["metadata"]["source_project"] = source_project
            p = Pattern.from_dict(p_data)
            # Avoid duplicates
            if p.pattern_id not in self._patterns:
                self._patterns[p.pattern_id] = p
                imported += 1
        if imported > 0:
            self._save_patterns()
        return imported

    def get_stats(self) -> Dict:
        """Get federation stats."""
        return {
            "project_id": self.project_id,
            "total_patterns": len(self._patterns),
            "success_patterns": len([p for p in self._patterns.values() if p.pattern_type == "success"]),
            "failure_patterns": len([p for p in self._patterns.values() if p.pattern_type == "failure"]),
            "subscriptions": self._subscriptions,
        }


async def spawn_agent_with_learning(
    prompt: str,
    store: FederatedPatternStore,
) -> Dict:
    """
    Spawn an agent and record patterns from execution.

    Args:
        prompt: The task prompt
        store: Pattern store for recording

    Returns:
        Dict with response and pattern info
    """
    try:
        from google import genai

        client = genai.Client()

        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        # Record success pattern
        pattern = Pattern(
            pattern_id=f"success_{hash(prompt) % 10000}",
            pattern_type="success",
            signature=f"prompt_prefix:{prompt[:50]}",
            metadata={"prompt_length": len(prompt), "response_length": len(response.text)},
        )
        store.record_pattern(pattern)

        return {
            "success": True,
            "response": response.text,
            "pattern_recorded": pattern.pattern_id,
        }

    except Exception as e:
        # Record failure pattern
        pattern = Pattern(
            pattern_id=f"failure_{hash(str(e)) % 10000}",
            pattern_type="failure",
            signature=f"error_type:{type(e).__name__}",
            metadata={"error": str(e)},
        )
        store.record_pattern(pattern)

        return {
            "success": False,
            "error": str(e),
            "pattern_recorded": pattern.pattern_id,
        }


async def main():
    """Run federation pattern example."""
    print("Federation Pattern Example")
    print("=" * 50)

    # Initialize pattern stores for two "projects"
    store_a = FederatedPatternStore(project_id="project-a")
    store_b = FederatedPatternStore(project_id="project-b")

    print(f"Project A patterns: {store_a.get_stats()['total_patterns']}")
    print(f"Project B patterns: {store_b.get_stats()['total_patterns']}")
    print("=" * 50)

    # Run some tasks in project A
    print("\n[Project A] Running tasks...")
    tasks = [
        "What is Python?",
        "Name a database",
    ]

    for task in tasks:
        print(f"  Task: {task}")
        result = await spawn_agent_with_learning(task, store_a)
        if result["success"]:
            print(f"  Recorded: {result['pattern_recorded']}")

    # Share patterns from A to B
    print("\n[Federation] Sharing patterns A → B...")
    exported = store_a.export_patterns()
    imported = store_b.import_patterns(exported, source_project="project-a")
    print(f"  Imported {imported} patterns to Project B")

    # Project B subscribes to A
    store_b.subscribe("project-a")

    # Final stats
    print("\n" + "=" * 50)
    print("FEDERATION STATS")
    print("=" * 50)

    for store in [store_a, store_b]:
        stats = store.get_stats()
        print(f"\n{stats['project_id']}:")
        print(f"  Total patterns: {stats['total_patterns']}")
        print(f"  Success patterns: {stats['success_patterns']}")
        print(f"  Failure patterns: {stats['failure_patterns']}")
        print(f"  Subscriptions: {stats['subscriptions']}")


if __name__ == "__main__":
    asyncio.run(main())
