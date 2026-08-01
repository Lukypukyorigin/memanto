# BOUNTY SOLUTION FOR ISSUE #1727
# WALLET TO PAY: 0x7145832f95a247fB7833403f85F114610B58A404

```python
# memanto/cli/commands/recall.py
"""Multi-agent recall command implementation."""

import click
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from memanto.core.memory_store import MemoryStore
from memanto.core.models import Memory, RecallResult
from memanto.core.query import QueryBuilder, TemporalMode
from memanto.utils.formatting import format_memory_panel


console = Console()


@click.command()
@click.argument("query", type=str)
@click.option(
    "--agent",
    "-a",
    "agents",
    multiple=True,
    help="Agent ID to query (can be specified multiple times for multi-agent recall)",
)
@click.option(
    "--limit",
    "-l",
    default=10,
    type=int,
    help="Maximum number of memories to return",
)
@click.option(
    "--threshold",
    "-t",
    default=0.7,
    type=float,
    help="Minimum relevance score threshold (0.0-1.0)",
)
@click.option(
    "--temporal/--no-temporal",
    default=False,
    help="Enable temporal query mode (sort by created_at instead of score)",
)
@click.option(
    "--since",
    type=str,
    help="Filter memories created after this timestamp (ISO format)",
)
@click.option(
    "--until",
    type=str,
    help="Filter memories created before this timestamp (ISO format)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["panel", "table", "json"]),
    default="panel",
    help="Output format",
)
@click.pass_context
def recall(
    ctx: click.Context,
    query: str,
    agents: tuple[str, ...],
    limit: int,
    threshold: float,
    temporal: bool,
    since: Optional[str],
    until: Optional[str],
    output_format: str,
) -> None:
    """Recall memories across one or more agents.
    
    Examples:
    
        memanto recall "project plan" --agent agent-1 --agent agent-2
    
        memanto recall "meeting notes" --agent alice --agent bob --temporal --limit 20
    """
    store: MemoryStore = ctx.obj["store"]
    
    agent_list = list(agents) if agents else None
    
    query_builder = QueryBuilder()
    query_builder.with_text(query)
    if agent_list:
        query_builder.with_agents(agent_list)
    query_builder.with_limit(limit)
    query_builder.with_threshold(threshold)
    
    if temporal:
        query_builder.with_temporal_mode(TemporalMode.CHRONOLOGICAL)
    else:
        query_builder.with_temporal_mode(TemporalMode.RELEVANCE)
    
    if since:
        query_builder.with_since(since)
    if until:
        query_builder.with_until(until)
    
    built_query = query_builder.build()
    
    results: List[RecallResult] = store.recall(built_query)
    
    if not results:
        console.print("[yellow]No memories found matching your query.[/yellow]")
        return
    
    if output_format == "json":
        _output_json(results)
    elif output_format == "table":
        _output_table(results)
    else:
        _output_panels(results)


def _output_panels(results: List[RecallResult]) -> None:
    """Output memories as rich panels with agent badges."""
    for result in results:
        panel = format_memory_panel(result.memory, result.score, result.agent_id)
        console.print(panel)
        console.print()


def _output_table(results: List[RecallResult]) -> None:
    """Output memories as a table."""
    table = Table(title="Recall Results")
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Score", style="green", justify="right")
    table.add_column("Created", style="yellow")
    table.add_column("Content", style="white")
    
    for result in results:
        table.add_row(
            result.agent_id,
            f"{result.score:.3f}",
            result.memory.created_at.strftime("%Y-%m-%d %H:%M"),
            result.memory.content[:100] + ("..." if len(result.memory.content) > 100 else ""),
        )
    
    console.print(table)


def _output_json(results: List[RecallResult]) -> None:
    """Output memories as JSON."""
    import json
    from datetime import datetime
    
    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    data = [
        {
            "agent_id": r.agent_id,
            "score": r.score,
            "memory": {
                "id": r.memory.id,
                "content": r.memory.content,
                "metadata": r.memory.metadata,
                "created_at": r.memory.created_at.isoformat(),
                "updated_at": r.memory.updated_at.isoformat(),
            },
        }
        for r in results
    ]
    
    console.print(json.dumps(data, indent=2, default=default_serializer))
```

```python
# memanto/core/query.py
"""Query builder and temporal mode definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


class TemporalMode(Enum):
    """Query temporal sorting mode."""
    RELEVANCE = "relevance"      # Sort by relevance score (descending)
    CHRONOLOGICAL = "chronological"  # Sort by created_at (descending)


@dataclass
class Query:
    """Built query object for memory retrieval."""
    text: str
    agents: Optional[List[str]] = None
    limit: int = 10
    threshold: float = 0.7
    temporal_mode: str = "relevance"
    since: Optional[str] = None
    until: Optional[str] = None
    metadata_filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata_filters is None:
            self.metadata_filters = {}


class QueryBuilder:
    """Fluent query builder for memory recall."""
    
    def __init__(self):
        self._text: str = ""
        self._agents: Optional[List[str]] = None
        self._limit: int = 10
        self._threshold: float = 0.7
        self._temporal_mode: TemporalMode = TemporalMode.RELEVANCE
        self._since: Optional[str] = None
        self._until: Optional[str] = None
        self._metadata_filters: Dict[str, Any] = {}
    
    def with_text(self, text: str) -> "QueryBuilder":
        self._text = text
        return self
    
    def with_agents(self, agents: List[str]) -> "QueryBuilder":
        self._agents = agents
        return self
    
    def with_limit(self, limit: int) -> "QueryBuilder":
        self._limit = limit
        return self
    
    def with_threshold(self, threshold: float) -> "QueryBuilder":
        self._threshold = max(0.0, min(1.0, threshold))
        return self
    
    def with_temporal_mode(self, mode: TemporalMode) -> "QueryBuilder":
        self._temporal_mode = mode
        return self
    
    def with_since(self, since: str) -> "QueryBuilder":
        self._since = since
        return self
    
    def with_until(self, until: str) -> "QueryBuilder":
        self._until = until
        return self
    
    def with_metadata_filter(self, key: str, value: Any) -> "QueryBuilder":
        self._metadata_filters[key] = value
        return self
    
    def build(self) -> Query:
        return Query(
            text=self._text,
            agents=self._agents,
            limit=self._limit,
            threshold=self._threshold,
            temporal_mode=self._temporal_mode.value,
            since=self._since,
            until=self._until,
            metadata_filters=self._metadata_filters,
        )


# memanto/core/memory_store.py
"""Memory store interface and implementation."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from uuid import UUID, uuid4
from enum import Enum

from memanto.core.query import Query, TemporalMode
from memanto.core.models import Memory, RecallResult


class MemoryStore(ABC):
    """Abstract memory store interface."""
    
    @abstractmethod
    def store(self, memory: Memory) -> Memory:
        pass
    
    @abstractmethod
    def recall(self, query: Query) -> List[RecallResult]:
        pass
    
    @abstractmethod
    def get_agent_memories(self, agent_id: str, limit: int = 100) -> List[Memory]:
        pass
    
    @abstractmethod
    def get_all_agents(self) -> List[str]:
        pass


@dataclass
class InMemoryMemoryStore(MemoryStore):
    """In-memory implementation of memory store for testing/development."""
    
    _memories: Dict[str, List[Memory]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, '_memories') or self._memories is None:
            self._memories = {}
    
    def store(self, memory: Memory) -> Memory:
        if memory.agent_id not in self._memories:
            self._memories[memory.agent_id] = []
        self._memories[memory.agent_id].append(memory)
        return memory
    
    def recall(self, query: Query) -> List[RecallResult]:
        """Recall memories across multiple agents with combined sorting."""
        all_results: List[RecallResult] = []
        
        target_agents = query.agents if query.agents else list(self._memories.keys())
        
        for agent_id in target_agents:
            if agent_id not in self._memories:
                continue
            
            agent_memories = self._memories[agent_id]
            results = self._search_agent_memories(agent_id, agent_memories, query)
            all_results.extend(results)
        
        return self._sort_combined_results(all_results, query.temporal_mode)
    
    def _search_agent_memories(
        self, 
        agent_id: str, 
        memories: List[Memory], 
        query: Query
    ) -> List[RecallResult]:
        """Search memories for a single agent."""
        results: List[RecallResult] = []
        query_text = query.text.lower()
        
        for memory in memories:
            if query.since:
                since_dt = datetime.fromisoformat(query.since)
                if memory.created_at < since_dt:
                    continue
            
            if query.until:
                until_dt = datetime.fromisoformat(query.until)
                if memory.created_at > until_dt:
                    continue
            
            score = self._calculate_relevance(memory, query.text)
            if score >= query.threshold:
                results.append(RecallResult(
                    memory=memory,
                    score=score,
                    agent_id=agent_id,
                ))
        
        return results
    
    def _calculate_relevance(self, memory: Memory, query_text: str) -> float:
        """Calculate relevance score (simple TF-IDF-like scoring)."""
        query_terms = query_text.lower().split()
        content = memory.content.lower()
        
        if not query_terms:
            return 0.0
        
        matches = sum(1 for term in query_terms if term in content)
        base_score = matches / len(query_terms)
        
        metadata_boost = 0.0
        for key, value in memory.metadata.items():
            if isinstance(value, str) and query_text.lower() in value.lower():
                metadata_boost += 0.1
        
        return min(1.0, base_score + metadata_boost)
    
    def _sort_combined_results(
        self, 
        results: List[RecallResult], 
        temporal_mode: str
    ) -> List[RecallResult]:
        """Sort combined results from multiple agents."""
        if temporal_mode == TemporalMode.CHRONOLOGICAL.value:
            return sorted(results, key=lambda r: r.memory.created_at, reverse=True)
        else:
            return sorted(results, key=lambda r: r.score, reverse=True)
    
    def get_agent_memories(self, agent_id: str, limit: int = 100) -> List[Memory]:
        memories = self._memories.get(agent_id, [])
        return sorted(memories, key=lambda m: m.created_at, reverse=True)[:limit]
    
    def get_all_agents(self) -> List[str]:
        return list(self._memories.keys())


# memanto/core/models.py
"""Core data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4


@dataclass
class Memory:
    """Memory record."""
    content: str
    agent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if isinstance(self.id, str):
            pass
        else:
            self.id = str(self.id)


@dataclass
class RecallResult:
    """Result from a recall query."""
    memory: 'Memory'
    score: float
    agent_id: str


# memanto/utils/formatting.py
"""Rich formatting utilities."""

from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.console import Console

from memanto.core.models import Memory, RecallResult


console = Console()


def format_memory_panel(memory: 'Memory', score: float, agent_id: str) -> Panel:
    """Format a memory as a rich panel with agent badge."""
    from rich.text import Text
    from rich.panel import Panel
    
    # Create agent badge
    agent_badge = Text(f" 🤖 {The user wants only code, no markdown, no explanations. I need to output the complete Python code for the multi-agent recall feature. Let me write the complete implementation.```python
# memanto/cli/commands/recall.py
"""Multi-agent recall command implementation."""

import click
import json
from typing import List, Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from memanto.core.memory_store import MemoryStore
from memanto.core.models import Memory, RecallResult
from memanto.core.query import QueryBuilder, TemporalMode
from memanto.utils.formatting import format_memory_panel

console = Console()


@click.command()
@click.argument("query", type=str)
@click.option(
    "--agent",
    "-a",
    "agents",
    multiple=True,
    help="Agent ID to query (can be specified multiple times for multi-agent recall)",
)
@click.option(
    "--limit",
    "-l",
    default=10,
    type=int,
    help="Maximum number of memories to return",
)
@click.option(
    "--threshold",
    "-t",
    default=0.7,
    type=float,
    help="Minimum relevance score threshold (0.0-1.0)",
)
@click.option(
    "--temporal/--no-temporal",
    default=False,
    help="Enable temporal query mode (sort by created_at instead of score)",
)
@click.option(
    "--since",
    type=str,
    help="Filter memories created after this timestamp (ISO format)",
)
@click.option(
    "--until",
    type=str,
    help="Filter memories created before this timestamp (ISO format)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["panel", "table", "json"]),
    default="panel",
    help="Output format",
)
@click.pass_context
def recall(
    ctx: click.Context,
    query: str,
    agents: tuple[str, ...],
    limit: int,
    threshold: float,
    temporal: bool,
    since: Optional[str],
    until: Optional[str],
    output_format: str,
) -> None:
    """Recall memories across one or more agents.
    
    Examples:
    
        memanto recall "project plan" --agent agent-1 --agent agent-2
    
        memanto recall "meeting notes" --agent alice --agent bob --temporal --limit 20
    """
    from typing import Optional
    store: MemoryStore = ctx.obj["store"]
    
    agent_list = list(agents) if agents else None
    
    query_builder = QueryBuilder()
    query_builder.with_text(query)
    if agent_list:
        query_builder.with_agents(agent_list)
    query_builder.with_limit(limit)
    query_builder.with_threshold(threshold)
    
    if temporal:
        query_builder.with_temporal_mode(TemporalMode.CHRONOLOGICAL)
    else:
        query_builder.with_temporal_mode(TemporalMode.RELEVANCE)
    
    if since:
        query_builder.with_since(since)
    if until:
        query_builder.with_until(until)
    
    built_query = query_builder.build()
    
    results: List[RecallResult] = store.recall(built_query)
    
    if not results:
        console.print("[yellow]No memories found matching your query.[/yellow]")
        return
    
    if output_format == "json":
        _output_json(results)
    elif output_format == "table":
        _output_table(results)
    else:
        _output_panels(results)


def _output_panels(results: List[RecallResult]) -> None:
    """Output memories as rich panels with agent badges."""
    for result in results:
        panel = format_memory_panel(result.memory, result.score, result.agent_id)
        console.print(panel)
        console.print()


def _output_table(results: List[RecallResult]) -> None:
    """Output memories as a table."""
    table = Table(title="Recall Results")
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Score", style="green", justify="right")
    table.add_column("Created", style="yellow")
    table.add_column("Content", style="white")
    
    for result in results:
        table.add_row(
            result.agent_id,
            f"{result.score:.3f}",
            result.memory.created_at.strftime("%Y-%m-%d %H:%M"),
            result.memory.content[:100] + ("..." if len(result.memory.content) > 100 else ""),
        )
    
    console.print(table)


def _output_json(results: List[RecallResult]) -> None:
    """Output memories as JSON."""
    import json
    from datetime import datetime
    
    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    data = [
        {
            "agent_id": r.agent_id,
            "score": r.score,
            "memory": {
                "id": r.memory.id,
                "content": r.memory.content,
                "metadata": r.memory.metadata,
                "created_at": r.memory.created_at.isoformat(),
                "updated_at": r.memory.updated_at.isoformat(),
            },
        }
        for r in results
    ]
    
    console.print(json.dumps(data, indent=2, default=default_serializer))


# memanto/cli/__init__.py
"""CLI package initialization."""

from memanto.cli.commands.recall import recall
from memanto.cli.commands.store import store
from memanto.cli.commands.list_agents import list_agents

__all__ = ["recall", "store", "list_agents"]


# memanto/cli/main.py
"""Main CLI entry point."""

import click
from memanto.core.memory_store import InMemoryMemoryStore
from memanto.cli.commands.recall import recall
from memanto.cli.commands.store import store
from memanto.cli.commands.list_agents import list_agents


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Memanto - Multi-agent memory management CLI."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = InMemoryMemoryStore()


cli.add_command(recall)
cli.add_command(store)
cli.add_command(list_agents)


if __name__ == "__main__":
    cli()


# memanto/cli/commands/store.py
"""Store command implementation."""

import click
from typing import Optional
from rich.console import Console

from memanto.core.memory_store import MemoryStore
from memanto.core.models import Memory

console = Console()


@click.command()
@click.argument("content", type=str)
@click.option(
    "--agent",
    "-a",
    required=True,
    help="Agent ID to associate with this memory",
)
@click.option(
    "--metadata",
    "-m",
    multiple=True,
    help="Metadata as key=value pairs (can be specified multiple times)",
)
@click.pass_context
def store(
    ctx: click.Context,
    content: str,
    agent: str,
    metadata: tuple[str, ...],
) -> None:
    """Store a new memory for an agent."""
    store: MemoryStore = ctx.obj["store"]
    
    metadata_dict = {}
    for item in metadata:
        if "=" in item:
            key, value = item.split("=", 1)
            metadata_dict[key] = value
    
    memory = Memory(
        content=content,
        agent_id=agent,
        metadata=metadata_dict,
    )
    
    stored = store.store(memory)
    console.print(f"[green]Memory stored with ID: {stored.id}[/green]")
    console.print(f"Agent: {stored.agent_id}")
    console.print(f"Content: {stored.content}")


# memanto/cli/commands/list_agents.py
"""List agents command implementation."""

import click
from rich.console import Console
from rich.table import Table

from memanto.core.memory_store import MemoryStore

console = Console()


@click.command()
@click.pass_context
def list_agents(ctx: click.Context) -> None:
    """List all agents with stored memories."""
    store = ctx.obj["store"]
    agents = store.get_all_agents()
    
    if not agents:
        console.print("[yellow]No agents found.[/yellow]")
        return
    
    table = Table(title="Agents")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Memory Count", style="green", justify="right")
    
    for agent_id in agents:
        memories = store.get_agent_memories(agent_id)
        table.add_row(agent_id, str(len(memories)))
    
    console.print(table)


# memanto/core/query.py
"""Query builder and temporal mode definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class TemporalMode(Enum):
    """Query temporal sorting mode."""
    RELEVANCE = "relevance"
    CHRONOLOGICAL = "chronological"


@dataclass
class Query:
    """Built query object for memory retrieval."""
    text: str
    agents: Optional[List[str]] = None
    limit: int = 10
    threshold: float = 0.7
    temporal_mode: str = "relevance"
    since: Optional[str] = None
    until: Optional[str] = None
    metadata_filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata_filters is None:
            self.metadata_filters = {}


class QueryBuilder:
    """Fluent query builder for memory recall."""
    
    def __init__(self):
        self._text: str = ""
        self._agents: Optional[List[str]] = None
        self._limit: int = 10
        self._threshold: float = 0.7
        self._temporal_mode: "TemporalMode" = TemporalMode.RELEVANCE
        self._since: Optional[str] = None
        self._until: Optional[str] = None
        self._metadata_filters: Dict[str, Any] = {}
    
    def with_text(self, text: str) -> "QueryBuilder":
        self._text = text
        return self
    
    def with_agents(self, agents: List[str]) -> "QueryBuilder":
        self._agents = agents
        return self
    
    def with_limit(self, limit: int) -> "QueryBuilder":
        self._limit = limit
        return self
    
    def with_threshold(self, threshold: float) -> "QueryBuilder":
        self._threshold = max(0.0, min(1.0, threshold))
        return self
    
    def with_temporal_mode(self, mode: "TemporalMode") -> "QueryBuilder":
        self._temporal_mode = mode
        return self
    
    def with_since(self, since: str) -> "QueryBuilder":
        self._since = since
        return self
    
    def with_until(self, until: str) -> "QueryBuilder":
        self._until = until
        return self
    
    def with_metadata_filter(self, key: str, value: Any) -> "QueryBuilder":
        self._metadata_filters[key] = value
        return self
    
    def build(self) -> "Query":
        from memanto.core.query import Query as QueryClass, TemporalMode as TM
        return QueryClass(
            text=self._text,
            agents=self._agents,
            limit=self._limit,
            threshold=self._threshold,
            temporal_mode=self._temporal_mode.value,
            since=self._since,
            until=self._until,
            metadata_filters=self._metadata_filters,
        )


# memanto/core/memory_store.py
"""Memory store interface and implementation."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from memanto.core.query import Query, TemporalMode
from memanto.core.models import Memory, RecallResult


class MemoryStore(ABC):
    """Abstract memory store interface."""
    
    @abstractmethod
    def store(self, memory: Memory) -> Memory:
        pass
    
    @abstractmethod
    def recall(self, query: Query) -> List[RecallResult]:
        pass
    
    @abstractmethod
    def get_agent_memories(self, agent_id: str, limit: int = 100) -> List[Memory]:
        pass
    
    @abstractmethod
    def get_all_agents(self) -> List[str]:
        pass


@dataclass
class InMemoryMemoryStore(MemoryStore):
    """In-memory implementation of memory store for testing/development."""
    
    _memories: Dict[str, List[Memory]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, '_memories') or self._memories is None:
            self._memories = {}
    
    def store(self, memory: Memory) -> Memory:
        if memory.agent_id not in self._memories:
            self._memories[memory.agent_id] = []
        self._memories[memory.agent_id].append(memory)
        return memory
    
    def recall(self, query: Query) -> List[RecallResult]:
        """Recall memories across multiple agents with combined sorting."""
        from memanto.core.models import RecallResult
        all_results: List[RecallResult] = []
        
        target_agents = query.agents if query.agents else list(self._memories.keys())
        
        for agent_id in target_agents:
            if agent_id not in self._memories:
                continue
            
            agent_memories = self._memories[agent_id]
            results = self._search_agent_memories(agent_id, agent_memories, query)
            all_results.extend(results)
        
        return self._sort_combined_results(all_results, query.temporal_mode)
    
    def _search_agent_memories(
        self, 
        agent_id: str, 
        memories: List[Memory], 
        query: Query
    ) -> List[RecallResult]:
        """Search memories for a single agent."""
        from memanto.core.models import RecallResult
        results: List[RecallResult] = []
        query_text = query.text.lower()
        
        for memory in memories:
            if query.since:
                since_dt = datetime.fromisoformat(query.since)
                if memory.created_at < since_dt:
                    continue
            
            if query.until:
                until_dt = datetime.fromisoformat(query.until)
                if memory.created_at > until_dt:
                    continue
            
            score = self._calculate_relevance(memory, query.text)
            if score >= query.threshold:
                results.append(RecallResult(
                    memory=memory,
                    score=score,
                    agent_id=agent_id,
                ))
        
        return results
    
    def _calculate_relevance(self, memory: Memory, query_text: str) -> float:
        """Calculate relevance score (simple TF-IDF-like scoring)."""
        query_terms = query_text.lower().split()
        content = memory.content.lower()
        
        if not query_terms:
            return 0.0
        
        matches = sum(1 for term in query_terms if term in content)
        base_score = matches / len(query_terms)
        
        metadata_boost = 0.0
        for key, value in memory.metadata.items():
            if isinstance(value, str) and query_text.lower() in value.lower():
                metadata_boost += 0.1
        
        return min(1.0, base_score + metadata_boost)
    
    def _sort_combined_results(
        self, 
        results: List[RecallResult], 
        temporal_mode: str
    ) -> List[RecallResult]:
        """Sort combined results from multiple agents."""
        if temporal_mode == TemporalMode.CHRONOLOGICAL.value:
            return sorted(results, key=lambda r: r.memory.created_at, reverse=True)
        else:
            return sorted(results, key=lambda r: r.score, reverse=True)
    
    def get_agent_memories(self, agent_id: str, limit: int = 100) -> List[Memory]:
        memories = self._memories.get(agent_id, [])
        return sorted(memories, key=lambda m: m.created_at, reverse=True)[:limit]
    
    def get_all_agents(self) -> List[str]:
        return list(self._memories.keys())


# memanto/core/models.py
"""Core data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4


@dataclass
class Memory:
    """Memory record."""
    content: str
    agent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if isinstance(self.id, str):
            pass
        else:
            self.id = str(self.id)


@dataclass
class RecallResult:
    """Result from a recall query."""
    memory: Memory
    score: float
    agent_id: str


# memanto/utils/formatting.py
"""Rich formatting utilities."""

from rich.panel import Panel
from rich.text import Text
from rich.console import Console

from memanto.core.models import Memory, RecallResult

console = Console()


def format_memory_panel(memory: Memory, score: float, agent_id: str) -> Panel:
    """Format a memory as a rich panel with agent badge."""
    from rich.text import Text
    from rich.panel import Panel
    
    # Create agent badge
    agent_badge = Text(f" 🤖 {agent_id} ", style="bold white on blue")
    score_text = Text(f" Score: {score:.3f} ", style="bold white on green")
    
    header = Text.assemble(agent_badge, " ", score_text)
    
    # Format timestamp
    timestamp = memory.created_at.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_text = Text(f"🕐 {timestamp}", style="dim")
    
    # Build content
    content_lines = [
        header,
        Text(""),
        timestamp_text,
        Text(""),
        Text(memory.content),
    ]
    
    if memory.metadata:
        content_lines.append(Text(""))
        content_lines.append(Text("Metadata:", style="bold dim"))
        for key, value in memory.metadata.items():
            content_lines.append(Text(f"  {key}: {value}", style="dim"))
    
    content = Text("\n").join(content_lines)
    
    return Panel(
        content,
        title=f"Memory {memory.id[:8]}",
        border_style="blue",
        padding=(1, 2),
    )


# memanto/core/__init__.py
"""Core package initialization."""

from memanto.core.models import Memory, RecallResult
from memanto.core.query import Query, QueryBuilder, TemporalMode
from memanto.core.memory_store import MemoryStore, InMemoryMemoryStore

__all__ = [
    "Memory",
    "RecallResult",
    "Query",
    "QueryBuilder",
    "TemporalMode",
    "MemoryStore",
    "InMemoryMemoryStore",
]


# tests/test_multi_agent_recall.py
"""Comprehensive tests for multi-agent recall functionality."""

import pytest
from datetime import datetime, timedelta
from memanto.core.memory_store import InMemoryMemoryStore
from memanto.core.models import Memory
from memanto.core.query import QueryBuilder, TemporalMode, Query


class TestMultiAgentRecall:
    """Test multi-agent recall functionality."""
    
    @pytest.fixture
    def store(self) -> InMemoryMemoryStore:
        return InMemoryMemoryStore()
    
    @pytest.fixture
    def sample_memories(self, store: InMemoryMemoryStore) -> dict:
        """Create sample memories across multiple agents."""
        now = datetime.utcnow()
        
        memories = {
            "agent-1": [
                Memory(
                    content="Project plan for Q1: focus on user authentication",
                    agent_id="agent-1",
                    metadata={"project": "auth", "priority": "high"},
                    created_at=now - timedelta(days=5),
                ),
                Memory(
                    content="Meeting notes: discussed database schema changes",
                    agent_id="agent-1",
                    metadata={"meeting": "db-schema", "attendees": ["alice", "bob"]},
                    created_at=now - timedelta(days=3),
                ),
                Memory(
                    content="Bug fix: login timeout issue on mobile",
                    agent_id="agent-1",
                    metadata={"bug": "login-timeout", "platform": "mobile"},
                    created_at=now - timedelta(days=1),
                ),
            ],
            "agent-2": [
                Memory(
                    content="Project plan for Q2: API rate limiting implementation",
                    agent_id="agent-2",
                    metadata={"project": "rate-limiting", "priority": "medium"},
                    created_at=now - timedelta(days=4),
                ),
                Memory(
                    content="Design review: new dashboard UI components",
                    agent_id="agent-2",
                    metadata={"design": "dashboard", "status": "approved"},
                    created_at=now - timedelta(days=2),
                ),
                Memory(
                    content="Performance optimization: caching strategy for API",
                    agent_id="agent-2",
                    metadata={"optimization": "caching", "target": "api"},
                    created_at=now - timedelta(hours=12),
                ),
            ],
            "agent-3": [
                Memory(
                    content="Security audit: JWT token validation",
                    agent_id="agent-3",
                    metadata={"security": "jwt", "audit": "passed"},
                    created_at=now - timedelta(days=6),
                ),
                Memory(
                    content="Documentation: API endpoint documentation update",
                    agent_id="agent-3",
                    metadata={"docs": "api", "version": "v2"},
                    created_at=now - timedelta(days=1),
                ),
            ],
        }
        
        for agent_memories in memories.values():
            for memory in agent_memories:
                store.store(memory)
        
        return memories
    
    def test_single_agent_recall(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test recall with single agent (backward compatibility)."""
        query = QueryBuilder().with_text("project plan").with_agents(["agent-1"]).build()
        results = store.recall(query)
        
        assert len(results) == 1
        assert results[0].agent_id == "agent-1"
        assert "project plan" in results[0].memory.content.lower()
    
    def test_multi_agent_recall_basic(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test basic multi-agent recall with multiple agents."""
        query = QueryBuilder().with_text("project plan").with_agents(["agent-1", "agent-2"]).build()
        results = store.recall(query)
        
        assert len(results) == 2
        agent_ids = {r.agent_id for r in results}
        assert agent_ids == {"agent-1", "agent-2"}
    
    def test_multi_agent_recall_all_agents(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test recall across all agents when no agents specified."""
        query = QueryBuilder().with_text("project").build()
        results = store.recall(query)
        
        assert len(results) == 2
        agent_ids = {r.agent_id for r in results}
        assert agent_ids == {"agent-1", "agent-2"}
    
    def test_multi_agent_recall_sorted_by_relevance(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test results sorted by relevance score (default)."""
        query = QueryBuilder().with_text("project plan").with_agents(["agent-1", "agent-2"]).build()
        results = store.recall(query)
        
        assert len(results) == 2
        assert results[0].score >= results[1].score
    
    def test_multi_agent_recall_sorted_chronologically(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test results sorted chronologically in temporal mode."""
        query = (
            QueryBuilder()
            .with_text("project")
            .with_agents(["agent-1", "agent-2"])
            .with_temporal_mode(TemporalMode.CHRONOLOGICAL)
            .build()
        )
        results = store.recall(query)
        
        assert len(results) == 2
        assert results[0].memory.created_at >= results[1].memory.created_at
    
    def test_multi_agent_recall_with_limit(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test limit parameter across multiple agents."""
        query = QueryBuilder().with_text("project").with_agents(["agent-1", "agent-2"]).with_limit(1).build()
        results = store.recall(query)
        
        assert len(results) == 1
    
    def test_multi_agent_recall_with_threshold(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test threshold filtering across multiple agents."""
        query = QueryBuilder().with_text("project plan").with_agents(["agent-1", "agent-2"]).with_threshold(0.9).build()
        results = store.recall(query)
        
        assert len(results) == 0
    
    def test_multi_agent_recall_with_temporal_filter(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test temporal filters (since/until) across multiple agents."""
        now = datetime.utcnow()
        since = (now - timedelta(days=2)).isoformat()
        
        query = QueryBuilder().with_text("project").with_agents(["agent-1", "agent-2"]).with_since(since).build()
        results = store.recall(query)
        
        for result in results:
            assert result.memory.created_at >= datetime.fromisoformat(since)
    
    def test_multi_agent_recall_agent_badge_in_result(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test that each result contains correct agent_id."""
        query = QueryBuilder().with_text("project plan").with_agents(["agent-1", "agent-2"]).build()
        results = store.recall(query)
        
        for result in results:
            assert result.agent_id in ["agent-1", "agent-2"]
            assert hasattr(result, 'agent_id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'memory')
    
    def test_multi_agent_recall_empty_agent_list(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test recall with non-existent agent returns empty results."""
        query = QueryBuilder().with_text("project").with_agents(["non-existent"]).build()
        results = store.recall(query)
        
        assert len(results) == 0
    
    def test_multi_agent_recall_mixed_existing_nonexistent(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test recall with mix of existing and non-existing agents."""
        query = QueryBuilder().with_text("project").with_agents(["agent-1", "non-existent"]).build()
        results = store.recall(query)
        
        assert len(results) == 1
        assert results[0].agent_id == "agent-1"
    
    def test_multi_agent_recall_case_insensitive(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test recall is case insensitive."""
        query = QueryBuilder().with_text("PROJECT PLAN").with_agents(["agent-1", "agent-2"]).build()
        results = store.recall(query)
        
        assert len(results) == 2
    
    def test_multi_agent_recall_partial_match(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test recall with partial text matches."""
        query = QueryBuilder().with_text("plan").with_agents(["agent-1", "agent-2"]).build()
        results = store.recall(query)
        
        assert len(results) == 2
    
    def test_multi_agent_recall_metadata_search(self, store: InMemoryMemoryStore, sample_memories: dict):
        """Test recall searches metadata as well."""
        query = QueryBuilder().with_text("auth").with_agents(["agent-1", "agent-2", "agent-3"]).build()
        results = store.recall(query)
        
        assert len(results) >= 1
        assert any("auth" in r.memory.content.lower() or "auth" in str(r.memory.metadata).lower() for r in results)
    
    def test_multi_agent_recall_combined_sorting_relevance(self, store: InMemoryMemoryStore):
        """Test combined sorting by relevance across agents with different scores."""
        now = datetime.utcnow()
        store.store(Memory(content="highly relevant project plan", agent_id="agent-A", created_at=now - timedelta(days=10)))
        store.store(Memory(content="project plan", agent_id="agent-B", created_at=now))
        store.store(Memory(content="project", agent_id="agent-C", created_at=now - timedelta(days=5)))
        
        query = QueryBuilder().with_text("project plan").with_agents(["agent-A", "agent-B", "agent-C"]).build()
        results = store.recall(query)
        
        assert len(results) == 3
        assert results[0].score >= results[1].score >= results[2].score
    
    def test_multi_agent_recall_combined_sorting_chronological(self, store: InMemoryMemoryStore):
        """Test combined chronological sorting across agents."""
        now = datetime.utcnow()
        store.store(Memory(content="old memory", agent_id="agent-A", created_at=now - timedelta(days=10)))
        store.store(Memory(content="new memory", agent_id="agent-B", created_at=now))
        store.store(Memory(content="medium memory", agent_id="agent-C", created_at=now - timedelta(days=5)))
        
        query = (
            QueryBuilder()
            .with_text("memory")
            .with_agents(["agent-A", "agent-B", "agent-C"])
            .with_temporal_mode(TemporalMode.CHRONOLOGICAL)
            .build()
        )
        results = store.recall(query)
        
        assert len(results) == 3
        assert results[0].memory.created_at >= results[1].memory.created_at >= results[2].memory.created_at


class TestQueryBuilder:
    """Test QueryBuilder functionality."""
    
    def test_query_builder_with_agents(self):
        """Test QueryBuilder with multiple agents."""
        query = QueryBuilder().with_text("test").with_agents(["a1", "a2", "a3"]).build()
        
        assert query.agents == ["a1", "a2", "a3"]
        assert query.text == "test"
    
    def test_query_builder_temporal_mode(self):
        """Test QueryBuilder temporal mode setting."""
        query = QueryBuilder().with_text("test").with_temporal_mode(TemporalMode.CHRONOLOGICAL).build()
        
        assert query.temporal_mode == "chronological"
    
    def test_query_builder_default_values(self):
        """Test QueryBuilder default values."""
        query = QueryBuilder().with_text("test").build()
        
        assert query.limit == 10
        assert query.threshold == 0.7
        assert query.temporal_mode == "relevance"
        assert query.agents is None
    
    def test_query_builder_chaining(self):
        """Test QueryBuilder method chaining."""
        query = (
            QueryBuilder()
            .with_text("test")
            .with_agents(["a1"])
            .with_limit(5)
            .with_threshold(0.8)
            .with_temporal_mode(TemporalMode.CHRONOLOGICAL)
            .with_since("2024-01-01")
            .with_until("2024-12-31")
            .build()
        )
        
        assert query.text == "test"
        assert query.agents == ["a1"]
        assert query.limit == 5
        assert query.threshold == 0.8
        assert query.temporal_mode == "chronological"
        assert query.since == "2024-01-01"
        assert query.until == "2024-12-31"


class TestMemoryModels:
    """Test core memory models."""
    
    def test_memory_creation(self):
        """Test Memory model creation."""
        memory = Memory(content="test content", agent_id="agent-1")
        
        assert memory.content == "test content"
        assert memory.agent_id == "agent-1"
        assert memory.id is not None
        assert memory.created_at is not None
        assert memory.metadata == {}
    
    def test_memory_with_metadata(self):
        """Test Memory with metadata."""
        memory = Memory(
            content="test",
            agent_id="agent-1",
            metadata={"key": "value", "number": 42}
        )
        
        assert memory.metadata["key"] == "value"
        assert memory.metadata["number"] == 42
    
    def test_recall_result(self):
        """Test RecallResult model."""
        memory = Memory(content="test", agent_id="agent-1")
        result = RecallResult(memory=memory, score=0.95, agent_id="agent-1")
        
        assert result.memory == memory
        assert result.score == 0.95
        assert result.agent_id == "agent-1"


# tests/conftest.py
"""Pytest configuration and fixtures."""

import pytest
from memanto.core.memory_store import InMemoryMemoryStore


@pytest.fixture
def memory_store() -> InMemoryMemoryStore:
    """Provide a clean memory store for each test."""
    return InMemoryMemoryStore()


# tests/__init__.py
"""Tests package initialization."""


# pyproject.toml
"""Project configuration."""

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "memanto"
version = "0.1.0"
description = "Multi-agent memory management system"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
memanto = "memanto.cli.main:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```