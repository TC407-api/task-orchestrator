"""
Auto Research Scheduler Module.

Automated research system that:
- Tracks topics of interest per project
- Runs scheduled searches daily (Perplexity or Firecrawl)
- Stores findings in Graphiti knowledge graph
- Auto-injects fresh context into Claude sessions
"""

from .scheduler import TopicRegistry, ResearchScheduler
from .runner import ResearchRunner
from .perplexity_runner import PerplexityRunner
from .summarizer import ResearchSummarizer
from .primer import ContextPrimer

__all__ = [
    "TopicRegistry",
    "ResearchScheduler",
    "ResearchRunner",
    "PerplexityRunner",
    "ResearchSummarizer",
    "ContextPrimer",
]
