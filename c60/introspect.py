"""
DEPRECATED: All introspection, agent, story, and hybrid node classes are now in c60/introspect/ subpackage.
This file remains for backward compatibility only. Please import from:
- c60.introspect.introspector (PipelineIntrospector)
- c60.introspect.agents (RLSearchAgent, NASearchAgent)
- c60.introspect.story (PipelineStory)
- c60.introspect.hybrid (HybridNode)
"""
from .introspect.introspector import PipelineIntrospector
from .introspect.agents import RLSearchAgent, NASearchAgent
from .introspect.story import PipelineStory
from .introspect.hybrid import HybridNode

# For backward compatibility
__all__ = [
    'PipelineIntrospector',
    'RLSearchAgent',
    'NASearchAgent',
    'PipelineStory',
    'HybridNode',
]
