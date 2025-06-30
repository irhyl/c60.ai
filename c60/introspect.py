"""
DEPRECATED: All introspection, agent, and story classes are now in c60/introspect/ subpackage.
This file remains for backward compatibility only. Please import from:
- c60.introspect.introspector (PipelineIntrospector)
- c60.introspect.agents (RLSearchAgent, NASearchAgent)
- c60.introspect.story (PipelineStory)
"""
from .introspect.introspector import PipelineIntrospector
from .introspect.agents import RLSearchAgent, NASearchAgent
from .introspect.story import PipelineStory
    self.history: List[Dict[str, Any]] = []

    def log(self, event_type: str, pipeline_id: str, details: Dict[str, Any]):
        self.history.append({
            "event": event_type,
            "pipeline_id": pipeline_id,
            "details": details
        })

    def get_history(self, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if pipeline_id is None:
            return self.history
        return [h for h in self.history if h["pipeline_id"] == pipeline_id]

    def explain(self, pipeline_id: str) -> str:
        events = self.get_history(pipeline_id)
        lines = [f"Pipeline {pipeline_id} evolution story:"]
        for e in events:
            lines.append(f"- [{e['event']}] {json.dumps(e['details'])}")
        return "\n".join(lines)

class PipelineStory:
    """
    Generates a human-readable story and visualization for a pipeline's evolution.
    """
    def __init__(self, pipeline, introspector: PipelineIntrospector):
        self.pipeline = pipeline
        self.introspector = introspector

    def to_markdown(self, pipeline_id: str) -> str:
        story = self.introspector.explain(pipeline_id)
        md = f"## Pipeline Evolution Story\n\n{story}\n\n## Pipeline Structure\n\n{self.pipeline}\n"
        return md

    def to_html(self, pipeline_id: str) -> str:
        md = self.to_markdown(pipeline_id)
        return f"<pre>{md}</pre>"

    def visualize_dag(self, dag: nx.DiGraph, pipeline_id: str, show: bool = True):
        if not PLOTLY_AVAILABLE:
            print("Plotly not installed. DAG visualization unavailable.")
            return None
        pos = nx.spring_layout(dag)
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        for edge in dag.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            marker=dict(showscale=False, color=[], size=20, line_width=2))
        for node in dag.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (str(node),)
            node_trace['marker']['color'] += ('#1f77b4',)
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'Pipeline DAG: {pipeline_id}',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40)))
        if show:
            fig.show()
        return fig

# Hybrid symbolic + neural node support
class HybridNode:
    """
    Represents a node in the pipeline that can be symbolic (rule-based) or neural (learned).
    """
    def __init__(self, node_id: str, node_type: str, symbolic_rule: Optional[Any] = None, neural_model: Optional[Any] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.symbolic_rule = symbolic_rule
        self.neural_model = neural_model

    def is_symbolic(self):
        return self.symbolic_rule is not None

    def is_neural(self):
        return self.neural_model is not None

    def __repr__(self):
        if self.is_symbolic() and self.is_neural():
            return f"HybridNode({self.node_id}, symbolic+neural)"
        elif self.is_symbolic():
            return f"HybridNode({self.node_id}, symbolic)"
        elif self.is_neural():
            return f"HybridNode({self.node_id}, neural)"
        else:
            return f"HybridNode({self.node_id}, empty)"

# RL/NAS agent stub (to be integrated into optimizer/search)
class RLSearchAgent:
    """
    Simple Q-learning RL agent for pipeline search.
    Stores Q-values for (state, action) pairs and updates them based on observed rewards.
    Extensible for more advanced RL algorithms.
    """
    def __init__(self, action_space: List[Any], state_space: List[Any], reward_fn: Any, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.action_space = action_space
        self.state_space = state_space
        self.reward_fn = reward_fn
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # (state, action) -> value

    def select_action(self, state):
        import random
        if random.random() < self.epsilon:
            # Explore
            return random.choice(self.action_space)
        # Exploit
        q_vals = [self.q_table.get((state, a), 0.0) for a in self.action_space]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(self.action_space, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        q_sa = self.q_table.get((state, action), 0.0)
        next_qs = [self.q_table.get((next_state, a), 0.0) for a in self.action_space]
        max_next_q = max(next_qs) if next_qs else 0.0
        self.q_table[(state, action)] = q_sa + self.alpha * (reward + self.gamma * max_next_q - q_sa)

class NASearchAgent:
    """
    Simple NAS agent that randomly samples architectures from a given search space.
    Can be extended for RL-guided or evolutionary NAS.
    """
    def __init__(self, search_space: List[Any], eval_fn: Any):
        self.search_space = search_space
        self.eval_fn = eval_fn
        self.history = []

    def search(self, current_arch=None):
        import random
        arch = random.choice(self.search_space)
        score = self.eval_fn(arch)
        self.history.append((arch, score))
        return arch

class NASearchAgent:
    """
    Neural Architecture Search agent for neural pipeline components.
    (Stub: Plug into Optimizer for neural node search.)
    """
    def __init__(self, search_space: List[Any], eval_fn: Any):
        self.search_space = search_space
        self.eval_fn = eval_fn
        # TODO: Implement NAS logic

    def search(self, current_arch):
        # TODO: Implement NAS search
        return self.search_space[0]
