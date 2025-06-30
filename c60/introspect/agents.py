# RLSearchAgent and NASearchAgent classes for pipeline search

class RLSearchAgent:
    """
    Q-learning agent for RL-based pipeline search.
    Selects actions (mutations) based on state and reward.
    """
    def __init__(self):
        self.q_table = {}

    def select_action(self, state):
        # Stub: random or greedy action selection
        return "mutate"

    def update(self, state, action, reward, next_state):
        # Stub: Q-table update logic
        pass

class NASearchAgent:
    """
    Random/guided search agent for neural architecture search.
    """
    def search(self):
        # Stub: return random architecture
        return {"arch": "random"}
