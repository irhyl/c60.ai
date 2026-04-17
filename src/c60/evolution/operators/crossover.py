"""
crossover.py — Subgraph exchange crossover for pipeline DAGs.

SubgraphExchangeCrossover is the primary recombination operator in C60.ai.

Algorithm
---------
Given two parent pipelines P1 and P2:

1. Find all compatible cut-point pairs: (v1 in P1, v2 in P2) where v1 and v2
   share the same (step_type, input_type, output_type) signature.
   - Matching step_type ensures semantic compatibility.
   - Matching input_type ensures the subgraph being grafted accepts the same
     kind of data its new parent will feed it.
   - Matching output_type ensures the subgraph's root still produces the
     type expected by any downstream nodes in the recipient pipeline.

2. For each candidate pair (in random order):
   a. Graft P2's subgraph rooted at v2 into P1, replacing P1's subgraph at v1.
   b. Graft P1's subgraph rooted at v1 into P2, replacing P2's subgraph at v2.
   c. Validate both offspring.
   d. Return (offspring1, offspring2) on first valid pair.

3. If no valid pair exists, return clones of both parents.

Grafting a subgraph
-------------------
"Graft donor's subgraph at d_root into recipient, replacing recipient's
subgraph at r_root" means:

  - Keep all recipient nodes NOT in the subgraph rooted at r_root.
  - Remove all recipient nodes IN the subgraph rooted at r_root.
  - Clone all donor nodes IN the subgraph rooted at d_root (new UUIDs).
  - Recreate all edges within the kept recipient portion.
  - Recreate all edges within the cloned donor portion.
  - Connect recipient's predecessors of r_root to the cloned d_root.

All cloned nodes get fresh UUIDs so offspring are fully independent objects.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import networkx as nx

from c60.core.pipeline import Pipeline, PipelineStep
from c60.execution.validator import PipelineValidator


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CrossoverOperator(ABC):
    """Abstract base for all crossover operators."""

    @abstractmethod
    def apply(
        self, parent1: Pipeline, parent2: Pipeline
    ) -> Tuple[Pipeline, Pipeline]:
        """
        Return two offspring pipelines.
        Falls back to clones of the parents if no valid crossover is found.
        """


# ---------------------------------------------------------------------------
# SubgraphExchangeCrossover
# ---------------------------------------------------------------------------

class SubgraphExchangeCrossover(CrossoverOperator):
    """
    Swap the subgraphs downstream of compatible cut-point nodes.

    Parameters
    ----------
    max_attempts : int
        Number of candidate cut-point pairs to try before giving up.
    """

    def __init__(self, max_attempts: int = 10) -> None:
        self.max_attempts = max_attempts
        self._validator = PipelineValidator()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(
        self, parent1: Pipeline, parent2: Pipeline
    ) -> Tuple[Pipeline, Pipeline]:
        cut_pairs = self.find_cut_points(parent1, parent2)
        if not cut_pairs:
            return parent1.clone(), parent2.clone()

        random.shuffle(cut_pairs)

        for v1_id, v2_id in cut_pairs[:self.max_attempts]:
            try:
                o1 = self._graft(
                    recipient=parent1, r_root_id=v1_id,
                    donor=parent2,    d_root_id=v2_id,
                )
                o2 = self._graft(
                    recipient=parent2, r_root_id=v2_id,
                    donor=parent1,    d_root_id=v1_id,
                )

                r1 = self._validator.validate(o1)
                r2 = self._validator.validate(o2)

                if r1.is_valid and r2.is_valid:
                    return o1, o2
            except Exception:
                continue

        return parent1.clone(), parent2.clone()

    def find_cut_points(
        self, p1: Pipeline, p2: Pipeline
    ) -> List[Tuple[str, str]]:
        """
        Return all (node_id_in_p1, node_id_in_p2) pairs where both nodes share
        the same (step_type, input_type, output_type) signature.

        These are guaranteed-compatible cut points: swapping them preserves
        type validity for all edges incident to the cut nodes.
        """
        # Build signature → [node_id] index for p1
        p1_by_sig: Dict[tuple, List[str]] = {}
        for step in p1.get_nodes():
            sig = (step.step_type, step.input_type, step.output_type)
            p1_by_sig.setdefault(sig, []).append(step.id)

        pairs: List[Tuple[str, str]] = []
        for step in p2.get_nodes():
            sig = (step.step_type, step.input_type, step.output_type)
            for p1_id in p1_by_sig.get(sig, []):
                pairs.append((p1_id, step.id))

        return pairs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _graft(
        self,
        recipient: Pipeline,
        r_root_id: str,
        donor: Pipeline,
        d_root_id: str,
    ) -> Pipeline:
        """
        Return a new pipeline = recipient with its subgraph at r_root_id
        replaced by a cloned copy of donor's subgraph at d_root_id.

        Steps
        -----
        1. Identify recipient subgraph nodes to REMOVE (r_root + descendants).
        2. Clone the nodes to KEEP from recipient (recipient ∖ remove set).
        3. Clone the nodes to GRAFT from donor (d_root + descendants).
        4. Build offspring graph:
           - Add all kept and grafted nodes.
           - Rewire kept-recipient edges (using new IDs).
           - Rewire donor-subgraph edges (using new IDs).
           - Connect recipient predecessors of r_root to cloned d_root.
        """
        # --- 1. Determine what to remove from recipient ---
        r_remove: set = (
            nx.descendants(recipient.graph, r_root_id) | {r_root_id}
        )
        r_keep: set = set(recipient.graph.nodes) - r_remove

        # --- 2. Clone kept recipient nodes (old_id → new_step) ---
        kept: Dict[str, PipelineStep] = {}
        for old_id in r_keep:
            new_step = recipient.get_step(old_id).clone()
            kept[old_id] = new_step

        # --- 3. Clone donor subgraph nodes (old_id → new_step) ---
        d_sub: set = (
            nx.descendants(donor.graph, d_root_id) | {d_root_id}
        )
        donated: Dict[str, PipelineStep] = {}
        for old_id in d_sub:
            new_step = donor.get_step(old_id).clone()
            donated[old_id] = new_step

        # --- 4. Build offspring ---
        offspring = Pipeline(name=recipient.name)

        # Add all nodes
        all_steps = list(kept.values()) + list(donated.values())
        for step in all_steps:
            offspring.graph.add_node(step.id, data=step)

        # Edges within kept recipient portion
        for f_old, t_old in recipient.get_edges():
            if f_old in r_keep and t_old in r_keep:
                offspring.graph.add_edge(kept[f_old].id, kept[t_old].id)

        # Edges within donor subgraph
        for f_old, t_old in donor.get_edges():
            if f_old in d_sub and t_old in d_sub:
                offspring.graph.add_edge(donated[f_old].id, donated[t_old].id)

        # Connect recipient predecessors of r_root to the cloned donor root
        for pred_old_id in recipient.graph.predecessors(r_root_id):
            if pred_old_id in r_keep:
                offspring.graph.add_edge(
                    kept[pred_old_id].id, donated[d_root_id].id
                )

        return offspring
