"""
selection.py — Selection mechanisms for the C60.ai evolution engine.

TournamentSelector  — randomly samples k individuals and returns the best
ElitismPreserver    — always carries the top-e individuals unchanged

Tournament selection is the default because:
  - It works on arbitrary fitness scales (no normalization needed)
  - Selection pressure is tunable via tournament size k
  - It preserves diversity better than fitness-proportionate selection
    at the same pressure level
  - It has O(k) cost per selection, independent of population size

Elitism ensures the best solution found so far is never lost between
generations.  Without it, a single bad generation can discard work.
"""

from __future__ import annotations

import random
from typing import List

from c60.evolution.population import Individual, Population


# ---------------------------------------------------------------------------
# TournamentSelector
# ---------------------------------------------------------------------------

class TournamentSelector:
    """
    Select one parent by running a tournament among k random individuals.

    Parameters
    ----------
    tournament_size : int
        Number of contestants per tournament.  Larger values increase
        selection pressure (fitter individuals win more often).
        Typical range: 2 (low pressure) to 7 (high pressure).
    """

    def __init__(self, tournament_size: int = 5) -> None:
        if tournament_size < 2:
            raise ValueError(
                f"tournament_size must be >= 2, got {tournament_size}."
            )
        self.tournament_size = tournament_size

    def select(self, population: Population) -> Individual:
        """
        Draw tournament_size individuals at random (with replacement) and
        return the one with the highest adjusted score.
        """
        contestants = random.choices(
            list(population), k=self.tournament_size
        )
        return max(contestants, key=lambda ind: ind.score)

    def select_pair(self, population: Population) -> tuple[Individual, Individual]:
        """
        Select two distinct parents.  Runs two independent tournaments.
        The two results may be the same individual (allowed — their clones
        will be used as offspring starting points).
        """
        p1 = self.select(population)
        p2 = self.select(population)
        return p1, p2

    def __repr__(self) -> str:
        return f"TournamentSelector(k={self.tournament_size})"


# ---------------------------------------------------------------------------
# ElitismPreserver
# ---------------------------------------------------------------------------

class ElitismPreserver:
    """
    Carry the top-e individuals unchanged into the next generation.

    This guarantees monotone non-decreasing best fitness across generations:
    the best solution ever found is always in the population.

    Parameters
    ----------
    elite_count : int
        Number of elite individuals to preserve.  Typical value: 1–3.
        Setting to 0 disables elitism entirely.
    """

    def __init__(self, elite_count: int = 2) -> None:
        if elite_count < 0:
            raise ValueError(
                f"elite_count must be >= 0, got {elite_count}."
            )
        self.elite_count = elite_count

    def get_elites(self, population: Population) -> List[Individual]:
        """
        Return clones of the top-e individuals.

        Cloning ensures the elite copies in the next generation are
        independent objects — mutations on other individuals cannot
        accidentally corrupt them.
        """
        if self.elite_count == 0:
            return []
        top = population.top_k(self.elite_count)
        return [
            Individual(
                pipeline=ind.pipeline.clone(),
                fitness=ind.fitness,  # carry over the known fitness score
            )
            for ind in top
        ]

    def __repr__(self) -> str:
        return f"ElitismPreserver(elite_count={self.elite_count})"
