"""Agent stages for ARES."""

from ares.agents.critic import Critic
from ares.agents.investigator import Investigator
from ares.agents.reviewer import Reviewer
from ares.agents.verifier import Verifier

__all__ = ["Critic", "Investigator", "Reviewer", "Verifier"]
