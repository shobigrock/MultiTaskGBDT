"""
GBDT Components Package

This package contains the modular components of the GBDT implementation,
refactored from the original gbdt_proto.py for better maintainability.
"""

from .tree_node import DecisionTreeNode
from .data_transforms import (
    _add_cvr_labels, 
    _compute_ips_weight, 
    _validate_strategy_n_tasks,
    _normalize_array,
    _clip_probabilities,
    _compute_dr_weights,
    _get_cvr_direct_estimates
)
from .gradient_computer import GradientComputer
from .weighting_strategies import WeightingStrategyManager
from .tree_builder import TreeBuilder
from .multi_task_tree import MultiTaskDecisionTree
from .mtgbdt_core import MTGBDT, MTGBMBase
from .mtrf import MTRF

__all__ = [
    'DecisionTreeNode',
    '_add_cvr_labels',
    '_compute_ips_weight',
    '_validate_strategy_n_tasks',
    '_normalize_array',
    '_clip_probabilities',
    '_compute_dr_weights',
    '_get_cvr_direct_estimates',
    'GradientComputer',
    'WeightingStrategyManager',
    'TreeBuilder',
    'MultiTaskDecisionTree',
    'MTGBDT',
    'MTGBMBase',
    'MTRF'
]
