"""
GBDT Proto 2 - Refactored Modular Implementation

This module serves as the main interface for the refactored GBDT implementation.
The original monolithic code has been split into modular components for better
maintainability and testability.

Original file: gbdt_proto.py (2,367 lines)
Refactored into: gbdt_components/ package with 8 modules

Architecture:
- tree_node.py: DecisionTreeNode class
- data_transforms.py: Utility functions for data processing
- gradient_computer.py: GradientComputer class
- weighting_strategies.py: WeightingStrategyManager class
- tree_builder.py: TreeBuilder class
- multi_task_tree.py: MultiTaskDecisionTree class
- mtgbdt_core.py: MTGBDT main class
- mtrf.py: MTRF class
"""

# Import all classes from the modular components
from .gbdt_components import (
    MTGBDT,
    MTRF,
    MTGBMBase,
    DecisionTreeNode,
    MultiTaskDecisionTree,
    GradientComputer,
    WeightingStrategyManager,
    TreeBuilder,
    _add_cvr_labels,
    _compute_ips_weight,
    _validate_strategy_n_tasks,
    _normalize_array,
    _clip_probabilities
)

# Export the main classes for external use
__all__ = [
    'MTGBDT',
    'MTRF',
    'MTGBMBase',
    'DecisionTreeNode',
    'MultiTaskDecisionTree',
    'GradientComputer',
    'WeightingStrategyManager',
    'TreeBuilder',
    '_add_cvr_labels',
    '_compute_ips_weight',
    '_validate_strategy_n_tasks',
    '_normalize_array',
    '_clip_probabilities'
]

# For backward compatibility, maintain the same interface
# Users can import MTGBDT and MTRF directly from this module
print("GBDT Proto 2 - Modular Implementation Loaded")
print("Main classes: MTGBDT, MTRF")
print("Components: DecisionTreeNode, MultiTaskDecisionTree, GradientComputer, WeightingStrategyManager, TreeBuilder")
