"""Model definitions and baseline implementations.

Every baseline exposes the same contract as ``cost.CoST`` -- ``fit`` / ``predict``
(and ``encode`` where a representation exists) -- so ``tasks/`` can consume any of
them without knowing which one it holds.

Imports are lazy: ``baselines.cosinor`` needs CosinorPy, which is optional on the
compute nodes, and importing it eagerly here would take the whole package down
with it.
"""
