"""Downstream evaluation protocols run on top of a fitted model.

Mirrors the upstream CoST layout: ``cost.py`` trains, ``tasks/`` evaluates. Each
module takes an already-fitted model plus data and returns metrics / artefacts;
none of them train.
"""
