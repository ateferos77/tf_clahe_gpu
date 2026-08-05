"""Test package for gpu_clahe.

Made a package so tests can share the golden reference via a relative import
(``from .reference import clahe_reference``) without depending on sys.path
manipulation or pytest's rootdir heuristics.
"""
