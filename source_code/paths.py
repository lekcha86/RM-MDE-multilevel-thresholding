"""Locations of the data and figure folders (override the root with the environment variable RMMDE_ROOT)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RMMDE_ROOT") or os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data_results")
FIG = os.path.join(ROOT, "figures")
