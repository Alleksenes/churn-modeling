# ============================================
# File: src/churn_model/__init__.py
# ============================================
# This file makes src/churn_model a Python package
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# from . import config
# from . import utils
# from . import processing
# from . import pipeline
# from . import tune
# from . import train
# from . import evaluate
# from . import predict

# NOTE: Explicitly importing submodules here may lead to 00 dependencies
# if they import from each other at the top . It's safer to let other
# modules import directly,`from src.churn_model.config import load_config`.
