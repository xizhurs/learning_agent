"""MLflow models-from-code entry point (not a notebook): `mlflow.pyfunc.log_model(python_model=<this file>, model_config=...)`.

The agent is built from the logged model_config (keys: see rag.serving_agent.DEFAULTS); `rag` travels with the model via code_paths.
"""

import mlflow
from mlflow.models import ModelConfig

from rag.serving_agent import build_agent

mlflow.models.set_model(build_agent(ModelConfig(development_config={}).to_dict()))
