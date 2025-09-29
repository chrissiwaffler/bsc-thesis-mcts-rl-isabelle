import os
from typing import Final, cast

from mirascope import LocalProvider, Provider

PROVIDER: Final[Provider | LocalProvider] = cast(
    Provider | LocalProvider, os.getenv("LLM_PROVIDER", "openai")
)
MODEL: Final[str] = os.getenv("LLM_MODEL", "gpt-4o-mini")
CRITIC_MODEL: Final[str] = os.getenv("LLM_CRITIC_MODEL", "gpt-4o")
