from pydantic import BaseModel, ConfigDict, Field


class BeamSearchConfig(BaseModel):
    """Configuration for beam search with context-aware evaluation"""

    model_config = ConfigDict(frozen=True)

    # beam search configs
    beam_width: int = Field(
        default=5, description="Number of nodes to keep at each level"
    )
    temperatures: list[float] = Field(default=[0.3, 0.6, 0.9, 1.2])
    seed: int = Field(
        default=42,
        description="Seed for deterministic sampling (beta feature of OpenAI)",
    )

    # sledgehammer and automation configs
    use_quick_tactics: bool = Field(
        default=True, description="Try common tactics first before LLM generation"
    )
    use_sledgehammer: bool = Field(
        default=True, description="Use sledgehammer for automation"
    )
    quick_tactics: list[str] = Field(
        default=[
            "apply simp",
            "apply auto",
            "apply blast",
            "apply arith",
            "apply linarith",
            "apply force",
            "apply fastforce",
        ],
        description="Quick tactics to try (evaluated by context score, not predefined confidence)",
    )
