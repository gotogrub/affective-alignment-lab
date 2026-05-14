from __future__ import annotations

from typing import Any, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 fallback
    ConfigDict = None  # type: ignore[assignment]


Intent = Literal[
    "product_search",
    "product_recommendation",
    "product_comparison",
    "order_status",
    "delivery_question",
    "return_policy",
    "technical_support",
    "complaint",
    "human_handoff",
    "unknown",
]

AnswerType = Literal[
    "recommend_products",
    "search_products",
    "compare_products",
    "answer_policy",
    "order_lookup",
    "support_ticket",
    "ask_clarification",
    "human_handoff",
    "no_match",
    "security_warning",
    "unknown",
]

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


class StrictModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - pydantic v1 fallback

        class Config:
            extra = "forbid"


class RejectedProduct(StrictModel):
    id: str
    reason: str


class StructuraOutput(StrictModel):
    intent: Intent
    category: Optional[str] = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    selected_products: list[str] = Field(default_factory=list)
    rejected_products: list[RejectedProduct] = Field(default_factory=list)
    answer_type: Optional[AnswerType] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    needs_human: bool = False
    handoff_reason: Optional[str] = None
    security_flags: list[str] = Field(default_factory=list)


class StructuraInput(StrictModel):
    user_query: str
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)


class StructuraSample(StrictModel):
    id: str
    input: StructuraInput
    target: StructuraOutput
    scenario: Optional[str] = None
    split: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)


def validate_model(model_cls: type[MODEL_T], value: Any) -> MODEL_T:
    """Validate data with pydantic v2 or v1 without leaking version details."""

    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(value)  # type: ignore[attr-defined]
    return model_cls.parse_obj(value)  # type: ignore[attr-defined]


def dump_model(model: BaseModel) -> dict[str, Any]:
    """Return a JSON-serializable dict from pydantic v2 or v1 models."""

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[attr-defined]
    return model.dict()  # pragma: no cover - pydantic v1 fallback
