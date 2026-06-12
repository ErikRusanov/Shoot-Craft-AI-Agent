"""Pricing — what a paid call costs, so the budget can reserve before spending.

Pay-as-you-go means the dollar budget must cover *every* upstream call, and a
reservation has to be made **before** the call returns its real ``usage.cost``.
This module turns the provider's published rates into two numbers per call: a
realistic forecast (:meth:`predict_generation_cost`) for the plan, and a padded
reservation (:meth:`generation_reserve`) the meter holds until settle.

Rates are data, not code: :class:`PricingTable` carries the June-2026 OpenRouter
numbers and can be overridden at startup (``pricing_overrides_json``) so a price
change does not need a deploy. An unknown model is a startup error (deps
validates the configured models against the table), never a mid-session crash.

Token accounting for the image model (Nano Banana 2): the output **image** is a
fixed token block by resolution and dominates the bill (~98%); text output is
negligible; input is the prompt text plus one fixed block per input image. The
auxiliary LLM calls (slot fill, use-case classify) are tiny and unpredictable in
shape, so they carry a flat padded estimate instead of a token model.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

from pydantic import BaseModel, ConfigDict, Field

from schemas.enums import PaidCallKind
from utils.money import from_micro, to_micro

# Generation output is a fixed token block per resolution; 1K is the default.
_DEFAULT_OUTPUT_SIZE = "1K"
# References are sent without a per-part detail, so the provider applies its
# default — priced as the `auto` input-image block.
_DEFAULT_INPUT_DETAIL = "auto"
# A pure-image response carries a token or two of text; modelled as a small
# constant rather than zero so the forecast isn't biased low.
_TEXT_OUTPUT_TOKENS = 8


class ModelRate(BaseModel):
    """Per-model USD rate per million tokens, by token class."""

    model_config = ConfigDict(extra="forbid")

    input_per_mtok: Decimal
    text_output_per_mtok: Decimal
    image_output_per_mtok: Decimal = Decimal("0")


# Published OpenRouter rates (June 2026) for the auxiliary text/vision models a
# stage may be configured to. Aux calls settle on the provider-reported
# usage.cost, so these rates are startup validation and forecasting only.
KNOWN_AUX_RATES: dict[str, ModelRate] = {
    "anthropic/claude-haiku-4.5": ModelRate(
        input_per_mtok=Decimal("1"),
        text_output_per_mtok=Decimal("5"),
    ),
}


class PricingTable(BaseModel):
    """Provider rates and the token model that turns a call into dollars."""

    model_config = ConfigDict(extra="forbid")

    model_rates: dict[str, ModelRate]
    # Fixed output-image token blocks by resolution (June-2026 Gemini image).
    image_output_tokens: dict[str, int] = Field(
        default_factory=lambda: {"0.5K": 747, "1K": 1120, "2K": 1680, "4K": 2520}
    )
    # Fixed input-image token blocks by per-part detail level.
    image_input_tokens: dict[str, int] = Field(
        default_factory=lambda: {"high": 1120, "medium": 560, "low": 280, "auto": 1120}
    )
    chars_per_token: int = 4
    # Reservations are padded so the real cost almost never exceeds them.
    reserve_safety_factor: Decimal = Decimal("1.15")
    # Flat (already padded) reservation for the cheap auxiliary LLM calls.
    # INVENTORY carries one input image (~1120 tokens) plus a few hundred output
    # tokens on a cheap vision model, so its pad sits above the text-only calls.
    flat_estimates: dict[PaidCallKind, Decimal] = Field(
        default_factory=lambda: {
            PaidCallKind.SLOT_FILL: Decimal("0.002"),
            PaidCallKind.CLASSIFY: Decimal("0.002"),
            PaidCallKind.INVENTORY: Decimal("0.005"),
        }
    )

    @classmethod
    def default(cls, *, generation_model: str, lite_model: str) -> PricingTable:
        """The June-2026 OpenRouter table for the two models the core calls.

        ``generation_model`` is Nano Banana 2 (Gemini 3.1 flash image);
        ``lite_model`` is the cheap text model behind slot fill and classify.
        """
        return cls(
            model_rates={
                generation_model: ModelRate(
                    input_per_mtok=Decimal("0.50"),
                    text_output_per_mtok=Decimal("3"),
                    image_output_per_mtok=Decimal("60"),
                ),
                lite_model: ModelRate(
                    input_per_mtok=Decimal("0.25"),
                    text_output_per_mtok=Decimal("1.50"),
                ),
                **KNOWN_AUX_RATES,
            }
        )

    def rate_for(self, model: str) -> ModelRate:
        """The rate for ``model``, or ``ValueError`` — deps validates at startup."""
        rate = self.model_rates.get(model)
        if rate is None:
            raise ValueError(
                f"no pricing for model {model!r}; known models: {sorted(self.model_rates)}"
            )
        return rate

    def predict_generation_cost(
        self,
        model: str,
        *,
        prompt_chars: int,
        reference_count: int,
        output_size: str = _DEFAULT_OUTPUT_SIZE,
        face_detail: str | None = None,
    ) -> Decimal:
        """Realistic (unpadded) USD for one generation — the plan's price."""
        rate = self.rate_for(model)
        input_tokens = -(-prompt_chars // self.chars_per_token)  # ceil div
        input_tokens += reference_count * self.image_input_tokens[_DEFAULT_INPUT_DETAIL]
        if face_detail is not None:
            input_tokens += self.image_input_tokens[face_detail]
        image_tokens = self.image_output_tokens[output_size]
        cost_micro = (
            input_tokens * rate.input_per_mtok
            + _TEXT_OUTPUT_TOKENS * rate.text_output_per_mtok
            + image_tokens * rate.image_output_per_mtok
        )  # rates are per-million; divide once at the end
        return (cost_micro / Decimal(1_000_000)).quantize(Decimal("0.000001"))

    def generation_reserve(
        self,
        model: str,
        *,
        prompt_chars: int,
        reference_count: int,
        output_size: str = _DEFAULT_OUTPUT_SIZE,
        face_detail: str | None = None,
    ) -> Decimal:
        """Padded reservation for one generation: forecast x safety, rounded up.

        Same cost inputs as :meth:`predict_generation_cost`; the result is
        quantized up to micro-USD so the reserved amount lands exactly on the
        budget grid and never under-reserves by a sub-micro crumb.
        """
        predicted = self.predict_generation_cost(
            model,
            prompt_chars=prompt_chars,
            reference_count=reference_count,
            output_size=output_size,
            face_detail=face_detail,
        )
        return from_micro(to_micro(predicted * self.reserve_safety_factor, rounding=ROUND_CEILING))

    def flat_estimate(self, kind: PaidCallKind) -> Decimal:
        """The flat padded reservation for an auxiliary LLM call kind."""
        estimate = self.flat_estimates.get(kind)
        if estimate is None:
            raise ValueError(f"no flat estimate for paid-call kind {kind!r}")
        return estimate
