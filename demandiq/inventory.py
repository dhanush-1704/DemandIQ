"""Simple review-period inventory recommendation heuristics."""
from math import ceil, sqrt
from pydantic import BaseModel, Field


class InventoryRequest(BaseModel):
    forecast_daily_units: float = Field(ge=0)
    on_hand: float = Field(ge=0)
    on_order: float = Field(default=0, ge=0)
    lead_time_days: float = Field(gt=0)
    review_period_days: float = Field(default=1, ge=0)
    daily_demand_std: float = Field(default=0, ge=0)
    service_z: float = Field(default=1.645, ge=0, le=5)


def recommend_inventory(request: InventoryRequest) -> dict:
    horizon = request.lead_time_days + request.review_period_days
    expected = request.forecast_daily_units * horizon
    safety = request.service_z * request.daily_demand_std * sqrt(horizon)
    target = expected + safety
    inventory_position = request.on_hand + request.on_order
    return {
        "expected_horizon_demand": round(expected, 2),
        "safety_stock": round(safety, 2),
        "order_up_to_level": ceil(target),
        "inventory_position": round(inventory_position, 2),
        "recommended_order_units": max(0, ceil(target - inventory_position)),
        "note": "Heuristic only; not a calibrated service-level guarantee.",
    }
