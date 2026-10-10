"""HTTP API for inventory decision support."""
from fastapi import FastAPI
from .inventory import InventoryRequest, recommend_inventory

app = FastAPI(title="DemandIQ API", version="0.1.0",
              description="Demand forecasting and inventory optimization foundation")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inventory/recommend")
def inventory_recommend(payload: InventoryRequest):
    return recommend_inventory(payload)
