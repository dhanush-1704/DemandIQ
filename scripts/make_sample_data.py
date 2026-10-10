"""Generate deterministic synthetic retail sales for smoke testing."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    rows = []
    for store in ("S1", "S2"):
        for product in ("P1", "P2", "P3"):
            for i, day in enumerate(pd.date_range("2025-01-01", periods=220)):
                promotion = int(i % 31 in (0, 1, 2))
                price = 12.0 + 2 * (product == "P3")
                demand = 18 + 4 * (day.dayofweek in (4, 5)) + 7 * promotion
                demand += 3 * (store == "S2") + 2 * (product == "P2")
                units = max(0, round(demand + rng.normal(0, 3)))
                rows.append((day.date(), store, product, units, price, promotion))
    output = Path("data/sample_sales.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "store_id", "product_id", "units_sold",
                                "price", "promotion"]).to_csv(output, index=False)
    print(f"Generated {len(rows)} synthetic rows at {output}")


if __name__ == "__main__":
    main()
