# Notebook entry point

The authoritative reproducible exploration is `src/pathsnap/explore.py`.
In a notebook using the existing `~/ml312` kernel, from the project root:

```python
import json
from pathlib import Path
import pandas as pd
from IPython.display import Image, IFrame, display
stats = pd.read_csv("experiments/trip_statistics.csv", dtype={"trip_id": str})
display(stats.groupby("split")[["gps_count", "node_count", "arc_count"]].sum())
display(Image("visualizations/dataset_overview.png"))
maps = json.loads(Path("experiments/representative_maps.json").read_text())
display(IFrame(f"visualizations/train_{maps[0]['trip_id']}.html", 1100, 750))
```

Run the pipeline before using these generated outputs.
