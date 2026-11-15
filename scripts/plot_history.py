"""Plot the force and energy histories written by an example."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path, help="Run directory containing history.csv")
args = parser.parse_args()
with (args.run / "history.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit("The history contains no accepted steps")

time = [1e6 * float(row["time_s"]) for row in rows]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
axes[0].plot(time, [float(row["reaction"]) for row in rows])
axes[0].set(xlabel="Time (µs)", ylabel="Reaction (N)")
for key in ("kinetic", "elastic", "fracture", "viscous", "work"):
    axes[1].plot(time, [float(row[key]) for row in rows], label=key)
axes[1].set(xlabel="Time (µs)", ylabel="Energy (see run.toml for units)")
axes[1].legend(frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(args.run / "history.png", dpi=200)
