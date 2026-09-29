"""Console and CSV logging of scalar metrics."""

import csv
import time

__all__ = ["Logger"]


class Logger:
    """Prints metric lines and appends the same rows to a CSV file.

    Args:
        path: CSV file to write; None logs to the console only.
        total: Final step value, used for the progress fraction and the ETA.
        unit: Name printed in front of the step value.
    """

    def __init__(
        self, path: str | None = None, total: int | None = None, unit: str = "step"
    ) -> None:
        self.total = total
        self.unit = unit
        self.start = time.time()
        self._file = open(path, "w", newline="") if path else None
        self._writer = None

    def log(self, step: int, **metrics: float) -> None:
        """Write one row and print it.

        Args:
            step: Step the metrics were measured at.
            metrics: Scalar values keyed by name.
        """
        if self._file is not None:
            if self._writer is None:
                self._writer = csv.DictWriter(self._file, ["step", *metrics])
                self._writer.writeheader()
            self._writer.writerow({"step": step, **metrics})
            self._file.flush()

        elapsed = time.time() - self.start
        progress = f"{step}/{self.total}" if self.total else f"{step}"
        values = "  ".join(f"{key} {value:.4f}" for key, value in metrics.items())
        timing = f"elapsed {elapsed:.0f}s"
        if self.total and step > 0:
            timing += f"  eta {elapsed * (self.total - step) / step:.0f}s"
        print(f"{self.unit} {progress}  {values}  {timing}", flush=True)

    def close(self) -> None:
        """Close the CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
