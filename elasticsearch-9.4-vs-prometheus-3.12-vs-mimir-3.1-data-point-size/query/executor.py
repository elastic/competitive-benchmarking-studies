import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

from .models import AttackReport, Defaults, QueryDefinition, VegetaConfig, VegetaTarget

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VegetaRunner:
    """Locates the vegeta binary and executes load test attacks."""

    def __init__(self) -> None:
        self._bin: str = self._resolve()

    @staticmethod
    def _resolve() -> str:
        found = shutil.which("vegeta")
        if found:
            return found
        local = os.path.join(_PROJECT_ROOT, ".bin", "vegeta")
        if os.path.isfile(local):
            return local
        sys.exit("vegeta not found — run 'make setup' to install it")

    def _attack_args(
        self, targets_path: str, results_path: str, cfg: VegetaConfig
    ) -> list[str]:
        args = [
            self._bin,
            "attack",
            "-targets",
            targets_path,
            "-format",
            "json",
            "-rate",
            cfg.effective_rate,
            "-duration",
            cfg.effective_duration,
            "-workers",
            str(cfg.effective_workers),
            "-timeout",
            cfg.effective_timeout,
            "-output",
            results_path,
            "-insecure",
        ]
        if cfg.effective_max_workers is not None:
            args += ["-max-workers", str(cfg.effective_max_workers)]
        return args

    def _run_attack(self, target: VegetaTarget, cfg: VegetaConfig) -> dict:
        targets_fd, targets_path = tempfile.mkstemp(suffix=".jsonl")
        results_fd, results_path = tempfile.mkstemp(suffix=".bin")
        try:
            os.write(targets_fd, (target.serialize() + "\n").encode())
            os.close(targets_fd)
            os.close(results_fd)
            subprocess.run(
                self._attack_args(targets_path, results_path, cfg), check=True
            )
            report = subprocess.run(
                [self._bin, "report", "--type", "json", results_path],
                capture_output=True,
                check=True,
            )
            return json.loads(report.stdout)
        finally:
            for path in (targets_path, results_path):
                try:
                    os.unlink(path)
                except OSError as e:
                    logging.warning("Failed to remove temp file %s: %s", path, e)

    def attack(self, target: VegetaTarget, cfg: VegetaConfig) -> AttackReport:
        raw = self._run_attack(target, cfg)
        lat = raw.get("latencies", {})
        return AttackReport(
            p50_ms=lat.get("50th", 0) / 1e6,
            p95_ms=lat.get("95th", 0) / 1e6,
            p99_ms=lat.get("99th", 0) / 1e6,
            mean_ms=lat.get("mean", 0) / 1e6,
            throughput=raw.get("throughput", 0.0),
            success_pct=raw.get("success", 0.0) * 100,
        )

    def warmup(
        self, target: VegetaTarget, query: QueryDefinition, defaults: Defaults
    ) -> None:
        """Run a warmup attack before the measured one; results are discarded."""
        warmup_rate = query.warmup_rate or query.rate or defaults.rate
        warmup_duration = (
            query.warmup_duration or defaults.warmup_duration or defaults.duration
        )
        warmup_cfg = VegetaConfig(
            effective_rate=warmup_rate,
            effective_duration=warmup_duration,
            effective_workers=query.workers or defaults.workers,
            effective_timeout=query.timeout or defaults.timeout,
            effective_max_workers=query.max_workers or defaults.max_workers,
        )
        self._run_attack(target, warmup_cfg)
