#!/usr/bin/env python3
"""Double-fork daemon launcher for SMPL binary fall training.

Keeps the job alive after Cursor/agent shells exit, writes PID/log, and
falls back to CPU if MPS keeps getting jetsam-killed.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "finetune" / "fall_nonfall_binary_smpl"
LOG = OUT / "train.log"
PIDFILE = OUT / "train.pid"
DATA = Path("/Volumes/data/fall down/radar_data/fall_nonfall_binary_balanced/dataset")
CONDA = Path.home() / "miniforge3" / "envs" / "mmap" / "bin" / "python"


def already_running() -> bool:
    if not PIDFILE.exists():
        return False
    try:
        pid = int(PIDFILE.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def daemonize() -> None:
    if os.fork() > 0:
        return  # parent returns to caller
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdin.close()
    # child continues


def build_cmd(device: str) -> list[str]:
    return [
        str(CONDA), "-u", str(ROOT / "run_finetuning_heatmap_wholemodel.py"),
        "--config", str(ROOT / "cfgs" / "finetune" / "fall_nonfall_binary.yaml"),
        "--device", device,
        "--data_path", str(DATA / "train"),
        "--eval_data_path", str(DATA / "val"),
        "--nb_classes", "2",
        "--batch_size", "1" if device == "mps" else "2",
        "--num_workers", "0",
        "--no_pin_mem",
        "--save_ckpt_freq", "1",
        "--epochs", "10",
        "--output_dir", str(OUT),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if already_running():
        print(f"Already running: {PIDFILE.read_text().strip()}")
        return
    if not CONDA.exists():
        raise SystemExit(f"Missing python: {CONDA}")
    if not (DATA / "train").is_dir():
        raise SystemExit(f"Missing dataset: {DATA}")

    device = os.environ.get("DEVICE", "mps")
    if device == "mps":
        # Quick availability check in parent before daemonizing.
        chk = subprocess.run(
            [str(CONDA), "-c", "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)"],
            check=False,
        )
        if chk.returncode != 0:
            device = "cpu"

    # Parent prints intent, then daemonizes the actual trainer.
    print(f"Daemonizing training on {device}")
    print(f"Log: {LOG}")
    if os.fork() > 0:
        # parent waits briefly for pid file
        for _ in range(50):
            time.sleep(0.1)
            if PIDFILE.exists():
                print(f"PID {PIDFILE.read_text().strip()}")
                return
        raise SystemExit("Daemon failed to write PID file")

    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    # Grandchild: real worker
    os.chdir(ROOT)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    PIDFILE.write_text(str(os.getpid()))
    with LOG.open("a", buffering=1) as log:
        log.write(f"\n===== daemon start pid={os.getpid()} device={device} =====\n")
        log.flush()
        # Prefer caffeinate when present.
        cmd = build_cmd(device)
        if Path("/usr/bin/caffeinate").exists():
            cmd = ["/usr/bin/caffeinate", "-dims", *cmd]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                start_new_session=True,
            )
            PIDFILE.write_text(str(proc.pid))
            code = proc.wait()
            log.write(f"\n===== daemon exit code={code} =====\n")
            # If MPS dies early with no useful exit, auto-fallback once to CPU.
            if code != 0 and device == "mps":
                log.write("MPS run failed; falling back to CPU\n")
                device = "cpu"
                cmd = build_cmd(device)
                if Path("/usr/bin/caffeinate").exists():
                    cmd = ["/usr/bin/caffeinate", "-dims", *cmd]
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True,
                )
                PIDFILE.write_text(str(proc.pid))
                code = proc.wait()
                log.write(f"\n===== cpu fallback exit code={code} =====\n")
        except Exception:
            import traceback
            traceback.print_exc(file=log)
            raise
        finally:
            if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink(missing_ok=True)
    os._exit(0)


if __name__ == "__main__":
    main()
