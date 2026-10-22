"""Check update success and rollback without a Docker daemon or real sleeps."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("healthy,run_fails,had_previous", [
    (True, False, True), (False, False, True),
    (False, True, True), (False, True, False),
])
def test_deployment_rollback(tmp_path, healthy, run_fails, had_previous):
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n" + '''
import os, sys
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
args = sys.argv[1:]
if args[:2] == ["container", "inspect"]:
    sys.exit(0 if args[-1] == "image-classifier" and os.environ["PREVIOUS"] == "1" else 1)
if args[0] == "inspect":
    print("healthy" if os.environ["HEALTHY"] == "1" else "unhealthy")
if args[0] == "run" and os.environ["RUN_FAILS"] == "1":
    sys.exit(1)
''')
    docker.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    log = tmp_path / "commands.log"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "IMAGE": "example:new", "COMMAND_LOG": str(log),
           "HEALTHY": str(int(healthy)), "RUN_FAILS": str(int(run_fails)),
           "PREVIOUS": str(int(had_previous))}
    env.pop("GHCR_TOKEN", None)
    result = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "scripts/deploy.sh")],
                            env=env, capture_output=True, text=True)
    commands = log.read_text()
    assert (result.returncode == 0) is healthy
    if healthy:
        assert "rm image-classifier-previous\n" in commands
        assert "start image-classifier\n" not in commands
    elif had_previous:
        assert "rename image-classifier-previous image-classifier\n" in commands
        assert "start image-classifier\n" in commands
    else:
        assert "start image-classifier\n" not in commands
