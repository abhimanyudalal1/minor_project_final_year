"""Exercise the public CLI with a small, explicitly non-scientific fixture."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir",default="artifacts/fixture")
parser.add_argument("--device",default="cpu",choices=["cpu","mps","cuda"])
args = parser.parse_args()
repository = Path(__file__).resolve().parents[1]
for stage in ("doctor","matrix","verify-solver","generate-data","validate-data",
              "train","evaluate","summarize","build-report"):
    print(f"\nFixture stage: {stage}",flush=True)
    subprocess.run([sys.executable,"-m","burgers_research.cli",stage,"--config",
                    str(repository/"configs"/"fixture.yaml"),"--work-dir",args.work_dir,
                    "--device",args.device],check=True,cwd=repository)
