"""Последовательно пересчитывает этапы курсового проекта."""

import subprocess
import sys

from src.config import ROOT


def main():
    for module in ("src.data.split_data", "src.modeling", "src.service", "src.verify"):
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
