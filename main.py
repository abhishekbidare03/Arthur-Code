"""
Backwards-compatible shim.

The real entry point is now `arthur` (see arthur/cli.py), installed on PATH by
`pip install -e .`. This file only exists so the older, documented invocation

    python main.py --repo demo_repo --task "..."

keeps working from a checkout without installing anything. New flags live on
the `arthur` command; this maps the two old ones onto it.
"""

import sys

from arthur.cli import main as arthur_main

if __name__ == "__main__":
    argv = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        # --task X  ->  -p X       and  --repo X  ->  -C X
        if args[i] == "--task" and i + 1 < len(args):
            argv += ["-p", args[i + 1]]
            i += 2
        elif args[i] == "--repo" and i + 1 < len(args):
            argv += ["-C", args[i + 1]]
            i += 2
        else:
            argv.append(args[i])
            i += 1
    sys.exit(arthur_main(argv))
