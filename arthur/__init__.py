"""
Arthur -- a hand-built coding agent you can run from any terminal.

Type `arthur` inside a repository and you get an interactive session: describe
a change in plain English, watch the agent read the repo, propose a patch, and
put that patch through a confidence gate before anything touches disk.

The whole point is that the mechanism is visible. There is no agent framework
underneath -- the loop, the protocol parser, the retrieval and the safety gate
are all here, in a few hundred readable lines, small enough to run on a 4GB
laptop GPU.
"""

__version__ = "0.1.0"
