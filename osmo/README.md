# OSMO pipeline: coming soon

This folder will hold the automated version of the sim-to-sim check, run with
[NVIDIA OSMO](https://github.com/NVIDIA/OSMO): one command submits a workflow that runs the
policy in MuJoCo, builds the Isaac Sim robot from it, runs it under several conditions
(nominal, lower floor friction, extra payload, weaker motors) and gates the result as
PASS or FAIL.

Until then, the same comparison for the nominal case runs locally:

```bash
./scripts/sim2sim_check.sh
```
