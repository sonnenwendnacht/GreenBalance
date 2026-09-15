# Validation and provenance

## Original artifacts

The notebook and report are preserved byte-for-byte from original commit
`0045e6b66a76c78cae34775b15f8240fb941b7a1`.

| Artifact | SHA-256 |
| --- | --- |
| `proj.ipynb` | `75e6264a000c7030bcfd8060f8f3a91012b9e5118ef046cc202b3b439c0fe20a` |
| `greenbalance.pdf` | `f2bc7aced1542cf6705ac1b16e75e0f390969b591485b640481c07526f1b9bcb` |

The report identifies Junzhe Zong as the sole original contributor. The current
maintenance work was prepared with Codex assistance. No third-party project
source or new license was introduced.

## Fixed recorded experiment

The maintained experiment was recorded after tests passed, using the
configuration below, selected before examining these held-out results. The
learning hyperparameters and model constants are inherited from the notebook.

```bash
python greenbalance.py --jobs 1000 --episodes 500 \
  --training-seed-start 0 --agent-seed 42 --baseline-seed 43 \
  --evaluation-seeds 1000000 1000001 1000002 1000003 1000004 \
  --output results/held_out.json
python render_results.py results/held_out.json results/held_out.svg
```

This is one trained agent evaluated on five shared, held-out workload traces.
The JSON records every seed, metric, configuration hash, source hash, and
environment version. Standard deviations across these traces are descriptive;
they are not confidence intervals or variability across independent trainings.

The recorded run used Python 3.12.3 and NumPy 2.4.2 on Linux x86-64.
Configuration SHA-256:
`969f0645ed74793f0dd8cea959f720414e7686c2042d79c4df548858b95ccc18`.
The trained table contained 541 states; final exploration probability was
0.08157186144027828. Evaluation uses greedy actions regardless of that value.

| Policy | Composite cost mean ± sample SD | Mean final backlog |
| --- | ---: | ---: |
| Q-learning | 284,086.4 ± 35,127.2 | 365.2 |
| Round robin | 396,105.0 ± 18,440.0 | 761.6 |
| Random | 418,502.6 ± 52,252.9 | 815.8 |
| Shortest completion | 227,607.4 ± 22,024.4 | 107.8 |

Q-learning's mean composite cost was 28.3% below round robin and 32.1% below
random, but 24.8% above shortest completion. The latter policy had the lowest
cost on all five evaluation traces. Q-learning's assignment-energy proxy was
higher than round robin's: its lower composite cost does not demonstrate an
energy saving. The compact state and inherited learning settings were not
adjusted after inspecting these evaluation results.

The default quick demo (200 jobs, 100 training episodes, three evaluation
traces) produces mean composite costs of 22,937.3 for Q-learning, 16,935.0 for
round robin, 18,013.0 for random, and 11,721.3 for shortest completion. The
learning agent loses to every baseline at that smaller budget. This demo is
provided for fast execution; it is not evidence of superior learning.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite checks exact queue transitions, service/work conservation, reward
accounting, terminal no-bootstrap and nonterminal updates, queue-bin edges,
repeatability, random-stream isolation, read-only evaluation, shared evaluation
arrivals, evaluation-order independence, source/configuration hashes, invalid
configurations, and the CLI's JSON output and refusal to overwrite files.

All 18 tests passed locally on Python 3.12.3 with NumPy 2.4.2. The default CLI
demo and the recorded 500,000-transition training run both completed. CI is
configured for Python 3.12; a successful hosted CI run is not claimed here.

A separate full rerun in the same Python 3.12 environment exactly reproduced
the serialized JSON record and SVG, including all recorded source hashes.

An independent scalar reference implementation additionally checked 500
randomized traces containing 37,963 transitions. Every state, reward, and proxy
metric matched, including reset and post-terminal behavior. This was a separate
review check, not an additional test in the repository's 18-test suite.

The result record hashes the maintained code, renderer, test file, dependency
file, workflow, and original artifacts. Documentation and result files are
excluded to avoid circular hashes. When changing any hashed source file, rerun
the fixed experiment and renderer before treating the record as current.

## Limits on interpretation

- Queue-work and assignment-energy values are synthetic proxies. No physical
  latency, power model, measured energy, arrival-time process, or deployment is
  included.
- The cost horizon stops after the last arrival; final backlog is reported
  separately rather than serviced to completion.
- Phase ordering is fixed even though job sizes vary. Evaluation only tests
  fresh draws from that distribution.
- Queue discretization merges all queues of 40 or more work units. The agent
  also cannot directly observe workload phase or time within the episode.
- Only one learning seed is trained in the recorded experiment. More training
  seeds and different arrival distributions would be needed for broad claims.
- The original notebook used one fixed training/evaluation trace. Its saved
  results and report remain historical, not freshly reproduced benchmarks.
