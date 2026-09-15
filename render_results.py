"""Render the recorded per-seed metrics as an SVG using only the standard library."""

import argparse
from html import escape
import json
from pathlib import Path


def render(result: dict) -> str:
    labels = {
        "q_learning": "Q-learning",
        "round_robin": "Round robin",
        "random": "Random",
        "shortest_completion": "Shortest completion",
    }
    columns = (
        ("queue_work_sum", "Queue-work sum"),
        ("assignment_energy_proxy", "Assignment-energy proxy"),
        ("composite_cost", "Composite cost"),
    )
    policies = tuple(labels)
    rows = result["evaluation"]
    seeds = len({row["workload_seed"] for row in rows})
    jobs = result["configuration"]["n_jobs"]
    weight = result["configuration"]["environment"]["energy_weight"]
    summary = result["summary"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="390" viewBox="0 0 1100 390" role="img" aria-labelledby="title desc">',
        '<title id="title">GreenBalance: held-out synthetic scheduling costs</title>',
        f'<desc id="desc">Bars show means across {seeds} shared workloads of {jobs} jobs. Dots show individual workload results. All metrics are synthetic proxies; lower is better.</desc>',
        '<rect width="1100" height="390" fill="#fff"/>',
        '<g font-family="system-ui, sans-serif" fill="#182430">',
        '<text x="24" y="32" font-size="22" font-weight="600">GreenBalance: held-out synthetic scheduling</text>',
        f'<text x="24" y="58" font-size="14">{seeds} shared workloads × {jobs:,} jobs · bars = means · dots = individual workloads · lower is better</text>',
    ]
    for index, policy in enumerate(policies):
        parts.append(f'<text x="24" y="{130 + index * 48}" font-size="14">{escape(labels[policy])}</text>')
    for column, (metric, title) in enumerate(columns):
        x = 215 + column * 290
        width = 240
        maximum = max(row[metric] for row in rows) * 1.08 or 1.0
        parts.append(f'<text x="{x}" y="94" font-size="14" font-weight="600">{escape(title)}</text>')
        for index, policy in enumerate(policies):
            y = 117 + index * 48
            value = summary[policy][metric]["mean"]
            length = value / maximum * width
            color = "#176b87" if policy == "shortest_completion" else "#bac9d2"
            parts.append(f'<rect x="{x}" y="{y}" width="{length:.2f}" height="22" rx="2" fill="{color}"/>')
            for row in rows:
                if row["policy"] == policy:
                    point = x + row[metric] / maximum * width
                    parts.append(f'<circle cx="{point:.2f}" cy="{y + 11}" r="3" fill="#243e50" opacity="0.8"/>')
            parts.append(f'<text x="{x}" y="{y + 36}" font-size="11" fill="#425b6c">mean {value:,.1f}</text>')
        parts.extend([
            f'<line x1="{x}" y1="314" x2="{x + width}" y2="314" stroke="#8c9ba5"/>',
            f'<text x="{x}" y="330" font-size="11">0</text>',
            f'<text x="{x + width}" y="330" text-anchor="end" font-size="11">{maximum:,.0f}</text>',
        ])
    parts.append(f'<text x="24" y="365" font-size="12">Composite cost = queue-work sum + {weight:g} × assignment-energy proxy. No physical energy or latency units.</text>')
    parts.append('</g></svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = json.loads(args.input.read_text(encoding="utf-8"))
        svg = render(result)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(svg)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
