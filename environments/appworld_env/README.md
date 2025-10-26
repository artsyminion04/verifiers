# appworld-env

> Replace the placeholders below, then remove this callout.

### Overview
- **Environment ID**: `appworld-env`
- **Short description**: <one-sentence description>
- **Tags**: <comma-separated tags>

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: <single-turn | multi-turn | tool use>
- **Parser**: <e.g., ThinkParser, XMLParser, custom>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval appworld-env
```

Configure model and sampling:

```bash
uv run vf-eval appworld-env   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON 
uv run vf-eval appworld-env -m artsyminion04/Qwen-8B-Appworld-SFT -mt qwen3-appworld-sft -n 20 -r 3 -pt 6237 -ls True -a '{"eval_set": "test_normal"}' --save-dataset
uv run vf-eval appworld-env -m Qwen/Qwen3-8B -mt qwen3-8b -n 20 -r 3 -pt 6237 -ls True -a '{"eval_set": "test_normal"}' --save-dataset
uv run vf-eval -m Qwen/Qwen3-8B -mt qwen3-8b --port 6237 
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

