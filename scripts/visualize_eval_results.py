from verifiers.types import ChatMessage, Messages
from appworld.common.io import read_jsonl 
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from verifiers.types import Messages
from collections.abc import Mapping
from typing import cast

def message_to_printable(message: ChatMessage) -> ChatMessage:
    """
    Removes image_url objects from message content.
    """
    new_message = {}
    new_message["role"] = message["role"]
    new_message["content"] = []
    if "tool_calls" in message:
        new_message["tool_calls"] = message["tool_calls"]
    content = message.get("content")
    if content is None:
        return cast(ChatMessage, new_message)
    if isinstance(content, str):
        new_message["content"].append(content)
    else:
        for c in content:
            if isinstance(c, str):
                new_message["content"].append(c)
            else:
                c_dict = dict(c)
                if c_dict["type"] == "text":
                    new_message["content"].append(c_dict["text"])
                elif c_dict["type"] == "image_url":
                    new_message["content"].append("[image]")
                elif str(c_dict.get("type", "")).startswith("input_audio"):
                    new_message["content"].append("[audio]")
    new_message["content"] = "\n\n".join(new_message["content"])
    return cast(ChatMessage, new_message)

def messages_to_printable(messages) -> Messages:
    """
    Removes image_url objects from messages.
    """
    if isinstance(messages, str):
        return messages
    return [message_to_printable(m) for m in messages or []]

def print_prompt_completions_sample(
    prompts: list[Messages],
    completions: list[Messages],
    rewards: list[float],
    step: int,
    num_samples: int = 1,
    output_file: str = "output.txt",
) -> None:
    def _attr_or_key(obj, key: str, default=None):
        """Return obj.key if present, else obj[key] if Mapping, else default."""
        val = getattr(obj, key, None)
        if val is not None:
            return val
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return default

    def _normalize_tool_call(tc):
        """Return {"name": ..., "args": ...} from a dict or Pydantic-like object."""
        src = (
            _attr_or_key(tc, "function") or tc
        )  # prefer nested function object if present
        name = _attr_or_key(src, "name", "") or ""
        args = _attr_or_key(src, "arguments", {}) or {}

        if not isinstance(args, str):
            try:
                args = json.dumps(args)
            except Exception:
                args = str(args)
        return {"name": name, "args": args}

    def _format_messages(messages) -> Text:
        if isinstance(messages, str):
            return Text(messages)

        out = Text()
        for idx, msg in enumerate(messages):
            if idx:
                out.append("\n\n")

            assert isinstance(msg, dict)
            role = msg.get("role", "")
            content = msg.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"

            out.append(f"{role}: ", style="bold")
            out.append(content, style=style)

            for tc in msg.get("tool_calls") or []:  # treat None as empty list
                payload = _normalize_tool_call(eval(tc))
                out.append(
                    "\n\n[tool call]\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False),
                    style=style,
                )

        return out

    # Create console with file output
    console = Console(file=open(output_file, 'w'), width=120)
    table = Table(show_header=True, header_style="bold white", expand=True)

    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    reward_values = rewards
    if len(reward_values) < len(prompts):
        reward_values = reward_values + [0.0] * (len(prompts) - len(reward_values))

    samples_to_show = min(num_samples, len(prompts))
    for i in range(samples_to_show):
        prompt = list(prompts)[i]
        completion = list(completions)[i]
        reward = reward_values[i]

        formatted_prompt = _format_messages(prompt)
        formatted_completion = _format_messages(completion)

        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        if i < samples_to_show - 1:
            table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
    console.file.close()

file_path = "/weka/oe-adapt-default/shaktis/general-tool-use/verifiers/environments/appworld_env/outputs/evals/appworld-env--Qwen--Qwen3-8B/2d402cbc/results.jsonl"
example_id = 2
output_file = "prompt_completions_output.txt"

results = read_jsonl(file_path)
result = results[example_id]

printable_prompts = messages_to_printable(result.get("prompt"))
printable_completions = messages_to_printable(result.get("completion"))

print_prompt_completions_sample(
    [printable_prompts],
    [printable_completions],
    [result.get("reward")],
    step=0,
    num_samples=1,
    output_file=output_file,
)

print(f"Output written to {output_file}")