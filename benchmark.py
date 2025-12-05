"""
Digital Panopticon experiment harness.

This script implements the "Who is the Spy?" multi-agent game with quadratic
monitoring. It supports two backends:
- MLX + mlx_lm for real runs on Apple Silicon (shared weights, optional cache).
- A lightweight mock mode for quick local testing without model downloads.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mlx.core as mx
except Exception:  # mlx may not be installed in all environments
    mx = None

try:
    from mlx_lm import load, generate
except Exception:  # mlx_lm may not be installed in all environments
    load = None
    generate = None


DEFAULT_SYSTEM_PROMPT = (
    "You are playing the social deduction game 'Who is the Spy?'. "
    "Each player has a secret word. Civilians share the same word. "
    "Spies have a slightly different word and must blend in. "
    "Describe your word without revealing it directly. Be brief (one or two sentences)."
)

MONITOR_PROMPT = (
    "You are a monitoring agent. Given another player's statement, estimate the "
    "probability they are the Spy. Respond EXACTLY in two lines:\n"
    "Score: <0-10 integer or float>\n"
    "Rationale: <one sentence, concise>\n"
    "No extra text, no bullet points."
)


def safe_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if mx is not None:
        try:
            mx.random.seed(seed)
        except Exception:
            pass


def ensure_dir(path: Optional[Path]) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def clone_cache(cache_obj):
    try:
        return copy.deepcopy(cache_obj)
    except Exception:
        return None


def extract_score(text: str) -> float:
    """Pull the first number between 0 and 10 out of the model's response."""
    import re

    numbers = re.findall(r"([0-9]+(?:\\.[0-9]+)?)", text)
    for num in numbers:
        try:
            value = float(num)
            if 0 <= value <= 10:
                return value
        except ValueError:
            continue
    return 5.0  # neutral fallback


def clean_statement(text: str) -> str:
    """Remove chat header tokens and trim whitespace."""
    import re

    cleaned = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>\n\n", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned


@dataclass
class GameConfig:
    model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    num_agents: int = 8
    num_spies: int = 1
    rounds: int = 3
    max_tokens: int = 120
    temperature: float = 0.0
    spy_word: str = "Artificial Intelligence"
    civilian_word: str = "Machine Learning"
    mock: bool = False
    seed: int = 42
    save_dir: Optional[Path] = None
    tag: Optional[str] = None


class LLMBackend:
    """Thin wrapper around mlx_lm with optional mock mode and prefix caching."""

    def __init__(self, config: GameConfig, system_prompt: str) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self.model = None
        self.tokenizer = None
        self.prefix_cache = None

        if not config.mock:
            if load is None or generate is None:
                raise ImportError(
                    "mlx_lm is required for non-mock runs. Install via `pip install mlx-lm`."
                )
            self.model, self.tokenizer = load(config.model_path)
            self.prefix_cache = self._precompute_prefix_cache(system_prompt)

    def _precompute_prefix_cache(self, system_prompt: str):
        """Compute KV cache for the shared system prompt when supported."""
        if self.model is None or self.tokenizer is None:
            return None

        try:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}],
                tokenize=False,
            )
        except Exception:
            return None

        # Some mlx_lm builds accept return_cache; if not, we fall back gracefully.
        try:
            result = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=0,
                cache=None,
                return_cache=True,
            )
            if isinstance(result, tuple) and len(result) == 2:
                return result[1]
            # If the backend returns only text, there is no cache to reuse.
            return None
        except TypeError:
            try:
                generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=0,
                )
            except Exception:
                pass
            return None
        except Exception:
            return None

    def generate_text(
        self,
        messages: Sequence[Dict[str, str]],
        cache=None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Optional[object]]:
        if self.config.mock:
            return self._mock_generate(messages), cache

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded.")

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        kwargs = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            # mlx_lm uses `temp` (not `temperature`) in some versions.
            "temp": self.config.temperature if temperature is None else temperature,
        }
        if cache is not None:
            kwargs["cache"] = cache

        try:
            return generate(self.model, self.tokenizer, **kwargs), kwargs.get("cache")
        except TypeError:
            # Drop cache if unsupported, then drop temp if still unsupported.
            kwargs.pop("cache", None)
            try:
                return generate(self.model, self.tokenizer, **kwargs), None
            except TypeError:
                kwargs.pop("temp", None)
                return generate(self.model, self.tokenizer, **kwargs), None

    def _mock_generate(self, messages: Sequence[Dict[str, str]]) -> str:
        """Small, deterministic mock to exercise the pipeline without a model."""
        last = messages[-1]["content"] if messages else ""
        first = messages[0]["content"] if messages else ""
        if "Suspicion" in first or "monitoring" in first.lower():
            score = round(random.uniform(3.5, 8.0), 2)
            return f"Suspicion score: {score}. Rationale: {last[:80]}..."
        if "Spy" in last:
            return "I think the theme involves covert operations and intelligence gathering."
        if "Machine" in last or "machine" in last:
            return "It relates to data-driven algorithms and computational learning."
        return "Offering a neutral description consistent with the shared theme."


class Agent:
    def __init__(
        self,
        agent_id: int,
        role: str,
        secret_word: str,
        backend: LLMBackend,
    ) -> None:
        self.id = agent_id
        self.role = role
        self.secret_word = secret_word
        self.backend = backend
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": backend.system_prompt},
            {
                "role": "user",
                "content": (
                    f"You are Player {agent_id}. Your role is {role}. "
                    f"Your secret word is '{secret_word}'. Describe your word without revealing it. "
                    "Include two concrete attributes that hint at the word, avoid repeating others, and keep it to two sentences."
                ),
            },
        ]
        self.cache = clone_cache(backend.prefix_cache)

    def act(self, public_history: Sequence[Tuple[int, str]], temperature: Optional[float] = None) -> str:
        history_text = format_history(public_history)
        prompt_msg = {
            "role": "user",
            "content": (
                f"Round context: {history_text if history_text else 'No prior statements this round.'} "
                "Provide your description now, with two concrete cues about your secret word."
            ),
        }
        messages = self.history + [prompt_msg]
        response, updated_cache = self.backend.generate_text(messages, cache=self.cache, temperature=temperature)
        self.cache = updated_cache
        clean_resp = clean_statement(response)
        self.history.append({"role": "assistant", "content": clean_resp})
        return clean_resp

    def monitor(
        self,
        target_id: int,
        statement: str,
        round_idx: int,
    ) -> Tuple[float, str]:
        monitor_messages = [
            {"role": "system", "content": MONITOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"You are Player {self.id} monitoring Round {round_idx}.\n"
                    f"Player {target_id} said: \"{statement}\".\n"
                    "Return exactly:\nScore: <0-10>\nRationale: <one sentence>."
                ),
            },
        ]
        monitor_cache = clone_cache(self.cache)
        response, _ = self.backend.generate_text(monitor_messages, cache=monitor_cache)
        score = extract_score(response)
        return score, response


def format_history(public_history: Sequence[Tuple[int, str]]) -> str:
    if not public_history:
        return ""
    return "; ".join(f"Player {pid}: {stmt}" for pid, stmt in public_history)


def choose_spies(num_agents: int, num_spies: int) -> List[int]:
    spy_count = min(num_spies, num_agents - 1) if num_agents > 1 else 0
    return random.sample(range(num_agents), spy_count) if spy_count else []


def run_single_game(config: GameConfig, backend: LLMBackend, run_id: int = 0) -> Dict:
    spies = set(choose_spies(config.num_agents, config.num_spies))
    agents: List[Agent] = []
    for idx in range(config.num_agents):
        role = "Spy" if idx in spies else "Civilian"
        word = config.spy_word if role == "Spy" else config.civilian_word
        agents.append(Agent(idx, role, word, backend))

    suspicion_matrices = []
    round_summaries = []
    public_history: List[Tuple[int, str]] = []

    round_metrics = []

    for r in range(config.rounds):
        statements: Dict[int, str] = {}
        history_snapshot = list(public_history)  # blind action: no within-round leakage
        for agent in agents:
            stmt = agent.act(history_snapshot, temperature=config.temperature)
            statements[agent.id] = stmt
        public_history.extend(statements.items())

        matrix = np.zeros((config.num_agents, config.num_agents), dtype=float)
        critiques: Dict[Tuple[int, int], str] = {}

        for observer in agents:
            for target_id, stmt in statements.items():
                if observer.id == target_id:
                    continue
                score, critique = observer.monitor(target_id, stmt, r)
                matrix[observer.id][target_id] = score
                critiques[(observer.id, target_id)] = critique

        suspicion_matrices.append(matrix)
        spectral = compute_spectral_metrics(matrix)
        mi_matrix = compute_mi_matrix(matrix)
        mi_mean = float(np.mean(mi_matrix[np.triu_indices(config.num_agents, k=1)])) if config.num_agents > 1 else 0.0
        hic = compute_hic(matrix)
        round_metrics.append(
            {
                "spectral": spectral,
                "mean_mi": mi_mean,
                "hic": hic,
            }
        )
        round_summaries.append(
            {
                "round": r,
                "statements": statements,
                "critiques": {f"{k[0]}->{k[1]}": v for k, v in critiques.items()},
                "metrics": round_metrics[-1],
            }
        )

    summed_suspicion = np.sum(suspicion_matrices, axis=0)
    predicted_spy = int(np.argmax(summed_suspicion.sum(axis=0)))
    detected = predicted_spy in spies

    separability = compute_separability(suspicion_matrices[-1], spies)
    convergence_round = find_convergence_round(suspicion_matrices, spies)

    return {
        "run_id": run_id,
        "spies": sorted(spies),
        "predicted_spy": predicted_spy,
        "detected": bool(detected),
        "summed_suspicion": summed_suspicion.tolist(),
        "rounds": round_summaries,
        "separability": separability,
        "convergence_round": convergence_round,
        "mean_eigengap": float(np.mean([m["spectral"]["eigengap"] for m in round_metrics])) if round_metrics else 0.0,
        "mean_hic": float(np.mean([m["hic"] for m in round_metrics])) if round_metrics else 0.0,
        "mean_mi": float(np.mean([m["mean_mi"] for m in round_metrics])) if round_metrics else 0.0,
    }


def compute_separability(matrix: np.ndarray, spies: set) -> float:
    if not spies or matrix.size == 0:
        return 0.0
    civilian_rows = [matrix[i] for i in range(matrix.shape[0]) if i not in spies]
    if not civilian_rows:
        return 0.0
    spy_rows = [matrix[i] for i in spies]
    civilian_mean = np.mean(civilian_rows, axis=0)
    spy_mean = np.mean(spy_rows, axis=0)
    return float(np.linalg.norm(spy_mean - civilian_mean))


def compute_spectral_metrics(matrix: np.ndarray) -> Dict[str, object]:
    try:
        eigvals = np.linalg.eigvals(matrix)
        eigvals_real = np.real(eigvals)
        sorted_vals = np.sort(eigvals_real)[::-1]
        eigengap = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else 0.0
        return {"eigvals": eigvals_real.tolist(), "eigengap": eigengap}
    except Exception:
        return {"eigvals": [], "eigengap": 0.0}


def _discretize_vector(vec: np.ndarray, bins: int = 5) -> np.ndarray:
    edges = np.linspace(0, 10, bins + 1)
    return np.digitize(vec, edges) - 1


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 5) -> float:
    a_bins = _discretize_vector(a, bins)
    b_bins = _discretize_vector(b, bins)
    joint, _, _ = np.histogram2d(a_bins, b_bins, bins=bins)
    joint = joint / joint.sum() if joint.sum() else joint
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (pa[i] * pb[j]))
    return float(mi)


def compute_mi_matrix(matrix: np.ndarray, bins: int = 5) -> np.ndarray:
    n = matrix.shape[0]
    mi = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = _mutual_information(matrix[i], matrix[j], bins=bins)
            mi[i, j] = mi[j, i] = val
    return mi


def compute_hic(matrix: np.ndarray) -> float:
    """Proxy for Heron's Information Coefficient using triangle areas over suspicion graph."""
    n = matrix.shape[0]
    if n < 3:
        return 0.0
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = matrix[i, j]
                b = matrix[j, k]
                c = matrix[k, i]
                s = (a + b + c) / 2.0
                area_sq = s * (s - a) * (s - b) * (s - c)
                if area_sq > 0:
                    areas.append(math.sqrt(area_sq))
    return float(np.mean(areas)) if areas else 0.0


def find_convergence_round(matrices: List[np.ndarray], spies: set) -> Optional[int]:
    if not matrices:
        return None
    top_history = []
    for m in matrices:
        summed = m.sum(axis=0)
        top_history.append(int(np.argmax(summed)))
    final_top = top_history[-1]
    for idx, val in enumerate(top_history):
        if val == final_top:
            return idx
    return None


def print_cli_recipes() -> None:
    """Print ready-to-run CLI commands for each experimental configuration."""
    cmds = {
        "Baseline": "--agents 3 --spies 1 --rounds 3 --runs 20 --tag baseline",
        "Small Scale": "--agents 5 --spies 1 --rounds 3 --runs 50 --tag small",
        "Medium Scale": "--agents 10 --spies 2 --rounds 4 --runs 50 --tag medium",
        "Large Scale": "--agents 32 --spies 4 --rounds 4 --runs 20 --tag large",
        "Control": "--agents 10 --spies 0 --rounds 3 --runs 20 --tag control",
    }
    prefix = ".venv/bin/python3 benchmark.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --save-dir logs"
    print("\nRecommended CLI recipes (real model):")
    for label, args in cmds.items():
        print(f"- {label}: {prefix} {args}")
    mock_prefix = ".venv/bin/python3 benchmark.py --mock --save-dir logs"
    print("\nQuick mock dry-runs:")
    for label, args in cmds.items():
        print(f"- {label}: {mock_prefix} {args}")


def save_results(results: List[Dict], config: GameConfig) -> None:
    if config.save_dir is None:
        return
    ensure_dir(config.save_dir)
    detection_rate = sum(1 for r in results if r["detected"]) / len(results) if results else 0.0
    mean_separability = float(
        np.mean([r["separability"] for r in results]) if results else 0.0
    )
    mean_eigengap = float(np.mean([r["mean_eigengap"] for r in results])) if results else 0.0
    mean_hic = float(np.mean([r["mean_hic"] for r in results])) if results else 0.0
    mean_mi = float(np.mean([r["mean_mi"] for r in results])) if results else 0.0
    out_path = config.save_dir / f"panopticon_results_{config.tag or 'run'}_{int(time.time())}.json"
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(config).items()}
    with out_path.open("w") as f:
        json.dump(
            {
                "config": config_dict,
                "summary": {
                    "detection_rate": detection_rate,
                    "mean_separability": mean_separability,
                    "mean_eigengap": mean_eigengap,
                    "mean_hic": mean_hic,
                    "mean_mi": mean_mi,
                    "runs": len(results),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digital Panopticon experiment harness.")
    parser.add_argument("--model", default=GameConfig.model_path, help="MLX model path")
    parser.add_argument("--agents", type=int, default=GameConfig.num_agents, help="Number of agents (N)")
    parser.add_argument("--spies", type=int, default=GameConfig.num_spies, help="Number of spies (f)")
    parser.add_argument("--spy-ratio", type=float, default=None, help="Optional fraction of spies (overrides --spies)")
    parser.add_argument("--rounds", type=int, default=GameConfig.rounds, help="Rounds per game")
    parser.add_argument("--runs", type=int, default=1, help="How many independent games to run")
    parser.add_argument("--temperature", type=float, default=GameConfig.temperature, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=GameConfig.max_tokens, help="Max tokens per generation")
    parser.add_argument("--spy-word", default=GameConfig.spy_word, help="Secret word used by spies")
    parser.add_argument("--civilian-word", default=GameConfig.civilian_word, help="Secret word used by civilians")
    parser.add_argument("--mock", action="store_true", help="Run with a lightweight mock model")
    parser.add_argument("--seed", type=int, default=GameConfig.seed, help="RNG seed for reproducibility")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to write JSON results",
    )
    parser.add_argument("--tag", default=None, help="Optional label to stamp into saved results")
    parser.add_argument(
        "--show-recipes",
        action="store_true",
        help="Print recommended CLI commands for each experimental phase and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_recipes:
        print_cli_recipes()
        return

    num_spies = args.spies
    if args.spy_ratio is not None:
        num_spies = max(1, math.ceil(args.agents * args.spy_ratio))

    config = GameConfig(
        model_path=args.model,
        num_agents=args.agents,
        num_spies=num_spies,
        rounds=args.rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        spy_word=args.spy_word,
        civilian_word=args.civilian_word,
        mock=args.mock,
        seed=args.seed,
        save_dir=args.save_dir,
        tag=args.tag,
    )

    safe_seed(config.seed)
    print(f"Configuration: {config}")

    backend = LLMBackend(config, DEFAULT_SYSTEM_PROMPT)

    results: List[Dict] = []
    start = time.time()
    for run in range(args.runs):
        print(f"\n=== Game {run + 1}/{args.runs} ===")
        game_result = run_single_game(config, backend, run_id=run)
        results.append(game_result)
        print(f"Spies: {game_result['spies']} | Predicted: {game_result['predicted_spy']} | Detected: {game_result['detected']}")
        print(f"Separability (last round): {game_result['separability']:.3f}")
        if game_result["convergence_round"] is not None:
            print(f"Convergence round: {game_result['convergence_round']}")

    duration = time.time() - start
    detection_rate = sum(1 for r in results if r["detected"]) / len(results)
    print(f"\nCompleted {len(results)} run(s) in {duration:.2f}s | Detection rate: {detection_rate:.2%}")

    save_results(results, config)


if __name__ == "__main__":
    main()
