"""V2 自动竞价 demo：以 GAVE 为基础出价器，结合经验知识库和多 LLM 决策代理。

流程：
1. 读取 final_aigb 的 trajectory/traffic 数据，并根据 test period 选择经验 period。
2. 加载或训练 GAVE，用它输出基础 alpha 和基础 bid。
3. 将非测试 period 的 traffic 明细合并为 SQLite，再预聚合成经验知识库 KB。
4. 每个 timestep 调用市场价值、预算控制、成功经验三个分析 agent。
5. 最终策略 agent 汇总三个分析结果，输出 alpha 调整和 pValue 分层 multiplier。
6. 离线模拟竞价，输出 GAVE baseline、agent 结果、comparison 和 decision trace。

English:
V2 auto-bidding demo: GAVE base bidder + experience KB + multi-LLM decision agents.

Pipeline:
1. Load final AIGB trajectory/traffic data and select test/experience periods.
2. Load or train GAVE, then use it to produce the base alpha and base bids.
3. Build raw experience SQLite from non-test periods, then pre-aggregate it into a KB.
4. For each timestep, run three analysis agents: market value, budget control, and success experience.
5. Let the final strategy agent merge those opinions into alpha adjustment and pValue-segment multipliers.
6. Run offline auction simulation and write GAVE baseline, agent results, comparison, and decision traces."""

# 流程备注 / Pipeline notes:
# 1. 选择 final_aigb 测试/经验 period，并加载 traffic/trajectory 数据。
#    Select final_aigb test/experience periods and load traffic/trajectory data.
# 2. 加载 300000 步 GAVE 权重，生成 base alpha 和 base bid。
#    Load the 300000-step GAVE checkpoint to produce base alpha and base bids.
# 3. 构建原始经验 SQLite 和 agent 使用的预聚合 KB。
#    Build raw experience SQLite and the pre-aggregated KB used by the agents.
# 4. 运行市场价值、预算控制、成功经验三个分析 agent。
#    Run market-value, budget-control, and success-experience agents.
# 5. 将分析结果合并为最终 alpha 调整和分层 multiplier。
#    Merge their opinions into a final alpha adjustment and segment multipliers.
# 6. 离线模拟竞价，并写出 baseline、agent、comparison 和 trace CSV。
#    Simulate auctions and write baseline, agent, comparison, and trace CSVs.

import argparse
import asyncio
import csv
import importlib.util
import json
import math
import os
import pickle
import random
import sqlite3
import sys
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
GAVE_CODE_DIR = WORKSPACE_ROOT / "gave" / "code"
FINAL_AIGB_ROOT = PROJECT_ROOT / "data" / "final_aigb"
FINAL_AIGB_TRAJECTORY_CSV = FINAL_AIGB_ROOT / "trajectory" / "trajectory_data.csv"
FINAL_AIGB_TRAFFIC_DIR = FINAL_AIGB_ROOT / "traffic"
DEFAULT_TRAJECTORY_CSV = (
    FINAL_AIGB_TRAJECTORY_CSV
    if FINAL_AIGB_TRAJECTORY_CSV.exists()
    else WORKSPACE_ROOT / "aigb_track_data_trajectory_data" / "trajectory_data.csv"
)
DEFAULT_SQL_CSV = (
    FINAL_AIGB_TRAFFIC_DIR / "period-7.csv"
    if (FINAL_AIGB_TRAFFIC_DIR / "period-7.csv").exists()
    else WORKSPACE_ROOT / "aigb_track_data_period_7-8" / "period-7.csv"
)
DEFAULT_TEST_CSV = (
    FINAL_AIGB_TRAFFIC_DIR / "period-8.csv"
    if (FINAL_AIGB_TRAFFIC_DIR / "period-8.csv").exists()
    else WORKSPACE_ROOT / "aigb_track_data_period_7-8" / "period-8.csv"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_FULL_GAVE_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "gave_full_steps300000_seed2026_20260510_202440"
)
RTBAGENT_DIR = WORKSPACE_ROOT / "RTBagent"
MIN_ADJUSTMENT = -6.0
MAX_ADJUSTMENT = 4.0
DEFAULT_TOTAL_TIMESTEPS = 48

# 调价区间构造 / Adjustment range construction.
def make_adjustment_ranges(min_value: float, max_value: float, step: float = 0.1) -> List[str]:
    """构造 alpha 加性调价的展示区间。

English:
Build display buckets for additive alpha adjustment."""
    ranges = []
    value = min_value
    while value < max_value - 1e-9:
        upper = min(value + step, max_value)
        right = "]" if upper >= max_value - 1e-9 else ")"
        ranges.append(f"[{value:.1f},{upper:.1f}{right}")
        value = upper
    return ranges


ADJUSTMENT_RANGES = make_adjustment_ranges(MIN_ADJUSTMENT, MAX_ADJUSTMENT)

# 运行环境检查 / Runtime environment check.
def require_runtime_dependencies() -> None:
    """提前检查当前 Python 环境是否能运行 GAVE 和 demo。

English:
Fail fast if the current Python environment cannot run GAVE/demo code."""
    missing = []
    for module_name in ("numpy", "pandas", "torch", "gin"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "Missing required modules: {}. Install them in the Python environment used to run this demo. "
            "For GAVE, `gin` is provided by the `gin-config` package.".format(", ".join(missing))
        )

# 引入 GAVE / Import GAVE package path.
def add_gave_to_path() -> None:
    """把 gave/code 加入导入路径，避免直接修改原始 GAVE 仓库。

English:
Make `gave/code` importable without modifying the original GAVE repository."""
    gave_code = str(GAVE_CODE_DIR)
    if gave_code not in sys.path:
        sys.path.insert(0, gave_code)

# 引入 RTBagent / Import RTBagent package path.
def add_rtbagent_to_path() -> None:
    """把 RTBagent 加入导入路径，用于可选的 MetaGPT provider 客户端。

English:
Make RTBagent importable for the optional MetaGPT provider client."""
    rtbagent = str(RTBAGENT_DIR)
    if rtbagent not in sys.path:
        sys.path.insert(0, rtbagent)

# 随机种子：控制模拟竞价过程中的随机性 / Seed randomness for offline auction simulation.
def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子。

English:
Set random seeds for Python, NumPy and PyTorch."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 安全转 float / Safe float conversion.
def as_float(value: Any, default: float = 0.0) -> float:
    """把 CSV、SQLite、JSON 中的值安全转换为 float。

English:
Convert values from CSV/SQLite/JSON to float with a fallback."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免分母为 0 导致崩溃。

English:
Division helper that avoids zero-denominator crashes."""
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def clip(value: float, lower: float, upper: float) -> float:
    """把数值裁剪到 [lower, upper] 区间内。

English:
Clamp a numeric value into [lower, upper]."""
    return max(lower, min(upper, value))


def parse_max_keys(value: str) -> Optional[int]:
    """解析 --max-keys，其中 all 表示不限制广告主 key 数量。

English:
Parse --max-keys, where `all` means no advertiser-key limit."""
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-keys must be a positive integer or `all`.")
    return parsed


def newest_checkpoint(save_dir: Path) -> Optional[Path]:
    """返回目录下编号最新的 .pt 权重文件。

English:
Return the newest numeric .pt checkpoint under a directory."""
    checkpoints = []
    for path in save_dir.glob("*.pt"):
        try:
            checkpoints.append((int(path.stem), path))
        except ValueError:
            continue
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda item: item[0])[-1][1]


def requested_checkpoint(save_dir: Path, step_num: int) -> Optional[Path]:
    """返回 CLI 指定 step_num 对应的 checkpoint。

English:
Return the checkpoint explicitly requested by the CLI step number."""
    path = save_dir / f"{int(step_num)}.pt"
    return path if path.exists() else None


def file_fingerprint(path: Path) -> str:
    """基于路径、文件大小和修改时间生成轻量缓存指纹。

English:
Small cache fingerprint based on path, size and modified time."""
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def files_fingerprint(paths: Sequence[Path]) -> str:
    """为一组 CSV 生成指纹，使缓存文件名能对应数据来源组合。

English:
Fingerprint a set of CSVs so cache filenames track the source combination."""
    payload = "|".join(
        f"{path.resolve()}:{path.stat().st_size}:{int(path.stat().st_mtime)}"
        for path in sorted((Path(p) for p in paths), key=lambda item: str(item))
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def baseline_cache_paths(args: argparse.Namespace, checkpoint: Path, model_param: Dict[str, Any]) -> Tuple[Path, Path]:
    """计算确定性 GAVE baseline 评估对应的缓存文件路径。

English:
Compute cache files for a deterministic GAVE baseline evaluation."""
    cache_key = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_tag": file_fingerprint(Path(checkpoint)),
        "test_csv": str(Path(args.test_csv).resolve()),
        "test_tag": file_fingerprint(Path(args.test_csv)),
        "max_keys": "all" if args.max_keys is None else args.max_keys,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "budget_rate": model_param.get("budget_rate", 1.0),
        "eval_version": "gave_baseline_v2",
    }
    digest = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_dir = DEFAULT_CACHE_DIR / "baseline_gave"
    return cache_dir / f"{digest}_results.csv", cache_dir / f"{digest}_summary.json"


def final_period_csv(period: int) -> Path:
    """生成 final_aigb 中 period-N.csv 的路径。

English:
Path helper for final_aigb traffic period-N.csv."""
    return FINAL_AIGB_TRAFFIC_DIR / f"period-{period}.csv"


def available_final_periods() -> List[int]:
    """扫描 data/final_aigb/traffic 下可用的 period 文件。

English:
Discover available final_aigb traffic periods from data/final_aigb/traffic."""
    periods = []
    if not FINAL_AIGB_TRAFFIC_DIR.exists():
        return periods
    for path in FINAL_AIGB_TRAFFIC_DIR.glob("period-*.csv"):
        try:
            periods.append(int(path.stem.replace("period-", "")))
        except ValueError:
            continue
    return sorted(periods)


def load_llm_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """从命令行 JSON、配置文件或环境变量读取 RTBagent/MetaGPT 风格的 LLM 配置。

English:
Load RTBagent/MetaGPT-style LLM config from CLI JSON, file, or env var."""
    if args.llm_config_json:
        return json.loads(args.llm_config_json)
    if args.llm_config_file:
        with open(args.llm_config_file, "r", encoding="utf-8") as file:
            text = file.read()
        if args.llm_config_file.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(text)
        return json.loads(text)
    env_config = os.getenv("AUTOBID_LLM_CONFIG")
    if env_config:
        return json.loads(env_config)
    return None


def extract_json_object(text: str) -> Dict[str, Any]:
    """从 LLM 返回文本中提取并解析第一个 JSON 对象。

English:
Extract and parse the first JSON object from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"LLM response does not contain a JSON object: {text[:300]}")
    return json.loads(text[start : end + 1])


class RTBAgentLLMClient:
    """可选 LLM 客户端，复用 RTBagent/MetaGPT 的 provider 注册机制。

English:
Optional LLM client that reuses RTBagent/MetaGPT provider registration.

这个类保留 RTBagent 风格的大模型接入方式；当前常用的是下面的
OpenAICompatibleLLMClient，因为 DashScope 兼容 OpenAI chat-completions API。"""

    def __init__(self, llm_config: Dict[str, Any], timeout: int = 60):
        add_rtbagent_to_path()
        # 导入 provider 模块以触发 @register_provider 注册逻辑。
        # Import provider modules so their @register_provider decorators run.
        import metagpt.provider.anthropic_api  # noqa: F401
        import metagpt.provider.azure_openai_api  # noqa: F401
        import metagpt.provider.dashscope_api  # noqa: F401
        import metagpt.provider.google_gemini_api  # noqa: F401
        import metagpt.provider.ollama_api  # noqa: F401
        import metagpt.provider.openai_api  # noqa: F401
        import metagpt.provider.qianfan_api  # noqa: F401
        import metagpt.provider.spark_api  # noqa: F401
        import metagpt.provider.zhipuai_api  # noqa: F401
        from metagpt.configs.llm_config import LLMConfig
        from metagpt.provider.llm_provider_registry import create_llm_instance

        config = dict(llm_config)
        config.setdefault("temperature", 0.0)
        config.setdefault("max_token", 512)
        config.setdefault("timeout", timeout)
        self.llm = create_llm_instance(LLMConfig.model_validate(config))
        self.timeout = timeout

    async def aask(self, prompt: str) -> str:
        """Call the async MetaGPT LLM provider and return raw text."""
        messages = [{"role": "user", "content": prompt}]
        return await self.llm.acompletion_text(messages, timeout=self.timeout)

    def ask_json(self, prompt: str) -> Dict[str, Any]:
        """供 agent 同步调用的包装器，返回解析后的 JSON。

English:
Synchronous wrapper used by the agent code; returns parsed JSON."""
        response = asyncio.run(self.aask(prompt))
        return extract_json_object(response)


class OpenAICompatibleLLMClient:
    """直接调用 OpenAI-compatible chat completions 接口的客户端。

English:
Direct OpenAI-compatible chat-completions client.

默认用于 DashScope：读取 DASHSCOPE_API_KEY/LLM_API_KEY，调用 qwen3.5-flash
等模型，并强制要求输出严格 JSON，方便后续程序化解析。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.5-plus",
        timeout: int = 120,
        retries: int = 3,
        retry_delay: float = 5.0,
    ):
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("Mode `llm` requires the `openai` package in the current Python environment.")
        from openai import OpenAI

        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Mode `llm` requires LLM_API_KEY/DASHSCOPE_API_KEY in the environment or --llm-api-key.")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.retries = max(0, retries)
        self.retry_delay = max(0.0, retry_delay)

    def ask_json(self, prompt: str) -> Dict[str, Any]:
        """带重试调用 chat-completions，并把响应解析为 JSON。

English:
Call chat-completions with retry, then parse the response as JSON."""
        last_error: Optional[Exception] = None
        total_attempts = self.retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=700,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert RTB auto-bidding agent. "
                                "Always return strict JSON only, without markdown fences."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return extract_json_object(completion.choices[0].message.content or "")
            except Exception as exc:
                last_error = exc
                if attempt >= total_attempts:
                    break
                print(
                    f"[demo] LLM request failed on attempt {attempt}/{total_attempts}: {type(exc).__name__}: {exc}. "
                    f"Retrying in {self.retry_delay}s...",
                    flush=True,
                )
                time.sleep(self.retry_delay)
        raise RuntimeError(f"LLM request failed after {total_attempts} attempts: {last_error}") from last_error


def build_model_param(args: argparse.Namespace, save_dir: Path) -> Dict[str, Any]:
    """把 CLI 参数转换为 GAVE 的 run_dt/GAVE 类需要的 model_param 字典。

English:
Translate CLI args into the model_param dict expected by GAVE's run_dt/GAVE classes."""
    return {
        "step_num": args.step_num,
        "save_step": args.save_step,
        "dir": str(args.trajectory_csv),
        "test_csv": str(args.test_csv),
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "time_dim": args.time_dim,
        "batch_size": args.batch_size,
        "device": args.device,
        "expectile": args.expectile,
        "loss_report": args.loss_report,
        "budget_rate": args.budget_rate,
        "gave_target_return": args.gave_target_return,
        "save_dir": str(save_dir),
        "block_config": {
            "n_ctx": args.n_ctx,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_inner": args.n_inner,
            "activation_function": args.activation_function,
            "n_position": args.n_position,
            "resid_pdrop": args.resid_pdrop,
            "attn_pdrop": args.attn_pdrop,
        },
    }


def train_or_load_gave(args: argparse.Namespace, model_param: Dict[str, Any], save_dir: Path) -> Path:
    """优先加载已有 GAVE 权重；如果没有可用权重则启动训练。

English:
Load an existing GAVE checkpoint, or train one if no usable checkpoint exists."""
    add_gave_to_path()
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = requested_checkpoint(save_dir, args.step_num)
    normalize_dict = save_dir / "normalize_dict.pkl"
    if existing is not None and normalize_dict.exists() and not args.force_train:
        print(f"[demo] Loading requested GAVE checkpoint: {existing}")
        return existing
    if normalize_dict.exists() and not args.force_train:
        raise FileNotFoundError(
            f"Requested checkpoint {save_dir / f'{int(args.step_num)}.pt'} was not found. "
            "Set --step-num to an existing checkpoint step, or use --force-train to train it."
        )

    from run.run_decision_transformer import run_dt

    print(f"[demo] Training GAVE model into: {save_dir}")
    run_dt(
        device=model_param["device"],
        step_num=model_param["step_num"],
        dir=model_param["dir"],
        save_step=model_param["save_step"],
        model_param=model_param,
        batch_size=model_param["batch_size"],
        save_dir=model_param["save_dir"],
        loss_report=model_param["loss_report"],
    )
    checkpoint = requested_checkpoint(save_dir, args.step_num)
    if checkpoint is None:
        raise RuntimeError(
            f"Requested checkpoint {save_dir / f'{int(args.step_num)}.pt'} was not produced in {save_dir}. "
            "Use --save-step less than or equal to --step-num."
        )
    return checkpoint


class TrafficDataLoader:
    """基于 CSV 的测试流量加载器。

English:
CSV-backed test traffic loader.

GAVE 原始 TestDataLoader 会复用 raw_data.pickle，容易在多 period 实验中串缓存。
这里直接按 CSV 读取并按 (deliveryPeriodIndex, advertiserNumber) 分组。"""

    def __init__(self, file_path: Path, max_keys: Optional[int], chunksize: int, max_rows: Optional[int] = None):
        import pandas as pd

        self.file_path = Path(file_path)
        self.raw_data = self._read_data(max_keys=max_keys, chunksize=chunksize, max_rows=max_rows)
        grouped = self.raw_data.sort_values(["deliveryPeriodIndex", "advertiserNumber", "timeStepIndex", "pvIndex"])
        self.test_dict = {
            key: group for key, group in grouped.groupby(["deliveryPeriodIndex", "advertiserNumber"], sort=True)
        }
        self.keys = list(self.test_dict.keys())
        self.advertisers = sorted({float(key[1]) for key in self.keys})
        self.categories = sorted({float(group["advertiserCategoryIndex"].iloc[0]) for group in self.test_dict.values()})

    def _read_data(self, max_keys: Optional[int], chunksize: int, max_rows: Optional[int]):
        """读取完整 CSV、前 N 行，或流式读取前 N 个广告主 key。

English:
Read either full CSV, first N rows, or streaming first N advertiser keys."""
        import pandas as pd

        if max_rows is not None:
            print(f"[demo] Loading first {max_rows} traffic rows from test CSV: {self.file_path}")
            return pd.read_csv(self.file_path, nrows=max_rows)

        if max_keys is None:
            print(f"[demo] Loading full test CSV: {self.file_path}")
            return pd.read_csv(self.file_path)

        print(f"[demo] Streaming test CSV for first {max_keys} advertiser keys: {self.file_path}")
        selected_keys: List[Tuple[float, float]] = []
        selected_key_set = set()
        frames = []
        key_columns = ["deliveryPeriodIndex", "advertiserNumber"]
        for chunk in pd.read_csv(self.file_path, chunksize=chunksize):
            for row in chunk[key_columns].drop_duplicates().itertuples(index=False, name=None):
                key = (float(row[0]), float(row[1]))
                if key not in selected_key_set and len(selected_keys) < max_keys:
                    selected_keys.append(key)
                    selected_key_set.add(key)
            if selected_key_set:
                mask = [
                    (float(period), float(advertiser)) in selected_key_set
                    for period, advertiser in zip(chunk["deliveryPeriodIndex"], chunk["advertiserNumber"])
                ]
                filtered = chunk.loc[mask]
                if not filtered.empty:
                    frames.append(filtered)
        if not frames:
            raise RuntimeError(f"No rows found for the first {max_keys} advertiser keys in {self.file_path}")
        return pd.concat(frames, ignore_index=True)

    def iter_keys(self, max_keys: Optional[int]) -> Iterable[Tuple[float, float]]:
        """按确定顺序返回需要评估的广告主 key。

English:
Yield evaluated advertiser keys in deterministic order."""
        if max_keys is None:
            yield from self.keys
        else:
            yield from self.keys[:max_keys]

    def mock_data(self, key: Tuple[float, float]) -> Tuple[int, list, list, list, float, float, float]:
        """返回离线竞价模拟器需要的逐 timestep 数组。

English:
Return per-timestep arrays needed by the offline bidding simulator."""
        import numpy as np

        data = self.test_dict[key]
        grouped = data.groupby("timeStepIndex", sort=True)
        p_values = grouped["pValue"].apply(list).apply(np.array).tolist()
        p_value_sigmas = grouped["pValueSigma"].apply(list).apply(np.array).tolist()
        least_winning_costs = grouped["leastWinningCost"].apply(list).apply(np.array).tolist()
        return (
            len(p_values),
            p_values,
            p_value_sigmas,
            least_winning_costs,
            float(data["budget"].iloc[0]),
            float(data["CPAConstraint"].iloc[0]),
            float(data["advertiserCategoryIndex"].iloc[0]),
        )


class GaveTool:
    """把训练好的 GAVE 权重封装为基础出价工具。

English:
Adapter that turns a trained GAVE checkpoint into a base bidding tool.

输入当前 state/history，输出：
- base_alpha: GAVE 预测的基础出价系数
- base_bids: base_alpha * pValue"""

    def __init__(self, checkpoint: Path, model_param: Dict[str, Any]):
        import torch
        from bidding_train_env.baseline.dt.dt import GAVE

        self.device = model_param["device"]
        self.target_return = float(model_param.get("gave_target_return", 0.01))
        with open(Path(model_param["save_dir"]) / "normalize_dict.pkl", "rb") as file:
            normalize_dict = pickle.load(file)
        self.model = GAVE(
            state_dim=16,
            act_dim=1,
            hidden_size=model_param["hidden_size"],
            state_mean=normalize_dict["state_mean"],
            state_std=normalize_dict["state_std"],
            device=model_param["device"],
            learning_rate=model_param["learning_rate"],
            time_dim=model_param["time_dim"],
            block_config=model_param["block_config"],
            expectile=model_param["expectile"],
        ).to(self.device)
        self.model.load_net(str(checkpoint), device=self.device)
        self.torch = torch

    def bid(
        self,
        time_step_index: int,
        p_values: Any,
        budget: float,
        cpa: float,
        remaining_budget: float,
        history: Dict[str, list],
    ) -> Tuple[float, Any]:
        """对单个 timestep 运行 GAVE，返回 alpha 和每个 PV 的 bid。

English:
Run GAVE for one timestep and return alpha plus per-PV bids."""
        import numpy as np

        if time_step_index == 0:
            self.model.init_eval()

        test_state = build_gave_state(
            time_step_index=time_step_index,
            p_values=p_values,
            budget=budget,
            remaining_budget=remaining_budget,
            history=history,
        )
        history_conversion = [result[:, 1] for result in history["historyImpressionResult"]]
        pre_reward = float(np.sum(history_conversion[-1])) if history_conversion else None
        target_return = self.torch.tensor(self.target_return, dtype=self.torch.float32, device=self.device)
        alpha = self.model.take_actions(
            test_state,
            target_return=target_return,
            budget=budget,
            cpa=cpa,
            pre_reward=pre_reward,
        )
        alpha_value = max(float(np.asarray(alpha).reshape(-1)[0]), 0.0)
        return alpha_value, alpha_value * p_values


def build_gave_state(
    time_step_index: int,
    p_values: Any,
    budget: float,
    remaining_budget: float,
    history: Dict[str, list],
) -> Any:
    """根据在线历史重建 GAVE 评估所需的 16 维状态。

English:
Rebuild the 16-dimensional GAVE evaluation state from online history.

这个 state 要和 GAVE 训练时 trajectory 中的 state 语义保持一致：
时间剩余、预算剩余、历史 bid/market/pValue/conversion/win 等统计。"""
    import numpy as np

    history_xi = [result[:, 0] for result in history["historyAuctionResult"]]
    history_p_value = [result[:, 0] for result in history["historyPValueInfo"]]
    history_conversion = [result[:, 1] for result in history["historyImpressionResult"]]

    def mean_nested(items: Sequence[Any]) -> float:
        return float(np.mean([np.mean(item) for item in items])) if items else 0.0

    def mean_last(items: Sequence[Any], n: int) -> float:
        if not items:
            return 0.0
        last_items = items[max(0, len(items) - n) : len(items)]
        return float(np.mean([np.mean(item) for item in last_items])) if last_items else 0.0

    historical_pv_num_total = sum(len(bids) for bids in history["historyBids"]) if history["historyBids"] else 0
    last_three_pv_num_total = (
        sum(len(history["historyBids"][i]) for i in range(max(0, time_step_index - 3), time_step_index))
        if history["historyBids"]
        else 0
    )

    return np.array(
        [
            (48 - time_step_index) / 48,
            remaining_budget / budget if budget > 0 else 0.0,
            mean_nested(history["historyBids"]),
            mean_last(history["historyBids"], 3),
            mean_nested(history["historyLeastWinningCost"]),
            mean_nested(history_p_value),
            mean_nested(history_conversion),
            mean_nested(history_xi),
            mean_last(history["historyLeastWinningCost"], 3),
            mean_last(history_p_value, 3),
            mean_last(history_conversion, 3),
            mean_last(history_xi, 3),
            float(np.mean(p_values)) if len(p_values) else 0.0,
            len(p_values),
            last_three_pv_num_total,
            historical_pv_num_total,
        ],
        dtype=np.float32,
    )


STATE_FEATURE_NAMES = [
    "time_left_ratio",
    "budget_left_ratio",
    "historical_bid_mean",
    "last_three_bid_mean",
    "historical_least_winning_cost_mean",
    "historical_pvalue_mean",
    "historical_conversion_mean",
    "historical_win_rate_mean",
    "last_three_least_winning_cost_mean",
    "last_three_pvalue_mean",
    "last_three_conversion_mean",
    "last_three_win_rate_mean",
    "current_pvalue_mean",
    "current_pv_count",
    "last_three_pv_count",
    "historical_pv_count",
]


def state_to_dict(state: Any) -> Dict[str, float]:
    """把 GAVE 数值状态向量转换为带字段名的字典，供 prompt 和 trace 使用。

English:
Convert the numeric GAVE state vector into a named dict for prompts/traces."""
    return {name: as_float(value) for name, value in zip(STATE_FEATURE_NAMES, state)}


def delivery_phase_policy(time_step_index: int) -> Dict[str, Any]:
    """根据投放早中晚阶段返回相应的竞价指导。

English:
Return phase-specific bidding guidance for early/mid/late delivery steps."""
    if time_step_index < 12:
        return {
            "phase": "cold_start",
            "step_range": "0-11",
            "policy": (
                "Explore cautiously and collect signal. Do not raise bids aggressively only because realized CPA is low, "
                "conversions are few, or budget remains high. Early CPA has high variance. Prefer small additive alpha "
                "changes unless experience-period market/budget pacing references show severe underspend and premium/strong traffic exists."
            ),
            "adjustment_bias": "prefer hold or small changes; avoid repeated max upward adjustments",
        }
    if time_step_index < 24:
        return {
            "phase": "calibration",
            "step_range": "12-23",
            "policy": (
                "Balance delivery and CPA. Compare current spend ratio with experience-period advertiser/category/global budget_pacing, "
                "not only with elapsed time. Increase gradually when under-spending with CPA headroom; reduce lower-quartile traffic "
                "when CPA exceeds constraint or budget spend is ahead of reliable references."
            ),
            "adjustment_bias": "moderate changes; protect premium/strong traffic but avoid broad lower-quartile overspend",
        }
    if time_step_index < 36:
        return {
            "phase": "control",
            "step_range": "24-35",
            "policy": (
                "Prioritize staying inside CPA and budget pacing. Use experience-period budget_pacing as the reference for whether "
                "spend is too fast or too slow. Prefer conservative adjustments if CPA is above target; increase only when "
                "budget is materially behind reliable references and traffic quality is strong."
            ),
            "adjustment_bias": "risk control; avoid chasing volume with weak CPA evidence",
        }
    return {
        "phase": "endgame",
        "step_range": "36-47",
        "policy": (
            "Final delivery stage. Spend remaining budget only if CPA is healthy or premium/strong traffic evidence is strong. "
            "If CPA is above constraint, preserve score by limiting spend. If CPA is safely below target and budget is far "
            "behind experience-period pacing references, allow measured increases."
        ),
        "adjustment_bias": "score preservation with selective catch-up",
    }


def build_decision_memory(recent_decisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """总结近期 agent 决策与结果，作为下一次 LLM 调用的记忆。

English:
Summarize recent agent decisions and outcomes for the next LLM call.

这是一段轻量“记忆”：只保留最近 5 步和本广告主 episode 累计表现，
用于提醒 LLM 不要反复做同方向、低收益或高 CPA 的动作。"""
    if not recent_decisions:
        return {
            "last_reason": "",
            "recent_5_summary": "No previous LLM decision in this advertiser episode.",
            "recent_5_steps": [],
            "episode_summary": "No previous outcome in this advertiser episode.",
            "memory_risk_flags": [],
            "followup_advice": (
                "Start from current GAVE state and experience-period references. Avoid strong repeated moves until live "
                "win/cost/conversion feedback appears."
            ),
        }

    recent = list(recent_decisions[-5:])
    episode = list(recent_decisions)
    last = recent_decisions[-1]
    total_cost = sum(as_float(item.get("cost")) for item in recent)
    total_conversions = sum(as_float(item.get("conversions")) for item in recent)
    total_expected_conversions = sum(as_float(item.get("expected_conversions")) for item in recent)
    total_wins = sum(as_float(item.get("win_count")) for item in recent)
    avg_adjustment = safe_div(sum(as_float(item.get("adjustment")) for item in recent), len(recent), 0.0)
    avg_final_alpha = safe_div(sum(as_float(item.get("final_alpha")) for item in recent), len(recent), 0.0)
    recent_cpa = safe_div(total_cost, total_conversions, 0.0)
    recent_expected_cpa = safe_div(total_cost, total_expected_conversions, 0.0)
    cpa_constraint = as_float(last.get("cpa_constraint"))
    remaining_budget_ratio_after = as_float(last.get("remaining_budget_ratio_after"))
    cpa_so_far_after = as_float(last.get("cpa_so_far_after"))
    expected_cpa_so_far_after = as_float(last.get("expected_cpa_so_far_after"))
    lower_quartile_steps = sum(1 for item in recent if item.get("value_segment") == "lower_quartile")
    standard_value_steps = sum(1 for item in recent if item.get("value_segment") == "standard_value")
    strong_value_steps = sum(1 for item in recent if item.get("value_segment") == "strong_value")
    premium_value_steps = sum(1 for item in recent if item.get("value_segment") == "premium_value")
    episode_cost = sum(as_float(item.get("cost")) for item in episode)
    episode_conversions = sum(as_float(item.get("conversions")) for item in episode)
    episode_expected_conversions = sum(as_float(item.get("expected_conversions")) for item in episode)
    episode_wins = sum(as_float(item.get("win_count")) for item in episode)
    episode_positive_adjustments = sum(1 for item in episode if as_float(item.get("adjustment")) > 0.03)
    episode_negative_adjustments = sum(1 for item in episode if as_float(item.get("adjustment")) < -0.03)
    recent_negative_adjustments = sum(1 for item in recent if as_float(item.get("adjustment")) < -0.03)
    recent_positive_adjustments = sum(1 for item in recent if as_float(item.get("adjustment")) > 0.03)
    episode_expected_cpa = safe_div(episode_cost, episode_expected_conversions, 0.0)
    episode_realized_cpa = safe_div(episode_cost, episode_conversions, 0.0)
    episode_avg_adjustment = safe_div(sum(as_float(item.get("adjustment")) for item in episode), len(episode), 0.0)
    recent_cost_share = safe_div(total_cost, episode_cost, 0.0)
    recent_expected_score_start = as_float(recent[0].get("expected_score_so_far_after")) if recent else 0.0
    recent_expected_score_end = as_float(recent[-1].get("expected_score_so_far_after")) if recent else 0.0
    repeated_down_without_score_gain = (
        recent_negative_adjustments >= 3
        and recent_expected_score_start > 0
        and recent_expected_score_end <= recent_expected_score_start * 1.02
    )
    risk_flags = []
    if cpa_constraint > 0 and recent_expected_cpa > cpa_constraint * 1.15:
        risk_flags.append("recent_expected_cpa_above_constraint")
    if cpa_constraint > 0 and episode_expected_cpa > cpa_constraint * 1.15:
        risk_flags.append("episode_expected_cpa_above_constraint")
    if recent_cost_share > 0.45 and len(episode) >= 5:
        risk_flags.append("recent_cost_spike")
    if episode_positive_adjustments >= 4 and episode_expected_cpa > cpa_constraint and cpa_constraint > 0:
        risk_flags.append("repeated_upward_adjustments_with_expensive_traffic")
    if repeated_down_without_score_gain:
        risk_flags.append("repeated_downward_adjustments_without_score_gain")
    if remaining_budget_ratio_after < 0.2:
        risk_flags.append("budget_low")

    if total_cost <= 1e-9 and total_wins <= 0:
        advice = (
            "Recent bids produced no wins and no spend. If experience-period market references and current traffic quality support it, "
            "consider a cautious upward alpha delta, preferably focused on premium/strong traffic."
        )
    elif "repeated_upward_adjustments_with_expensive_traffic" in risk_flags:
        advice = (
            "Memory shows repeated upward adjustments while won traffic remains expensive versus the CPA constraint. "
            "Do not keep raising bids just because win references are higher or budget looked under-spent; prefer hold/down, "
            "lower lower_quartile multiplier, and only protect premium_value traffic with strong evidence."
        )
    elif "recent_cost_spike" in risk_flags and cpa_constraint > 0 and recent_expected_cpa > cpa_constraint:
        advice = (
            "Recent spend jumped and expected CPA is above constraint. Treat this as expensive traffic: avoid broad alpha "
            "increases, reduce lower-quartile exposure, and require strong premium/strong-value evidence before any upward move."
        )
    elif "repeated_downward_adjustments_without_score_gain" in risk_flags:
        advice = (
            "Repeated down adjustments have not improved expected score. Stop broad alpha suppression; prefer hold or a "
            "small rebound if pacing/quality support it, and use segment multipliers for selectivity."
        )
    elif cpa_constraint > 0 and total_conversions > 0 and recent_cpa > cpa_constraint:
        advice = (
            "Recent 5-step CPA is above constraint. Avoid broad upward adjustments; reduce or hold lower-quartile traffic unless "
            "premium/strong-value evidence is strong and budget pacing is still safe."
        )
    elif cpa_constraint > 0 and (cpa_so_far_after > cpa_constraint or expected_cpa_so_far_after > cpa_constraint) and remaining_budget_ratio_after < 0.5:
        advice = (
            "Cumulative CPA is above target and remaining budget is tightening. Prefer CPA control, smaller positive deltas, "
            "and stronger selectivity through segment multipliers."
        )
    elif total_wins > 0 and total_conversions <= 0:
        advice = (
            "Recent wins have not converted yet. Treat CPA evidence as noisy but avoid repeatedly increasing alpha from the "
            "same underspend reason without stronger premium/strong-value or pacing support."
        )
    else:
        advice = (
            "Recent outcomes are not alarming. Continue measured adjustments, compare with experience-period pacing, and avoid "
            "repeating the same direction if the latest outcome worsened CPA or budget pace."
        )

    compact_steps = [
        {
            "timeStepIndex": item.get("timeStepIndex"),
            "value_segment": item.get("value_segment"),
            "adjustment": round(as_float(item.get("adjustment")), 4),
            "final_alpha": round(as_float(item.get("final_alpha")), 6),
            "cost": round(as_float(item.get("cost")), 6),
            "win_count": int(as_float(item.get("win_count"))),
            "conversions": round(as_float(item.get("conversions")), 6),
            "remaining_budget_ratio_after": round(as_float(item.get("remaining_budget_ratio_after")), 6),
            "cpa_so_far_after": round(as_float(item.get("cpa_so_far_after")), 6),
            "expected_conversions": round(as_float(item.get("expected_conversions")), 6),
            "expected_cpa_after": round(as_float(item.get("expected_cpa_so_far_after")), 6),
            "expected_score_after": round(as_float(item.get("expected_score_so_far_after")), 6),
            "reason": str(item.get("reason", ""))[:220],
        }
        for item in recent
    ]

    summary = (
        f"last_{len(recent)}_steps: cost={total_cost:.4f}, wins={total_wins:.0f}, conversions={total_conversions:.0f}, "
        f"expected_conversions={total_expected_conversions:.4f}, recent_cpa={recent_cpa:.4f}, "
        f"recent_expected_cpa={recent_expected_cpa:.4f}, avg_adjustment={avg_adjustment:.4f}, avg_final_alpha={avg_final_alpha:.4f}, "
        f"recent_positive_adjustments={recent_positive_adjustments}, recent_negative_adjustments={recent_negative_adjustments}, "
        f"lower_quartile_steps={lower_quartile_steps}, standard_value_steps={standard_value_steps}, "
        f"strong_value_steps={strong_value_steps}, premium_value_steps={premium_value_steps}, remaining_budget_ratio_after={remaining_budget_ratio_after:.4f}, "
        f"cpa_so_far_after={cpa_so_far_after:.4f}, expected_cpa_so_far_after={expected_cpa_so_far_after:.4f}, "
        f"expected_score_change_5step={recent_expected_score_end - recent_expected_score_start:.4f}."
    )
    episode_summary = (
        f"episode_so_far: steps={len(episode)}, cost={episode_cost:.4f}, wins={episode_wins:.0f}, "
        f"conversions={episode_conversions:.0f}, expected_conversions={episode_expected_conversions:.4f}, "
        f"realized_cpa={episode_realized_cpa:.4f}, expected_cpa={episode_expected_cpa:.4f}, "
        f"avg_adjustment={episode_avg_adjustment:.4f}, positive_adjustments={episode_positive_adjustments}, "
        f"negative_adjustments={episode_negative_adjustments}, remaining_budget_ratio={remaining_budget_ratio_after:.4f}."
    )
    return {
        "last_reason": str(last.get("reason", ""))[:500],
        "recent_5_summary": summary,
        "recent_5_steps": compact_steps,
        "episode_summary": episode_summary,
        "memory_risk_flags": risk_flags,
        "followup_advice": advice,
    }


class SqlInsightTool:
    """由经验 period 流量明细构建的原始 SQLite 洞察后端。

English:
Raw SQLite insight backend built from experience-period traffic rows.

这是较精确但较慢的后端：先把多个 experience period 的 traffic CSV 合成一张
`traffic` 明细表，再在每个 timestep 做 SQL 聚合查询。现在默认使用更快的
ExperienceKnowledgeTool；该类主要保留用于构建 KB 和调试精确 SQL。"""

    CACHE_SCHEMA_VERSION = "7"
    FAST_MARKET_QUANTILES = True
    SAMPLE_REFERENCE_THRESHOLDS = {
        "advertiser": 1000,
        "category": 10000,
        "global": 0,
    }
    SQL_COLUMNS = [
        "deliveryPeriodIndex",
        "advertiserNumber",
        "advertiserCategoryIndex",
        "budget",
        "CPAConstraint",
        "timeStepIndex",
        "remainingBudget",
        "pValue",
        "bid",
        "xi",
        "cost",
        "isExposed",
        "conversionAction",
        "leastWinningCost",
    ]

    def __init__(
        self,
        traffic_csv: Any,
        sqlite_path: Path,
        chunksize: int,
        advertisers: Optional[Sequence[float]] = None,
        categories: Optional[Sequence[float]] = None,
    ):
        """为一组经验 period 创建或加载原始 SQLite 缓存。

English:
Create/load the raw experience SQLite cache for one experience-period set."""
        if isinstance(traffic_csv, (list, tuple)):
            self.traffic_csvs = [Path(path) for path in traffic_csv]
        else:
            self.traffic_csvs = [Path(traffic_csv)]
        self.traffic_csv = self.traffic_csvs[0]
        self.sqlite_path = Path(sqlite_path)
        self.chunksize = chunksize
        self.advertisers = None if advertisers is None else {float(value) for value in advertisers}
        self.categories = None if categories is None else {float(value) for value in categories}
        self._market_summary_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._distribution_stats_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._won_pvalue_stats_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._success_stats_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_database()

    def _ensure_database(self) -> None:
        """当缓存缺失或过期时，构建原始 traffic 表和索引。

English:
Build the raw `traffic` SQLite table and indexes when cache is missing/stale."""
        import pandas as pd

        should_rebuild = not self.sqlite_path.exists()
        if self.sqlite_path.exists():
            newest_source_mtime = max(path.stat().st_mtime for path in self.traffic_csvs)
            should_rebuild = self.sqlite_path.stat().st_mtime < newest_source_mtime
            if not should_rebuild:
                try:
                    with sqlite3.connect(self.sqlite_path) as conn:
                        row = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
                    should_rebuild = row is None or row[0] != self.CACHE_SCHEMA_VERSION
                except sqlite3.Error:
                    should_rebuild = True
        if not should_rebuild:
            return

        scope = "full" if self.advertisers is None else "filtered"
        print(f"[demo] Building {scope} SQLite cache from {len(self.traffic_csvs)} traffic csv(s)")
        for source_csv in self.traffic_csvs:
            print(f"[demo]   SQL source: {source_csv}")
        if self.sqlite_path.exists():
            try:
                self.sqlite_path.unlink()
            except PermissionError:
                locked_path = self.sqlite_path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.sqlite_path = self.sqlite_path.with_name(f"{self.sqlite_path.stem}_rebuild_{timestamp}.sqlite")
                print(f"[demo] Existing SQLite cache is locked, rebuilding into: {self.sqlite_path}")
                print(f"[demo] Locked cache kept in place: {locked_path}")
        source_rows_loaded = 0
        with sqlite3.connect(self.sqlite_path) as conn:
            wrote_any = False
            for source_csv in self.traffic_csvs:
                for chunk in pd.read_csv(source_csv, usecols=self.SQL_COLUMNS, chunksize=self.chunksize):
                    if self.advertisers is not None:
                        mask = chunk["advertiserNumber"].astype(float).isin(self.advertisers)
                        if self.categories:
                            mask = mask | chunk["advertiserCategoryIndex"].astype(float).isin(self.categories)
                        chunk = chunk.loc[mask]
                    if chunk.empty:
                        continue
                    source_rows_loaded += len(chunk)
                    chunk.to_sql("traffic", conn, if_exists="append" if wrote_any else "replace", index=False)
                    wrote_any = True
            if not wrote_any:
                empty_df = pd.DataFrame(columns=self.SQL_COLUMNS)
                empty_df.to_sql("traffic", conn, if_exists="replace", index=False)
            conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('source_rows_loaded', ?)",
                (str(source_rows_loaded),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (self.CACHE_SCHEMA_VERSION,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('source_files', ?)",
                ("|".join(str(path) for path in self.traffic_csvs),),
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_adv_step ON traffic(advertiserNumber, timeStepIndex)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traffic_cat_step ON traffic(advertiserCategoryIndex, timeStepIndex)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_adv_pvalue ON traffic(advertiserNumber, pValue)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_cat_pvalue ON traffic(advertiserCategoryIndex, pValue)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_step_pvalue ON traffic(timeStepIndex, pValue)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_adv_step_bid ON traffic(advertiserNumber, timeStepIndex, bid)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traffic_cat_step_bid ON traffic(advertiserCategoryIndex, timeStepIndex, bid)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_step_bid ON traffic(timeStepIndex, bid)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traffic_adv_step_lwc ON traffic(advertiserNumber, timeStepIndex, leastWinningCost)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traffic_cat_step_lwc ON traffic(advertiserCategoryIndex, timeStepIndex, leastWinningCost)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_step_lwc ON traffic(timeStepIndex, leastWinningCost)")

    def row_count(self) -> int:
        """返回原始 traffic 表当前存储的行数。

English:
Number of rows currently stored in the raw traffic table."""
        with sqlite3.connect(self.sqlite_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM traffic").fetchone()[0])

    def source_rows_loaded(self) -> int:
        """返回构建缓存时加载的源 CSV 行数。

English:
Number of source CSV rows loaded when this cache was built."""
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = 'source_rows_loaded'").fetchone()
            return int(row[0]) if row else self.row_count()

    def summarize(
        self,
        advertiser_number: float,
        category: float,
        time_step_index: int,
        current_pvalue_mean: float,
        current_pv_count: int,
        current_budget: float = 0.0,
        current_pvalue_min: float = 0.0,
        current_pvalue_p25: float = 0.0,
        current_pvalue_p50: float = 0.0,
        current_pvalue_p75: float = 0.0,
        current_pvalue_p90: float = 0.0,
        current_pvalue_max: float = 0.0,
    ) -> Dict[str, Any]:
        """返回三个分析 agent 在某个 timestep 需要的全部 SQL 证据。

English:
Return all SQL evidence needed by the three analysis agents for one timestep."""
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            market_context = {
                "advertiser": self._market_summary(
                    conn,
                    "advertiserNumber = ? AND timeStepIndex = ?",
                    (advertiser_number, time_step_index),
                ),
                "category": self._market_summary(
                    conn,
                    "advertiserCategoryIndex = ? AND timeStepIndex = ?",
                    (category, time_step_index),
                ),
                "global": self._market_summary(conn, "timeStepIndex = ?", (time_step_index,)),
            }
            if current_budget > 0:
                market_context["budget_peer"] = self._market_summary(
                    conn,
                    "timeStepIndex = ? AND budget BETWEEN ? AND ?",
                    (time_step_index, current_budget * 0.8, current_budget * 1.2),
                )
            else:
                market_context["budget_peer"] = self._market_summary(conn, "timeStepIndex = ?", (time_step_index,))

            pvalue_context = self._build_pvalue_context(
                conn,
                advertiser_number=advertiser_number,
                category=category,
                time_step_index=time_step_index,
                current_budget=current_budget,
                current_pvalue_mean=current_pvalue_mean,
                current_pv_count=current_pv_count,
                current_pvalue_min=current_pvalue_min,
                current_pvalue_p25=current_pvalue_p25,
                current_pvalue_p50=current_pvalue_p50,
                current_pvalue_p75=current_pvalue_p75,
                current_pvalue_p90=current_pvalue_p90,
                current_pvalue_max=current_pvalue_max,
            )
            success_context = self._build_success_context(
                conn,
                advertiser_number=advertiser_number,
                category=category,
                time_step_index=time_step_index,
                current_pvalue_mean=current_pvalue_mean,
                current_pvalue_p25=current_pvalue_p25,
                current_pvalue_p75=current_pvalue_p75,
                pvalue_context=pvalue_context,
            )

        summary = {
            "query_scope": "all_scopes",
            "market_context": market_context,
            "sample_reference_thresholds": self.SAMPLE_REFERENCE_THRESHOLDS,
            "current_pvalue_mean": current_pvalue_mean,
            "current_pv_count": current_pv_count,
            "current_pvalue_min": current_pvalue_min,
            "current_pvalue_p25": current_pvalue_p25,
            "current_pvalue_p50": current_pvalue_p50,
            "current_pvalue_p75": current_pvalue_p75,
            "current_pvalue_p90": current_pvalue_p90,
            "current_pvalue_max": current_pvalue_max,
            "pvalue_context": pvalue_context,
            "success_context": success_context,
        }
        return self._clean_nested(summary)

    def _market_summary(
        self,
        conn: sqlite3.Connection,
        where_clause: str,
        params: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        """按一个统计范围聚合市场价、出价、胜率、成本和预算节奏。

English:
Aggregate market price, bid, win-rate, cost and budget-pacing stats for one scope."""
        cache_key = ("market", where_clause, tuple(params))
        if cache_key in self._market_summary_cache:
            return dict(self._market_summary_cache[cache_key])
        query = """
            SELECT
                COUNT(*) AS traffic_num,
                AVG(pValue) AS avg_pvalue,
                AVG(bid) AS avg_bid,
                AVG(leastWinningCost) AS avg_market_price,
                AVG(CASE WHEN xi > 0 THEN bid ELSE NULL END) AS avg_winning_bid,
                AVG(CASE WHEN xi > 0 THEN leastWinningCost ELSE NULL END) AS avg_winning_market_price,
                AVG(CASE WHEN budget > 0 THEN (budget - remainingBudget) / budget ELSE NULL END) AS row_weighted_budget_spend_ratio,
                AVG(CASE WHEN budget > 0 THEN remainingBudget / budget ELSE NULL END) AS row_weighted_budget_left_ratio,
                AVG(xi) AS win_rate,
                AVG(CASE WHEN isExposed > 0 THEN conversionAction ELSE NULL END) AS conversion_rate,
                AVG(cost) AS avg_cost,
                SUM(cost) AS total_cost,
                SUM(conversionAction) AS conversions,
                SUM(CASE WHEN xi > 0 THEN cost ELSE 0 END) /
                    NULLIF(SUM(conversionAction), 0) AS cpa_proxy
            FROM traffic
            WHERE {where_clause}
        """
        row = conn.execute(query.format(where_clause=where_clause), params).fetchone()
        if self.FAST_MARKET_QUANTILES:
            quantiles = {
                "bid_p50": 0.0,
                "bid_p75": 0.0,
                "bid_p90": 0.0,
                "market_price_p50": 0.0,
                "market_price_p75": 0.0,
                "market_price_p90": 0.0,
                "quantile_mode": "fast_disabled",
            }
        else:
            quantiles = {
                "bid_p50": self._column_quantile(conn, "bid", where_clause, params, 0.50),
                "bid_p75": self._column_quantile(conn, "bid", where_clause, params, 0.75),
                "bid_p90": self._column_quantile(conn, "bid", where_clause, params, 0.90),
                "market_price_p50": self._column_quantile(conn, "leastWinningCost", where_clause, params, 0.50),
                "market_price_p75": self._column_quantile(conn, "leastWinningCost", where_clause, params, 0.75),
                "market_price_p90": self._column_quantile(conn, "leastWinningCost", where_clause, params, 0.90),
                "quantile_mode": "exact",
            }
        budget_pacing = self._budget_spend_summary(conn, where_clause, params)
        if row and int(row["traffic_num"] or 0) > 0:
            summary = dict(row)
            summary.update(quantiles)
            summary["budget_pacing"] = budget_pacing
            self._market_summary_cache[cache_key] = dict(summary)
            return summary
        empty_summary = {
            "traffic_num": 0,
            "avg_pvalue": 0.0,
            "avg_bid": 0.0,
            "avg_market_price": 0.0,
            "avg_winning_bid": 0.0,
            "avg_winning_market_price": 0.0,
            "row_weighted_budget_spend_ratio": 0.0,
            "row_weighted_budget_left_ratio": 0.0,
            "win_rate": 0.0,
            "conversion_rate": 0.0,
            "avg_cost": 0.0,
            "total_cost": 0.0,
            "conversions": 0.0,
            "cpa_proxy": 0.0,
        }
        empty_summary.update(quantiles)
        empty_summary["budget_pacing"] = budget_pacing
        self._market_summary_cache[cache_key] = dict(empty_summary)
        return empty_summary

    def _budget_spend_summary(
        self,
        conn: sqlite3.Connection,
        where_clause: str,
        params: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        """在某个市场范围内聚合广告主级别的预算消耗进度。

English:
Aggregate advertiser-level spend progress inside a market scope."""
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS advertiser_count,
                AVG(budget) AS avg_budget,
                AVG(remaining_budget) AS avg_remaining_budget,
                AVG(spent_so_far) AS avg_spent_so_far,
                AVG(budget_spend_ratio) AS advertiser_weighted_budget_spend_ratio,
                AVG(budget_left_ratio) AS advertiser_weighted_budget_left_ratio,
                AVG(step_cost) AS avg_timestep_cost,
                SUM(step_cost) AS total_timestep_cost,
                AVG(step_cost_to_budget_ratio) AS avg_timestep_cost_to_budget_ratio,
                AVG(cpa_constraint) AS avg_cpa_constraint
            FROM (
                SELECT
                    deliveryPeriodIndex,
                    advertiserNumber,
                    AVG(budget) AS budget,
                    AVG(remainingBudget) AS remaining_budget,
                    AVG(CPAConstraint) AS cpa_constraint,
                    AVG(CASE WHEN budget > 0 THEN (budget - remainingBudget) / budget ELSE NULL END) AS budget_spend_ratio,
                    AVG(CASE WHEN budget > 0 THEN remainingBudget / budget ELSE NULL END) AS budget_left_ratio,
                    AVG(CASE WHEN budget > 0 THEN budget - remainingBudget ELSE NULL END) AS spent_so_far,
                    SUM(cost) AS step_cost,
                    SUM(cost) / NULLIF(AVG(budget), 0) AS step_cost_to_budget_ratio
                FROM traffic
                WHERE {where_clause}
                GROUP BY deliveryPeriodIndex, advertiserNumber
            )
            """,
            params,
        ).fetchone()
        if not row or int(row["advertiser_count"] or 0) <= 0:
            return {
                "advertiser_count": 0,
                "avg_budget": 0.0,
                "avg_remaining_budget": 0.0,
                "avg_spent_so_far": 0.0,
                "advertiser_weighted_budget_spend_ratio": 0.0,
                "advertiser_weighted_budget_left_ratio": 0.0,
                "avg_timestep_cost": 0.0,
                "total_timestep_cost": 0.0,
                "avg_timestep_cost_to_budget_ratio": 0.0,
                "avg_cpa_constraint": 0.0,
            }
        return dict(row)

    @staticmethod
    def _clean_nested(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: SqlInsightTool._clean_nested(item) for key, item in value.items()}
        if isinstance(value, list):
            return [SqlInsightTool._clean_nested(item) for item in value]
        return 0.0 if value is None else value

    def _distribution_stats(self, conn: sqlite3.Connection, where_clause: str, params: Tuple[Any, ...]) -> Dict[str, Any]:
        cache_key = ("distribution", where_clause, tuple(params))
        if cache_key in self._distribution_stats_cache:
            return dict(self._distribution_stats_cache[cache_key])
        count = int(conn.execute(f"SELECT COUNT(*) FROM traffic WHERE {where_clause}", params).fetchone()[0])
        if count == 0:
            stats = {"count": 0, "avg": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0}
            self._distribution_stats_cache[cache_key] = dict(stats)
            return stats
        row = conn.execute(
            f"""
            SELECT AVG(pValue) AS avg, AVG(pValue * pValue) AS avg_square
            FROM traffic
            WHERE {where_clause}
            """,
            params,
        ).fetchone()
        avg = float(row["avg"] or 0.0)
        avg_square = float(row["avg_square"] or 0.0)
        std = math.sqrt(max(avg_square - avg * avg, 0.0))

        def quantile(q: float) -> float:
            offset = min(max(int(math.floor((count - 1) * q)), 0), count - 1)
            quantile_row = conn.execute(
                f"""
                SELECT pValue
                FROM traffic
                WHERE {where_clause}
                ORDER BY pValue
                LIMIT 1 OFFSET {offset}
                """,
                params,
            ).fetchone()
            return float(quantile_row["pValue"] or 0.0) if quantile_row else 0.0

        stats = {
            "count": count,
            "avg": avg,
            "std": std,
            "p25": quantile(0.25),
            "p50": quantile(0.50),
            "p75": quantile(0.75),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
        }
        self._distribution_stats_cache[cache_key] = dict(stats)
        return stats

    def _won_distribution_stats(self, conn: sqlite3.Connection, where_clause: str, params: Tuple[Any, ...]) -> Dict[str, Any]:
        return self._distribution_stats(conn, f"{where_clause} AND xi = 1", params)

    def _column_quantile(
        self,
        conn: sqlite3.Connection,
        column: str,
        where_clause: str,
        params: Tuple[Any, ...],
        q: float,
    ) -> float:
        count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM traffic WHERE {where_clause} AND {column} IS NOT NULL",
                params,
            ).fetchone()[0]
        )
        if count <= 0:
            return 0.0
        offset = min(max(int(math.floor((count - 1) * q)), 0), count - 1)
        row = conn.execute(
            f"""
            SELECT {column}
            FROM traffic
            WHERE {where_clause}
              AND {column} IS NOT NULL
            ORDER BY {column}
            LIMIT 1 OFFSET {offset}
            """,
            params,
        ).fetchone()
        return float(row[column] or 0.0) if row else 0.0

    def _expression_quantile(
        self,
        conn: sqlite3.Connection,
        expression: str,
        where_clause: str,
        params: Tuple[Any, ...],
        q: float,
    ) -> float:
        count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM traffic WHERE {where_clause} AND ({expression}) IS NOT NULL",
                params,
            ).fetchone()[0]
        )
        if count <= 0:
            return 0.0
        offset = min(max(int(math.floor((count - 1) * q)), 0), count - 1)
        row = conn.execute(
            f"""
            SELECT ({expression}) AS quantile_value
            FROM traffic
            WHERE {where_clause}
              AND ({expression}) IS NOT NULL
            ORDER BY quantile_value
            LIMIT 1 OFFSET {offset}
            """,
            params,
        ).fetchone()
        return float(row["quantile_value"] or 0.0) if row else 0.0

    def _scope_rank(self, current_value: float, stats: Dict[str, Any]) -> Dict[str, Any]:
        if stats["count"] <= 0:
            return {"level": "unknown", "percentile_band": "unknown", "relative_to_p75": 0.0, "relative_to_p90": 0.0}
        if current_value >= stats["p90"]:
            level = "high"
            band = ">=p90"
        elif current_value >= stats["p75"]:
            level = "above_average"
            band = "p75-p90"
        elif current_value <= stats["p50"]:
            level = "low"
            band = "<=p50"
        else:
            level = "medium"
            band = "p50-p75"
        return {
            "level": level,
            "percentile_band": band,
            "relative_to_p75": safe_div(current_value, stats["p75"], default=0.0),
            "relative_to_p90": safe_div(current_value, stats["p90"], default=0.0),
        }

    def _sample_reliability(self, scope_name: str, count: int) -> Dict[str, Any]:
        threshold = self.SAMPLE_REFERENCE_THRESHOLDS.get(scope_name, 0)
        reliable = count >= threshold
        warning = ""
        if not reliable:
            warning = f"{scope_name} sample count {count} is below recommended threshold {threshold}; reference may be weak."
        return {
            "sample_count": count,
            "recommended_min_samples": threshold,
            "is_reliable": reliable,
            "warning": warning,
        }

    def _won_pvalue_stats(
        self,
        conn: sqlite3.Connection,
        where_clause: str,
        params: Tuple[Any, ...],
        pvalue_threshold: float,
    ) -> Dict[str, Any]:
        cache_key = ("won_pvalue", where_clause, tuple(params), round(float(pvalue_threshold), 12))
        if cache_key in self._won_pvalue_stats_cache:
            return dict(self._won_pvalue_stats_cache[cache_key])
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS win_count,
                MIN(pValue) AS min_pvalue,
                AVG(pValue) AS avg_pvalue,
                AVG(CASE WHEN pValue > ? THEN 1.0 ELSE 0.0 END) AS share_above_p50
            FROM traffic
            WHERE {where_clause}
              AND xi = 1
            """,
            (pvalue_threshold,) + params,
        ).fetchone()
        win_count = int(row["win_count"] or 0) if row else 0
        min_pvalue = float(row["min_pvalue"] or 0.0) if row else 0.0
        all_above_threshold = bool(win_count > 0 and min_pvalue > pvalue_threshold)
        stats = {
            "win_count": win_count,
            "pvalue_threshold": pvalue_threshold,
            "min_pvalue": min_pvalue,
            "avg_pvalue": float(row["avg_pvalue"] or 0.0) if row else 0.0,
            "share_above_threshold": float(row["share_above_p50"] or 0.0) if row else 0.0,
            "all_won_pvalues_above_threshold": all_above_threshold,
        }
        self._won_pvalue_stats_cache[cache_key] = dict(stats)
        return stats

    def _build_pvalue_context(
        self,
        conn: sqlite3.Connection,
        advertiser_number: float,
        category: float,
        time_step_index: int,
        current_budget: float,
        current_pvalue_mean: float,
        current_pv_count: int,
        current_pvalue_min: float,
        current_pvalue_p25: float,
        current_pvalue_p50: float,
        current_pvalue_p75: float,
        current_pvalue_p90: float,
        current_pvalue_max: float,
    ) -> Dict[str, Any]:
        """为全局、类别、广告主、预算同类范围构建 pValue 分布和排名证据。

English:
Build pValue distribution/rank evidence for global/category/advertiser/budget scopes."""
        scopes = {
            "global_timestep": ("timeStepIndex = ?", (time_step_index,)),
            "advertiser_timestep": ("advertiserNumber = ? AND timeStepIndex = ?", (advertiser_number, time_step_index)),
            "category_timestep": ("advertiserCategoryIndex = ? AND timeStepIndex = ?", (category, time_step_index)),
        }
        if current_budget > 0:
            scopes["budget_peer_timestep"] = (
                "timeStepIndex = ? AND budget BETWEEN ? AND ?",
                (time_step_index, current_budget * 0.8, current_budget * 1.2),
            )
        context: Dict[str, Any] = {}
        for scope_name, (where_clause, params) in scopes.items():
            stats = self._distribution_stats(conn, where_clause, params)
            won_distribution = self._won_distribution_stats(conn, where_clause, params)
            won_stats = self._won_pvalue_stats(conn, where_clause, params, float(stats["p75"]))
            context[scope_name] = {
                "history": stats,
                "won_history": won_distribution,
                "sample_reliability": self._sample_reliability(scope_name, int(stats["count"])),
                "won_sample_reliability": self._sample_reliability(scope_name, int(won_distribution["count"])),
                "won_traffic_pvalue": won_stats,
                "current_mean_rank": self._scope_rank(current_pvalue_mean, stats),
                "current_p25_rank": self._scope_rank(current_pvalue_p25, stats),
                "current_p50_rank": self._scope_rank(current_pvalue_p50, stats),
                "current_p75_rank": self._scope_rank(current_pvalue_p75, stats),
                "current_p90_rank": self._scope_rank(current_pvalue_p90, stats),
                "current_mean_rank_vs_won": self._scope_rank(current_pvalue_mean, won_distribution),
            }

        advertiser_reference = context.get("advertiser_timestep", {}).get("won_history", {})
        if int(as_float(advertiser_reference.get("count"))) <= 0:
            advertiser_reference = context.get("category_timestep", {}).get("won_history", {})
        if int(as_float(advertiser_reference.get("count"))) <= 0:
            advertiser_reference = context.get("global_timestep", {}).get("won_history", {})
        advertiser_won_p25 = float(advertiser_reference.get("p25", 0.0))
        advertiser_won_p50 = float(advertiser_reference.get("p50", 0.0))
        advertiser_won_p75 = float(advertiser_reference.get("p75", 0.0))
        current_timestep_p25 = current_pvalue_p25
        current_timestep_p50 = current_pvalue_p50
        current_timestep_p75 = current_pvalue_p75
        effective_standard_value_threshold = min(
            threshold for threshold in (advertiser_won_p25, current_timestep_p25) if threshold > 0
        ) if advertiser_won_p25 > 0 or current_timestep_p25 > 0 else 0.0
        effective_strong_value_threshold = min(
            threshold for threshold in (advertiser_won_p50, current_timestep_p50) if threshold > 0
        ) if advertiser_won_p50 > 0 or current_timestep_p50 > 0 else 0.0
        effective_premium_value_threshold = min(
            threshold for threshold in (advertiser_won_p75, current_timestep_p75) if threshold > 0
        ) if advertiser_won_p75 > 0 or current_timestep_p75 > 0 else 0.0
        has_premium_value_traffic = bool(effective_premium_value_threshold > 0 and current_pvalue_max >= effective_premium_value_threshold)
        has_strong_value_traffic = bool(effective_strong_value_threshold > 0 and current_pvalue_max >= effective_strong_value_threshold)
        has_standard_value_traffic = bool(effective_standard_value_threshold > 0 and current_pvalue_max >= effective_standard_value_threshold)
        if has_premium_value_traffic:
            segment = "premium_value"
        elif has_strong_value_traffic:
            segment = "strong_value"
        elif has_standard_value_traffic:
            segment = "standard_value"
        else:
            segment = "lower_quartile"

        context["decision_basis"] = {
            "scope_policy": "all_scopes_provided_no_priority_fallback",
            "segment": segment,
            "segment_rule": (
                "For each traffic row: premium_value if pValue >= advertiser experience-period won-traffic p75 OR "
                "pValue >= current test-period timestep p75; otherwise strong_value if pValue >= advertiser "
                "experience-period won-traffic p50 OR pValue >= current test-period timestep p50; otherwise standard_value "
                "if pValue >= advertiser experience-period won-traffic p25 OR pValue >= current test-period timestep p25; "
                "otherwise lower_quartile."
            ),
            "advertiser_won_p25_threshold": advertiser_won_p25,
            "advertiser_won_p50_threshold": advertiser_won_p50,
            "advertiser_won_p75_threshold": advertiser_won_p75,
            "advertiser_won_sample_count": int(as_float(advertiser_reference.get("count"))),
            "current_timestep_p25_threshold": current_timestep_p25,
            "current_timestep_p50_threshold": current_timestep_p50,
            "current_timestep_p75_threshold": current_timestep_p75,
            "advertiser_overall_p25_threshold": advertiser_won_p25,
            "advertiser_overall_p50_threshold": advertiser_won_p50,
            "advertiser_overall_p75_threshold": advertiser_won_p75,
            "global_timestep_p25_threshold": current_timestep_p25,
            "global_timestep_p50_threshold": current_timestep_p50,
            "global_timestep_p75_threshold": current_timestep_p75,
            "effective_standard_value_threshold": effective_standard_value_threshold,
            "effective_strong_value_threshold": effective_strong_value_threshold,
            "effective_premium_value_threshold": effective_premium_value_threshold,
            "effective_mid_value_threshold": effective_strong_value_threshold,
            "effective_high_value_threshold": effective_premium_value_threshold,
            "current_has_lower_quartile_traffic": True,
            "current_has_standard_value_traffic": has_standard_value_traffic,
            "current_has_strong_value_traffic": has_strong_value_traffic,
            "current_has_premium_value_traffic": has_premium_value_traffic,
            "current_has_mid_value_traffic": has_strong_value_traffic,
            "current_has_high_value_traffic": has_premium_value_traffic,
            "sample_warnings": {
                scope: item["sample_reliability"]["warning"]
                for scope, item in context.items()
                if item["sample_reliability"]["warning"]
            },
            "current_pvalue_mean": current_pvalue_mean,
            "current_pvalue_min": current_pvalue_min,
            "current_pvalue_p25": current_pvalue_p25,
            "current_pvalue_p50": current_pvalue_p50,
            "current_pvalue_p75": current_pvalue_p75,
            "current_pvalue_p90": current_pvalue_p90,
            "current_pvalue_max": current_pvalue_max,
        }
        return context

    def _success_stats(
        self,
        conn: sqlite3.Connection,
        scope_name: str,
        where_clause: str,
        params: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        """总结某个范围内的历史竞胜样本和 winning-alpha 参考。

English:
Summarize historical winning rows and winning-alpha references for one scope."""
        cache_key = ("success", scope_name, where_clause, tuple(params))
        if cache_key in self._success_stats_cache:
            return dict(self._success_stats_cache[cache_key])
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS win_count,
                AVG(pValue) AS avg_pvalue,
                AVG(bid) AS avg_bid,
                AVG(leastWinningCost) AS avg_least_winning_cost,
                AVG(leastWinningCost / NULLIF(pValue, 0)) AS avg_winning_alpha,
                AVG(bid / NULLIF(pValue, 0)) AS avg_bid_alpha,
                AVG(cost) AS avg_cost,
                AVG(CASE WHEN isExposed > 0 THEN conversionAction ELSE NULL END) AS conversion_rate,
                SUM(conversionAction) AS conversions
            FROM traffic
            WHERE {where_clause}
              AND xi = 1
            """,
            params,
        ).fetchone()
        count = int(row["win_count"] or 0) if row else 0

        if count > 0:
            win_where_clause = f"{where_clause} AND xi = 1"
            p75_lwc = self._column_quantile(conn, "leastWinningCost", win_where_clause, params, 0.75)
            p90_lwc = self._column_quantile(conn, "leastWinningCost", win_where_clause, params, 0.90)
            alpha_expr = "leastWinningCost / NULLIF(pValue, 0)"
            p50_alpha = self._expression_quantile(conn, alpha_expr, win_where_clause, params, 0.50)
            p75_alpha = self._expression_quantile(conn, alpha_expr, win_where_clause, params, 0.75)
            p90_alpha = self._expression_quantile(conn, alpha_expr, win_where_clause, params, 0.90)
        else:
            p75_lwc = 0.0
            p90_lwc = 0.0
            p50_alpha = 0.0
            p75_alpha = 0.0
            p90_alpha = 0.0

        avg_lwc = float(row["avg_least_winning_cost"] or 0.0) if row else 0.0
        stats = {
            "scope": scope_name,
            "sample_filter": "experience-period rows with xi = 1 under this scope",
            "win_count": count,
            "avg_pvalue": float(row["avg_pvalue"] or 0.0) if row else 0.0,
            "avg_bid": float(row["avg_bid"] or 0.0) if row else 0.0,
            "avg_least_winning_cost": avg_lwc,
            "p75_least_winning_cost": p75_lwc,
            "p90_least_winning_cost": p90_lwc,
            "avg_winning_alpha": float(row["avg_winning_alpha"] or 0.0) if row else 0.0,
            "p50_winning_alpha": p50_alpha,
            "p75_winning_alpha": p75_alpha,
            "p90_winning_alpha": p90_alpha,
            "avg_bid_alpha": float(row["avg_bid_alpha"] or 0.0) if row else 0.0,
            "avg_cost": float(row["avg_cost"] or 0.0) if row else 0.0,
            "conversion_rate": float(row["conversion_rate"] or 0.0) if row else 0.0,
            "conversions": float(row["conversions"] or 0.0) if row else 0.0,
            "reference_alpha": float(row["avg_winning_alpha"] or 0.0) if row else 0.0,
            "confidence": "high" if count >= 1000 else ("medium" if count >= 100 else ("low" if count > 0 else "none")),
        }
        self._success_stats_cache[cache_key] = dict(stats)
        return stats

    def _build_success_context(
        self,
        conn: sqlite3.Connection,
        advertiser_number: float,
        category: float,
        time_step_index: int,
        current_pvalue_mean: float,
        current_pvalue_p25: float,
        current_pvalue_p75: float,
        pvalue_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """收集广告主、类别、全局和相似 pValue 的 winning-alpha 参考。

English:
Collect advertiser/category/global/similar-pValue winning-alpha references."""
        scopes = [
            (
                "advertiser_timestep_wins",
                "advertiserNumber = ? AND timeStepIndex = ?",
                (advertiser_number, time_step_index),
            ),
            (
                "category_timestep_wins",
                "advertiserCategoryIndex = ? AND timeStepIndex = ?",
                (category, time_step_index),
            ),
            ("global_timestep_wins", "timeStepIndex = ?", (time_step_index,)),
        ]
        pvalue_low = current_pvalue_p25 if current_pvalue_p25 > 0 else current_pvalue_mean * 0.9
        pvalue_high = current_pvalue_p75 if current_pvalue_p75 > pvalue_low else current_pvalue_mean * 1.1
        if pvalue_high <= pvalue_low:
            pvalue_low = max(current_pvalue_mean * 0.8, 0.0)
            pvalue_high = current_pvalue_mean * 1.2
        scopes.extend(
            [
                (
                    "similar_pvalue_global_timestep_wins",
                    "timeStepIndex = ? AND pValue BETWEEN ? AND ?",
                    (time_step_index, pvalue_low, pvalue_high),
                ),
                (
                    "similar_pvalue_category_timestep_wins",
                    "advertiserCategoryIndex = ? AND timeStepIndex = ? AND pValue BETWEEN ? AND ?",
                    (category, time_step_index, pvalue_low, pvalue_high),
                ),
                (
                    "similar_pvalue_advertiser_wins",
                    "advertiserNumber = ? AND pValue BETWEEN ? AND ?",
                    (advertiser_number, pvalue_low, pvalue_high),
                ),
            ]
        )
        success_stats = {
            name: self._success_stats(conn, name, where_clause, params)
            for name, where_clause, params in scopes
        }
        reference_alpha_summary = {
            name: item["avg_winning_alpha"] for name, item in success_stats.items()
        }
        return {
            "included": True,
            "win_reference_by_scope": success_stats,
            "reference_alpha_summary": reference_alpha_summary,
            "similar_pvalue_band": {
                "low": pvalue_low,
                "high": pvalue_high,
                "description": "current timestep p25-p75 pValue band when available, otherwise +/-20% around current mean",
            },
            "policy_hint": "Use experience-period winning traffic as alpha references: winning_alpha = leastWinningCost / pValue. Compare advertiser, category, global, budget-peer, and similar-pValue references.",
        }


class ExperienceKnowledgeTool:
    """预聚合后的快速经验知识库后端。

English:
Fast pre-aggregated experience knowledge backend.

该类把 raw SQLite 的 `traffic` 明细表离线聚合成多张小表：
- market 表：市场价、bid、win-rate、预算节奏。
- pValue 表：流量价值分布和胜出流量分布。
- success 表：历史竞胜 alpha 参考。

运行时每个 timestep 只做 indexed lookup，避免反复扫几十 GB 明细表。"""

    KB_SCHEMA_VERSION = "1"
    PVALUE_BUCKET_SCALE = 10000.0
    BUDGET_BUCKET_SIZE = 500.0

    def __init__(
        self,
        source_sql_tool: SqlInsightTool,
        kb_path: Path,
        force_rebuild: bool = False,
    ):
        """基于原始 SqlInsightTool 创建或加载预聚合 KB。

English:
Create/load a pre-aggregated KB from a raw SqlInsightTool source."""
        self.source_sql_tool = source_sql_tool
        self.sqlite_path = Path(kb_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._market_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._pvalue_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._success_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._ensure_database(force_rebuild=force_rebuild)

    def row_count(self) -> int:
        """暴露原始源数据行数，以复用现有校验逻辑。

English:
Expose raw source row count so existing validation code can stay shared."""
        return self.source_sql_tool.row_count()

    def source_rows_loaded(self) -> int:
        """暴露原始源数据加载行数，以复用现有校验逻辑。

English:
Expose raw source loaded row count so existing validation code can stay shared."""
        return self.source_sql_tool.source_rows_loaded()

    def _ensure_database(self, force_rebuild: bool) -> None:
        """当 KB 缺失、过期或被强制重建时构建 KB 表。

English:
Build KB tables if missing, stale, or explicitly forced."""
        should_rebuild = force_rebuild or not self.sqlite_path.exists()
        if self.sqlite_path.exists() and not should_rebuild:
            try:
                with sqlite3.connect(self.sqlite_path) as conn:
                    row = conn.execute("SELECT value FROM metadata WHERE key = 'kb_schema_version'").fetchone()
                    source_row = conn.execute("SELECT value FROM metadata WHERE key = 'source_sqlite'").fetchone()
                should_rebuild = (
                    row is None
                    or row[0] != self.KB_SCHEMA_VERSION
                    or source_row is None
                    or source_row[0] != str(self.source_sql_tool.sqlite_path)
                )
            except sqlite3.Error:
                should_rebuild = True
        if not should_rebuild:
            return

        if self.sqlite_path.exists():
            try:
                self.sqlite_path.unlink()
            except PermissionError:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.sqlite_path = self.sqlite_path.with_name(f"{self.sqlite_path.stem}_rebuild_{timestamp}.sqlite")
                print(f"[demo] Existing KB cache is locked, rebuilding into: {self.sqlite_path}", flush=True)

        print(f"[demo] Building experience knowledge base: {self.sqlite_path}", flush=True)
        build_timer = time.perf_counter()
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("ATTACH DATABASE ? AS src", (str(self.source_sql_tool.sqlite_path),))
            self._build_market_tables(conn)
            self._build_pvalue_tables(conn)
            self._build_success_tables(conn)
            conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('kb_schema_version', ?)",
                (self.KB_SCHEMA_VERSION,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('source_sqlite', ?)",
                (str(self.source_sql_tool.sqlite_path),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('source_rows_loaded', ?)",
                (str(self.source_sql_tool.source_rows_loaded()),),
            )
        print(f"[demo] Experience KB built in {time.perf_counter() - build_timer:.1f}s", flush=True)

    def _build_market_tables(self, conn: sqlite3.Connection) -> None:
        """按全局、类别、广告主、预算桶预计算市场和预算统计。

English:
Precompute market/budget statistics by global/category/advertiser/budget bucket."""
        market_select = """
            COUNT(*) AS traffic_num,
            AVG(pValue) AS avg_pvalue,
            AVG(bid) AS avg_bid,
            AVG(leastWinningCost) AS avg_market_price,
            AVG(CASE WHEN xi > 0 THEN bid ELSE NULL END) AS avg_winning_bid,
            AVG(CASE WHEN xi > 0 THEN leastWinningCost ELSE NULL END) AS avg_winning_market_price,
            AVG(CASE WHEN budget > 0 THEN (budget - remainingBudget) / budget ELSE NULL END) AS row_weighted_budget_spend_ratio,
            AVG(CASE WHEN budget > 0 THEN remainingBudget / budget ELSE NULL END) AS row_weighted_budget_left_ratio,
            AVG(xi) AS win_rate,
            AVG(CASE WHEN isExposed > 0 THEN conversionAction ELSE NULL END) AS conversion_rate,
            AVG(cost) AS avg_cost,
            SUM(cost) AS total_cost,
            SUM(conversionAction) AS conversions,
            SUM(CASE WHEN xi > 0 THEN cost ELSE 0 END) / NULLIF(SUM(conversionAction), 0) AS cpa_proxy
        """
        budget_select = """
            COUNT(*) AS advertiser_count,
            AVG(budget) AS avg_budget,
            AVG(remainingBudget) AS avg_remaining_budget,
            AVG(CASE WHEN budget > 0 THEN budget - remainingBudget ELSE NULL END) AS avg_spent_so_far,
            AVG(CASE WHEN budget > 0 THEN (budget - remainingBudget) / budget ELSE NULL END) AS advertiser_weighted_budget_spend_ratio,
            AVG(CASE WHEN budget > 0 THEN remainingBudget / budget ELSE NULL END) AS advertiser_weighted_budget_left_ratio,
            AVG(cost) AS avg_timestep_cost,
            SUM(cost) AS total_timestep_cost,
            AVG(cost / NULLIF(budget, 0)) AS avg_timestep_cost_to_budget_ratio,
            AVG(CPAConstraint) AS avg_cpa_constraint
        """
        specs = [
            ("kb_market_global", "timeStepIndex", "timeStepIndex"),
            ("kb_market_category", "advertiserCategoryIndex, timeStepIndex", "advertiserCategoryIndex, timeStepIndex"),
            ("kb_market_advertiser", "advertiserNumber, timeStepIndex", "advertiserNumber, timeStepIndex"),
            (
                "kb_market_budget",
                f"CAST(budget / {self.BUDGET_BUCKET_SIZE} AS INTEGER) AS budgetBucket, timeStepIndex",
                "budgetBucket, timeStepIndex",
            ),
        ]
        for table, keys, group_by in specs:
            print(f"[demo]   KB market table: {table}", flush=True)
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                f"""
                CREATE TABLE {table} AS
                SELECT {keys}, {market_select}, {budget_select}
                FROM src.traffic
                GROUP BY {group_by}
                """
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_market_global ON kb_market_global(timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_market_category ON kb_market_category(advertiserCategoryIndex, timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_market_advertiser ON kb_market_advertiser(advertiserNumber, timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_market_budget ON kb_market_budget(budgetBucket, timeStepIndex)")

    def _build_pvalue_tables(self, conn: sqlite3.Connection) -> None:
        """按全局、类别、广告主、预算桶预计算 pValue 分布统计。

English:
Precompute pValue distribution moments by global/category/advertiser/budget bucket."""
        pvalue_select = """
            COUNT(*) AS count,
            AVG(pValue) AS avg,
            AVG(pValue * pValue) AS avg_square,
            MIN(pValue) AS min_value,
            MAX(pValue) AS max_value,
            SUM(CASE WHEN xi > 0 THEN 1 ELSE 0 END) AS won_count,
            AVG(CASE WHEN xi > 0 THEN pValue ELSE NULL END) AS won_avg,
            AVG(CASE WHEN xi > 0 THEN pValue * pValue ELSE NULL END) AS won_avg_square,
            MIN(CASE WHEN xi > 0 THEN pValue ELSE NULL END) AS won_min_value,
            MAX(CASE WHEN xi > 0 THEN pValue ELSE NULL END) AS won_max_value
        """
        specs = [
            ("kb_pvalue_global", "timeStepIndex", "timeStepIndex"),
            ("kb_pvalue_category", "advertiserCategoryIndex, timeStepIndex", "advertiserCategoryIndex, timeStepIndex"),
            ("kb_pvalue_advertiser", "advertiserNumber, timeStepIndex", "advertiserNumber, timeStepIndex"),
            (
                "kb_pvalue_budget",
                f"CAST(budget / {self.BUDGET_BUCKET_SIZE} AS INTEGER) AS budgetBucket, timeStepIndex",
                "budgetBucket, timeStepIndex",
            ),
        ]
        for table, keys, group_by in specs:
            print(f"[demo]   KB pValue table: {table}", flush=True)
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                f"""
                CREATE TABLE {table} AS
                SELECT {keys}, {pvalue_select}
                FROM src.traffic
                GROUP BY {group_by}
                """
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_pvalue_global ON kb_pvalue_global(timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_pvalue_category ON kb_pvalue_category(advertiserCategoryIndex, timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_pvalue_advertiser ON kb_pvalue_advertiser(advertiserNumber, timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_pvalue_budget ON kb_pvalue_budget(budgetBucket, timeStepIndex)")

    def _build_success_tables(self, conn: sqlite3.Connection) -> None:
        """为常规范围和 pValue 桶范围预计算 winning-alpha 参考。

English:
Precompute winning-alpha references for normal and pValue-bucket scopes."""
        success_select = """
            COUNT(*) AS win_count,
            AVG(pValue) AS avg_pvalue,
            AVG(bid) AS avg_bid,
            AVG(leastWinningCost) AS avg_least_winning_cost,
            AVG(leastWinningCost / NULLIF(pValue, 0)) AS avg_winning_alpha,
            AVG(bid / NULLIF(pValue, 0)) AS avg_bid_alpha,
            AVG(cost) AS avg_cost,
            AVG(CASE WHEN isExposed > 0 THEN conversionAction ELSE NULL END) AS conversion_rate,
            SUM(conversionAction) AS conversions
        """
        bucket_expr = f"CAST(pValue * {self.PVALUE_BUCKET_SCALE} AS INTEGER)"
        specs = [
            ("kb_success_global", "timeStepIndex", "timeStepIndex"),
            ("kb_success_category", "advertiserCategoryIndex, timeStepIndex", "advertiserCategoryIndex, timeStepIndex"),
            ("kb_success_advertiser", "advertiserNumber, timeStepIndex", "advertiserNumber, timeStepIndex"),
            ("kb_success_pvalue_global", f"timeStepIndex, {bucket_expr} AS pvalueBucket", "timeStepIndex, pvalueBucket"),
            (
                "kb_success_pvalue_category",
                f"advertiserCategoryIndex, timeStepIndex, {bucket_expr} AS pvalueBucket",
                "advertiserCategoryIndex, timeStepIndex, pvalueBucket",
            ),
            (
                "kb_success_pvalue_advertiser",
                f"advertiserNumber, {bucket_expr} AS pvalueBucket",
                "advertiserNumber, pvalueBucket",
            ),
        ]
        for table, keys, group_by in specs:
            print(f"[demo]   KB success table: {table}", flush=True)
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                f"""
                CREATE TABLE {table} AS
                SELECT {keys}, {success_select}
                FROM src.traffic
                WHERE xi = 1
                GROUP BY {group_by}
                """
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_success_global ON kb_success_global(timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_success_category ON kb_success_category(advertiserCategoryIndex, timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_success_advertiser ON kb_success_advertiser(advertiserNumber, timeStepIndex)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_success_pvalue_global ON kb_success_pvalue_global(timeStepIndex, pvalueBucket)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_success_pvalue_category ON kb_success_pvalue_category(advertiserCategoryIndex, timeStepIndex, pvalueBucket)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_success_pvalue_advertiser ON kb_success_pvalue_advertiser(advertiserNumber, pvalueBucket)")

    def summarize(
        self,
        advertiser_number: float,
        category: float,
        time_step_index: int,
        current_pvalue_mean: float,
        current_pv_count: int,
        current_budget: float = 0.0,
        current_pvalue_min: float = 0.0,
        current_pvalue_p25: float = 0.0,
        current_pvalue_p50: float = 0.0,
        current_pvalue_p75: float = 0.0,
        current_pvalue_p90: float = 0.0,
        current_pvalue_max: float = 0.0,
    ) -> Dict[str, Any]:
        """快速 KB 查询，返回与 SqlInsightTool.summarize 相同的高层结构。

English:
Fast KB lookup that returns the same high-level shape as SqlInsightTool.summarize."""
        budget_bucket = self._budget_bucket(current_budget)
        pvalue_bucket = self._pvalue_bucket(current_pvalue_mean)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            market_context = {
                "advertiser": self._market_lookup(conn, "advertiser", advertiser_number, time_step_index),
                "category": self._market_lookup(conn, "category", category, time_step_index),
                "global": self._market_lookup(conn, "global", None, time_step_index),
                "budget_peer": self._market_lookup(conn, "budget", budget_bucket, time_step_index),
            }
            pvalue_context = self._build_pvalue_context(
                conn,
                advertiser_number=advertiser_number,
                category=category,
                time_step_index=time_step_index,
                budget_bucket=budget_bucket,
                current_pvalue_mean=current_pvalue_mean,
                current_pv_count=current_pv_count,
                current_pvalue_min=current_pvalue_min,
                current_pvalue_p25=current_pvalue_p25,
                current_pvalue_p50=current_pvalue_p50,
                current_pvalue_p75=current_pvalue_p75,
                current_pvalue_p90=current_pvalue_p90,
                current_pvalue_max=current_pvalue_max,
            )
            success_context = self._build_success_context(
                conn,
                advertiser_number=advertiser_number,
                category=category,
                time_step_index=time_step_index,
                pvalue_bucket=pvalue_bucket,
                current_pvalue_mean=current_pvalue_mean,
                current_pvalue_p25=current_pvalue_p25,
                current_pvalue_p75=current_pvalue_p75,
            )
        summary = {
            "query_scope": "experience_kb_preaggregated",
            "market_context": market_context,
            "sample_reference_thresholds": SqlInsightTool.SAMPLE_REFERENCE_THRESHOLDS,
            "current_pvalue_mean": current_pvalue_mean,
            "current_pv_count": current_pv_count,
            "current_pvalue_min": current_pvalue_min,
            "current_pvalue_p25": current_pvalue_p25,
            "current_pvalue_p50": current_pvalue_p50,
            "current_pvalue_p75": current_pvalue_p75,
            "current_pvalue_p90": current_pvalue_p90,
            "current_pvalue_max": current_pvalue_max,
            "pvalue_context": pvalue_context,
            "success_context": success_context,
            "knowledge_backend": {
                "type": "preaggregated_kb",
                "budget_bucket": budget_bucket,
                "pvalue_bucket": pvalue_bucket,
                "pvalue_bucket_scale": self.PVALUE_BUCKET_SCALE,
                "budget_bucket_size": self.BUDGET_BUCKET_SIZE,
            },
        }
        return SqlInsightTool._clean_nested(summary)

    def _budget_bucket(self, budget: float) -> int:
        """把连续预算映射到粗粒度预算同类组。

English:
Map continuous budget into a coarse peer group."""
        return int(float(budget) / self.BUDGET_BUCKET_SIZE) if budget > 0 else 0

    def _pvalue_bucket(self, pvalue: float) -> int:
        """把当前 pValue 均值映射到粗粒度价值桶，用于成功经验参考。

English:
Map current pValue mean into a coarse value bucket for success references."""
        return int(max(float(pvalue), 0.0) * self.PVALUE_BUCKET_SCALE)

    def _fetch_one(self, conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...]) -> Dict[str, Any]:
        """查询一行 SQLite 结果并转成字典；不存在时返回空字典。

English:
Fetch a single SQLite row as a dict; return empty dict when absent."""
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else {}

    def _market_lookup(
        self,
        conn: sqlite3.Connection,
        scope: str,
        key: Optional[float],
        time_step_index: int,
    ) -> Dict[str, Any]:
        """查询某个范围的预计算市场证据。

English:
Lookup precomputed market evidence for one scope."""
        cache_key = ("market", scope, key, time_step_index)
        if cache_key in self._market_cache:
            return dict(self._market_cache[cache_key])
        if scope == "global":
            row = self._fetch_one(conn, "SELECT * FROM kb_market_global WHERE timeStepIndex = ?", (time_step_index,))
        elif scope == "category":
            row = self._fetch_one(
                conn,
                "SELECT * FROM kb_market_category WHERE advertiserCategoryIndex = ? AND timeStepIndex = ?",
                (key, time_step_index),
            )
        elif scope == "advertiser":
            row = self._fetch_one(
                conn,
                "SELECT * FROM kb_market_advertiser WHERE advertiserNumber = ? AND timeStepIndex = ?",
                (key, time_step_index),
            )
        else:
            row = self._fetch_one(
                conn,
                "SELECT * FROM kb_market_budget WHERE budgetBucket = ? AND timeStepIndex = ?",
                (key, time_step_index),
            )
        summary = self._market_row_to_summary(row)
        self._market_cache[cache_key] = dict(summary)
        return summary

    def _market_row_to_summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """把一行 KB 市场统计规范化为 prompt 构造器需要的格式。

English:
Normalize one KB market row into the schema expected by prompt builders."""
        if not row:
            return {
                "traffic_num": 0,
                "avg_pvalue": 0.0,
                "avg_bid": 0.0,
                "avg_market_price": 0.0,
                "avg_winning_bid": 0.0,
                "avg_winning_market_price": 0.0,
                "row_weighted_budget_spend_ratio": 0.0,
                "row_weighted_budget_left_ratio": 0.0,
                "win_rate": 0.0,
                "conversion_rate": 0.0,
                "avg_cost": 0.0,
                "total_cost": 0.0,
                "conversions": 0.0,
                "cpa_proxy": 0.0,
                "bid_p50": 0.0,
                "bid_p75": 0.0,
                "bid_p90": 0.0,
                "market_price_p50": 0.0,
                "market_price_p75": 0.0,
                "market_price_p90": 0.0,
                "quantile_mode": "kb_aggregate_only",
                "budget_pacing": self._budget_pacing_from_row({}),
            }
        summary = dict(row)
        summary.update(
            {
                "bid_p50": as_float(row.get("avg_bid")),
                "bid_p75": as_float(row.get("avg_bid")),
                "bid_p90": as_float(row.get("avg_bid")),
                "market_price_p50": as_float(row.get("avg_market_price")),
                "market_price_p75": as_float(row.get("avg_market_price")),
                "market_price_p90": as_float(row.get("avg_market_price")),
                "quantile_mode": "kb_mean_proxy",
                "budget_pacing": self._budget_pacing_from_row(row),
            }
        )
        return summary

    def _budget_pacing_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """从 KB 市场行中提取预算节奏字段。

English:
Extract budget-pacing fields from a market KB row."""
        return {
            "advertiser_count": int(as_float(row.get("advertiser_count"))),
            "avg_budget": as_float(row.get("avg_budget")),
            "avg_remaining_budget": as_float(row.get("avg_remaining_budget")),
            "avg_spent_so_far": as_float(row.get("avg_spent_so_far")),
            "advertiser_weighted_budget_spend_ratio": as_float(row.get("advertiser_weighted_budget_spend_ratio")),
            "advertiser_weighted_budget_left_ratio": as_float(row.get("advertiser_weighted_budget_left_ratio")),
            "avg_timestep_cost": as_float(row.get("avg_timestep_cost")),
            "total_timestep_cost": as_float(row.get("total_timestep_cost")),
            "avg_timestep_cost_to_budget_ratio": as_float(row.get("avg_timestep_cost_to_budget_ratio")),
            "avg_cpa_constraint": as_float(row.get("avg_cpa_constraint")),
        }

    def _pvalue_lookup(
        self,
        conn: sqlite3.Connection,
        scope: str,
        key: Optional[float],
        time_step_index: int,
    ) -> Dict[str, Any]:
        """查询某个范围的预计算 pValue 分布统计。

English:
Lookup precomputed pValue moments for one scope."""
        cache_key = ("pvalue", scope, key, time_step_index)
        if cache_key in self._pvalue_cache:
            return dict(self._pvalue_cache[cache_key])
        if scope == "global_timestep":
            row = self._fetch_one(conn, "SELECT * FROM kb_pvalue_global WHERE timeStepIndex = ?", (time_step_index,))
        elif scope == "category_timestep":
            row = self._fetch_one(
                conn,
                "SELECT * FROM kb_pvalue_category WHERE advertiserCategoryIndex = ? AND timeStepIndex = ?",
                (key, time_step_index),
            )
        elif scope == "advertiser_timestep":
            row = self._fetch_one(
                conn,
                "SELECT * FROM kb_pvalue_advertiser WHERE advertiserNumber = ? AND timeStepIndex = ?",
                (key, time_step_index),
            )
        else:
            row = self._fetch_one(
                conn,
                "SELECT * FROM kb_pvalue_budget WHERE budgetBucket = ? AND timeStepIndex = ?",
                (key, time_step_index),
            )
        self._pvalue_cache[cache_key] = dict(row)
        return row

    def _stats_from_pvalue_row(self, row: Dict[str, Any], won: bool = False) -> Dict[str, Any]:
        """把矩统计近似转换为 pValue 分位数，作为 LLM 证据。

English:
Convert moment statistics into approximate pValue quantiles for LLM evidence."""
        prefix = "won_" if won else ""
        count_key = "won_count" if won else "count"
        count = int(as_float(row.get(count_key)))
        avg = as_float(row.get(f"{prefix}avg"))
        avg_square = as_float(row.get(f"{prefix}avg_square"))
        std = math.sqrt(max(avg_square - avg * avg, 0.0)) if count > 0 else 0.0
        min_value = as_float(row.get(f"{prefix}min_value"))
        max_value = as_float(row.get(f"{prefix}max_value"))
        if count <= 0:
            return {"count": 0, "avg": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0}
        return {
            "count": count,
            "avg": avg,
            "std": std,
            "p25": clip(avg - 0.674 * std, min_value, max_value),
            "p50": clip(avg, min_value, max_value),
            "p75": clip(avg + 0.674 * std, min_value, max_value),
            "p90": clip(avg + 1.282 * std, min_value, max_value),
            "p95": clip(avg + 1.645 * std, min_value, max_value),
        }

    def _sample_reliability(self, scope_name: str, count: int) -> Dict[str, Any]:
        """标记某个范围是否有足够历史样本可供信任。

English:
Mark whether a scope has enough historical rows to be trusted."""
        short_scope = "global" if "global" in scope_name else ("category" if "category" in scope_name else "advertiser")
        threshold = SqlInsightTool.SAMPLE_REFERENCE_THRESHOLDS.get(short_scope, 0)
        warning = "" if count >= threshold else f"{scope_name} sample count {count} is below recommended threshold {threshold}; reference may be weak."
        return {
            "sample_count": count,
            "recommended_min_samples": threshold,
            "is_reliable": count >= threshold,
            "warning": warning,
        }

    def _scope_rank(self, current_value: float, stats: Dict[str, Any]) -> Dict[str, Any]:
        """描述当前 pValue 统计量在历史分布中的相对位置。

English:
Describe where the current pValue statistic falls versus historical distribution."""
        if stats["count"] <= 0:
            return {"level": "unknown", "percentile_band": "unknown", "relative_to_p75": 0.0, "relative_to_p90": 0.0}
        if current_value >= stats["p90"]:
            level = "high"
            band = ">=p90"
        elif current_value >= stats["p75"]:
            level = "above_average"
            band = "p75-p90"
        elif current_value <= stats["p50"]:
            level = "low"
            band = "<=p50"
        else:
            level = "medium"
            band = "p50-p75"
        return {
            "level": level,
            "percentile_band": band,
            "relative_to_p75": safe_div(current_value, stats["p75"], 0.0),
            "relative_to_p90": safe_div(current_value, stats["p90"], 0.0),
        }

    def _build_pvalue_context(
        self,
        conn: sqlite3.Connection,
        advertiser_number: float,
        category: float,
        time_step_index: int,
        budget_bucket: int,
        current_pvalue_mean: float,
        current_pv_count: int,
        current_pvalue_min: float,
        current_pvalue_p25: float,
        current_pvalue_p50: float,
        current_pvalue_p75: float,
        current_pvalue_p90: float,
        current_pvalue_max: float,
    ) -> Dict[str, Any]:
        """基于 KB 构建 pValue 排名和价值分层阈值证据。

English:
Build KB-backed pValue rank and segment-threshold evidence."""
        scope_specs = {
            "global_timestep": ("global_timestep", None),
            "category_timestep": ("category_timestep", category),
            "advertiser_timestep": ("advertiser_timestep", advertiser_number),
            "budget_peer_timestep": ("budget_peer_timestep", budget_bucket),
        }
        context: Dict[str, Any] = {}
        for output_scope, (lookup_scope, key) in scope_specs.items():
            row = self._pvalue_lookup(conn, lookup_scope, key, time_step_index)
            stats = self._stats_from_pvalue_row(row)
            won_distribution = self._stats_from_pvalue_row(row, won=True)
            context[output_scope] = {
                "history": stats,
                "won_history": won_distribution,
                "sample_reliability": self._sample_reliability(output_scope, int(stats["count"])),
                "won_sample_reliability": self._sample_reliability(output_scope, int(won_distribution["count"])),
                "won_traffic_pvalue": {
                    "win_count": int(won_distribution["count"]),
                    "pvalue_threshold": stats["p75"],
                    "min_pvalue": won_distribution["p25"],
                    "avg_pvalue": won_distribution["avg"],
                    "share_above_threshold": 0.0,
                    "all_won_pvalues_above_threshold": False,
                },
                "current_mean_rank": self._scope_rank(current_pvalue_mean, stats),
                "current_p25_rank": self._scope_rank(current_pvalue_p25, stats),
                "current_p50_rank": self._scope_rank(current_pvalue_p50, stats),
                "current_p75_rank": self._scope_rank(current_pvalue_p75, stats),
                "current_p90_rank": self._scope_rank(current_pvalue_p90, stats),
                "current_mean_rank_vs_won": self._scope_rank(current_pvalue_mean, won_distribution),
            }

        advertiser_reference = context.get("advertiser_timestep", {}).get("won_history", {})
        if int(as_float(advertiser_reference.get("count"))) <= 0:
            advertiser_reference = context.get("category_timestep", {}).get("won_history", {})
        if int(as_float(advertiser_reference.get("count"))) <= 0:
            advertiser_reference = context.get("global_timestep", {}).get("won_history", {})
        advertiser_won_p25 = as_float(advertiser_reference.get("p25"))
        advertiser_won_p50 = as_float(advertiser_reference.get("p50"))
        advertiser_won_p75 = as_float(advertiser_reference.get("p75"))
        effective_standard_value_threshold = min(
            threshold for threshold in (advertiser_won_p25, current_pvalue_p25) if threshold > 0
        ) if advertiser_won_p25 > 0 or current_pvalue_p25 > 0 else 0.0
        effective_strong_value_threshold = min(
            threshold for threshold in (advertiser_won_p50, current_pvalue_p50) if threshold > 0
        ) if advertiser_won_p50 > 0 or current_pvalue_p50 > 0 else 0.0
        effective_premium_value_threshold = min(
            threshold for threshold in (advertiser_won_p75, current_pvalue_p75) if threshold > 0
        ) if advertiser_won_p75 > 0 or current_pvalue_p75 > 0 else 0.0
        has_premium_value_traffic = bool(effective_premium_value_threshold > 0 and current_pvalue_max >= effective_premium_value_threshold)
        has_strong_value_traffic = bool(effective_strong_value_threshold > 0 and current_pvalue_max >= effective_strong_value_threshold)
        has_standard_value_traffic = bool(effective_standard_value_threshold > 0 and current_pvalue_max >= effective_standard_value_threshold)
        segment = "premium_value" if has_premium_value_traffic else ("strong_value" if has_strong_value_traffic else ("standard_value" if has_standard_value_traffic else "lower_quartile"))
        context["decision_basis"] = {
            "scope_policy": "preaggregated_kb_all_scopes",
            "segment": segment,
            "segment_rule": "KB uses preaggregated experience-period pValue moments; thresholds are mean/std approximations plus current timestep quantiles.",
            "advertiser_won_p25_threshold": advertiser_won_p25,
            "advertiser_won_p50_threshold": advertiser_won_p50,
            "advertiser_won_p75_threshold": advertiser_won_p75,
            "advertiser_won_sample_count": int(as_float(advertiser_reference.get("count"))),
            "current_timestep_p25_threshold": current_pvalue_p25,
            "current_timestep_p50_threshold": current_pvalue_p50,
            "current_timestep_p75_threshold": current_pvalue_p75,
            "effective_standard_value_threshold": effective_standard_value_threshold,
            "effective_strong_value_threshold": effective_strong_value_threshold,
            "effective_premium_value_threshold": effective_premium_value_threshold,
            "effective_mid_value_threshold": effective_strong_value_threshold,
            "effective_high_value_threshold": effective_premium_value_threshold,
            "current_has_lower_quartile_traffic": True,
            "current_has_standard_value_traffic": has_standard_value_traffic,
            "current_has_strong_value_traffic": has_strong_value_traffic,
            "current_has_premium_value_traffic": has_premium_value_traffic,
            "current_has_mid_value_traffic": has_strong_value_traffic,
            "current_has_high_value_traffic": has_premium_value_traffic,
            "sample_warnings": {
                scope: item["sample_reliability"]["warning"]
                for scope, item in context.items()
                if isinstance(item, dict) and item.get("sample_reliability", {}).get("warning")
            },
            "current_pvalue_mean": current_pvalue_mean,
            "current_pvalue_min": current_pvalue_min,
            "current_pvalue_p25": current_pvalue_p25,
            "current_pvalue_p50": current_pvalue_p50,
            "current_pvalue_p75": current_pvalue_p75,
            "current_pvalue_p90": current_pvalue_p90,
            "current_pvalue_max": current_pvalue_max,
            "kb_note": "Quantiles are moment-based approximations to avoid online scans over the raw traffic table.",
        }
        return context

    def _success_lookup(
        self,
        conn: sqlite3.Connection,
        scope_name: str,
        table: str,
        where_sql: str,
        params: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        """查询并合并某个范围的预计算成功经验行。

English:
Lookup and merge precomputed success rows for one scope."""
        cache_key = ("success", scope_name, table, params)
        if cache_key in self._success_cache:
            return dict(self._success_cache[cache_key])
        row = self._fetch_one(
            conn,
            f"""
            SELECT
                SUM(win_count) AS win_count,
                SUM(avg_pvalue * win_count) / NULLIF(SUM(win_count), 0) AS avg_pvalue,
                SUM(avg_bid * win_count) / NULLIF(SUM(win_count), 0) AS avg_bid,
                SUM(avg_least_winning_cost * win_count) / NULLIF(SUM(win_count), 0) AS avg_least_winning_cost,
                SUM(avg_winning_alpha * win_count) / NULLIF(SUM(win_count), 0) AS avg_winning_alpha,
                SUM(avg_bid_alpha * win_count) / NULLIF(SUM(win_count), 0) AS avg_bid_alpha,
                SUM(avg_cost * win_count) / NULLIF(SUM(win_count), 0) AS avg_cost,
                SUM(conversion_rate * win_count) / NULLIF(SUM(win_count), 0) AS conversion_rate,
                SUM(conversions) AS conversions
            FROM {table}
            WHERE {where_sql}
            """,
            params,
        )
        stats = self._success_row_to_stats(scope_name, row)
        self._success_cache[cache_key] = dict(stats)
        return stats

    def _success_row_to_stats(self, scope_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """把一行 KB 成功经验规范化为 winning-reference 格式。

English:
Normalize a KB success row into the winning-reference schema."""
        count = int(as_float(row.get("win_count")))
        avg_lwc = as_float(row.get("avg_least_winning_cost"))
        avg_alpha = as_float(row.get("avg_winning_alpha"))
        confidence = "high" if count >= 1000 else ("medium" if count >= 100 else ("low" if count > 0 else "none"))
        return {
            "scope": scope_name,
            "sample_filter": "preaggregated experience-period rows with xi = 1 under this scope",
            "win_count": count,
            "avg_pvalue": as_float(row.get("avg_pvalue")),
            "avg_bid": as_float(row.get("avg_bid")),
            "avg_least_winning_cost": avg_lwc,
            "p75_least_winning_cost": avg_lwc * 1.15 if avg_lwc > 0 else 0.0,
            "p90_least_winning_cost": avg_lwc * 1.35 if avg_lwc > 0 else 0.0,
            "avg_winning_alpha": avg_alpha,
            "p50_winning_alpha": avg_alpha,
            "p75_winning_alpha": avg_alpha * 1.15 if avg_alpha > 0 else 0.0,
            "p90_winning_alpha": avg_alpha * 1.35 if avg_alpha > 0 else 0.0,
            "avg_bid_alpha": as_float(row.get("avg_bid_alpha")),
            "avg_cost": as_float(row.get("avg_cost")),
            "conversion_rate": as_float(row.get("conversion_rate")),
            "conversions": as_float(row.get("conversions")),
            "reference_alpha": avg_alpha,
            "confidence": confidence,
            "quantile_mode": "kb_mean_proxy",
        }

    def _build_success_context(
        self,
        conn: sqlite3.Connection,
        advertiser_number: float,
        category: float,
        time_step_index: int,
        pvalue_bucket: int,
        current_pvalue_mean: float,
        current_pvalue_p25: float,
        current_pvalue_p75: float,
    ) -> Dict[str, Any]:
        """构建 SuccessExperienceAgent 需要的全部 winning-alpha 参考。

English:
Build all winning-alpha references consumed by SuccessExperienceAgent."""
        success_stats = {
            "advertiser_timestep_wins": self._success_lookup(
                conn,
                "advertiser_timestep_wins",
                "kb_success_advertiser",
                "advertiserNumber = ? AND timeStepIndex = ?",
                (advertiser_number, time_step_index),
            ),
            "category_timestep_wins": self._success_lookup(
                conn,
                "category_timestep_wins",
                "kb_success_category",
                "advertiserCategoryIndex = ? AND timeStepIndex = ?",
                (category, time_step_index),
            ),
            "global_timestep_wins": self._success_lookup(
                conn,
                "global_timestep_wins",
                "kb_success_global",
                "timeStepIndex = ?",
                (time_step_index,),
            ),
            "similar_pvalue_global_timestep_wins": self._success_lookup(
                conn,
                "similar_pvalue_global_timestep_wins",
                "kb_success_pvalue_global",
                "timeStepIndex = ? AND pvalueBucket BETWEEN ? AND ?",
                (time_step_index, max(pvalue_bucket - 1, 0), pvalue_bucket + 1),
            ),
            "similar_pvalue_category_timestep_wins": self._success_lookup(
                conn,
                "similar_pvalue_category_timestep_wins",
                "kb_success_pvalue_category",
                "advertiserCategoryIndex = ? AND timeStepIndex = ? AND pvalueBucket BETWEEN ? AND ?",
                (category, time_step_index, max(pvalue_bucket - 1, 0), pvalue_bucket + 1),
            ),
            "similar_pvalue_advertiser_wins": self._success_lookup(
                conn,
                "similar_pvalue_advertiser_wins",
                "kb_success_pvalue_advertiser",
                "advertiserNumber = ? AND pvalueBucket BETWEEN ? AND ?",
                (advertiser_number, max(pvalue_bucket - 1, 0), pvalue_bucket + 1),
            ),
        }
        reference_alpha_summary = {name: item["avg_winning_alpha"] for name, item in success_stats.items()}
        return {
            "included": True,
            "win_reference_by_scope": success_stats,
            "reference_alpha_summary": reference_alpha_summary,
            "similar_pvalue_band": {
                "low": current_pvalue_p25,
                "high": current_pvalue_p75 if current_pvalue_p75 > current_pvalue_p25 else current_pvalue_mean,
                "bucket": pvalue_bucket,
                "description": "KB lookup uses fixed pValue bucket around current mean instead of scanning a dynamic p25-p75 range.",
            },
            "policy_hint": "Use preaggregated experience-period winning traffic as alpha references: winning_alpha = leastWinningCost / pValue. Quantile fields are mean proxies in KB mode.",
        }


@dataclass
class AdjustmentDecision:
    """最终 LLM 策略 agent 返回的结构化决策。

English:
Structured decision returned by the final LLM strategy agent."""

    direction: str
    adjustment_range: str
    adjustment: float
    reason: str
    pvalue_multipliers: Dict[str, float]
    source: str = "llm"


class AdjustmentAnalysisAgent:
    """多 LLM 调价控制器。

English:
Multi-LLM adjustment controller.

该类不直接执行竞价，而是让三个分析 agent 并行产出结构化判断，
再让 FinalStrategyAgent 汇总为一个 AdjustmentDecision。"""

    MULTIPLIER_KEYS = ["lower_quartile", "standard_value", "strong_value", "premium_value"]
    MAX_REPAIR_ATTEMPTS = 2

    def __init__(self, llm_client: Optional[Any] = None, print_agent_outputs: bool = True):
        """保存 LLM 客户端和终端输出偏好。

English:
Store the LLM client and console-output preference."""
        if llm_client is None:
            raise RuntimeError("AdjustmentAnalysisAgent now requires an LLM client; offline decision logic is disabled.")
        self.llm_client = llm_client
        self.print_agent_outputs = print_agent_outputs

    def _print_agent_output(self, label: str, payload: Dict[str, Any]) -> None:
        """按需把每个 agent 的 JSON 输出打印到终端，便于调试。

English:
Optionally print each agent's JSON output to terminal for debugging."""
        if not self.print_agent_outputs:
            return
        print(
            "[agent_output] "
            + json.dumps(
                {
                    "agent": label,
                    "output": payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    def decide(
        self,
        base_alpha: float,
        sql_summary: Dict[str, Any],
        time_step_index: int,
        budget: float,
        cpa: float,
        remaining_budget: float,
        history: Dict[str, list],
        current_state: Optional[Dict[str, float]] = None,
        decision_memory: Optional[Dict[str, Any]] = None,
    ) -> AdjustmentDecision:
        """公开入口：为当前 timestep 生成一次调价决策。

English:
Public entry: produce one adjustment decision for the current timestep."""
        return self._llm_decide(
            base_alpha=base_alpha,
            sql_summary=sql_summary,
            time_step_index=time_step_index,
            budget=budget,
            cpa=cpa,
            remaining_budget=remaining_budget,
            history=history,
            current_state=current_state,
            decision_memory=decision_memory,
        )

    def _llm_decide(
        self,
        base_alpha: float,
        sql_summary: Dict[str, Any],
        time_step_index: int,
        budget: float,
        cpa: float,
        remaining_budget: float,
        history: Dict[str, list],
        current_state: Optional[Dict[str, float]],
        decision_memory: Optional[Dict[str, Any]],
    ) -> AdjustmentDecision:
        """把 LLM 模式路由到当前多 agent 实现。

English:
Route LLM mode to the current multi-agent implementation."""
        return self._multi_agent_decide(
            base_alpha=base_alpha,
            sql_summary=sql_summary,
            time_step_index=time_step_index,
            budget=budget,
            cpa=cpa,
            remaining_budget=remaining_budget,
            history=history,
            current_state=current_state,
            decision_memory=decision_memory,
        )

    def _ask_analysis_json(self, agent_name: str, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """请求单个分析 agent 输出 JSON；失败时返回安全兜底结果。

English:
Ask one analysis agent for JSON and return a safe fallback on failure."""
        try:
            data = self.llm_client.ask_json(prompt)
            if not isinstance(data, dict):
                raise ValueError(f"{agent_name} response is not a JSON object")
            data.setdefault("agent", agent_name)
            return data
        except Exception as exc:
            return {
                "agent": agent_name,
                "status": "fallback",
                "analysis_conclusion": "LLM analysis failed; downstream final strategy should rely on other agents and guardrails.",
                "core_evidence": [f"{type(exc).__name__}: {exc}"],
                "strategy_advice": fallback.get("strategy_advice", "Use conservative hold unless other evidence is strong."),
            }

    def _multi_agent_decide(
        self,
        base_alpha: float,
        sql_summary: Dict[str, Any],
        time_step_index: int,
        budget: float,
        cpa: float,
        remaining_budget: float,
        history: Dict[str, list],
        current_state: Optional[Dict[str, float]],
        decision_memory: Optional[Dict[str, Any]],
    ) -> AdjustmentDecision:
        """并行运行三个分析 agent，并用 FinalStrategyAgent 合并结果。

English:
Run three analysis agents in parallel and merge them with FinalStrategyAgent."""
        import numpy as np

        step_timer = time.perf_counter()
        spent = budget - remaining_budget
        historical_conversions = float(
            sum(np.sum(result[:, 1]) for result in history["historyImpressionResult"])
        )
        phase_policy = delivery_phase_policy(time_step_index)
        success_refs = success_reference_by_scope(sql_summary)
        score_status = (current_state or {}).get("score_status", {})
        decision_context = build_decision_brief(
            base_alpha=base_alpha,
            sql_summary=sql_summary,
            success_refs=success_refs,
            score_status=score_status,
            decision_memory=decision_memory or {},
            budget=budget,
            remaining_budget=remaining_budget,
            time_step_index=time_step_index,
        )
        gave_state_features = dict(current_state or {})
        gave_state_features.pop("score_status", None)
        gave_state_features.pop("score_optimization_hint", None)

        shared_status = {
            "base_alpha": base_alpha,
            "base_bid_formula": "bid = base_alpha * pValue",
            "timeStepIndex": time_step_index,
            "budget": budget,
            "remaining_budget": remaining_budget,
            "budget_left_ratio": safe_div(remaining_budget, budget, 0.0),
            "cpa_constraint": cpa,
            "spent_so_far": spent,
            "conversions_so_far": historical_conversions,
            "realized_cpa_so_far": safe_div(spent, historical_conversions, 0.0),
            "phase_policy": phase_policy,
            "gave_state_features": gave_state_features,
        }

        market_context = {
            "shared_status": shared_status,
            "market_context_by_scope": {
                key: sql_summary.get("market_context", {}).get(key, {})
                for key in ("global", "category", "advertiser", "budget_peer")
            },
            "current_pvalue_distribution": {
                "mean": sql_summary.get("current_pvalue_mean"),
                "min": sql_summary.get("current_pvalue_min"),
                "p25": sql_summary.get("current_pvalue_p25"),
                "p50": sql_summary.get("current_pvalue_p50"),
                "p75": sql_summary.get("current_pvalue_p75"),
                "p90": sql_summary.get("current_pvalue_p90"),
                "max": sql_summary.get("current_pvalue_max"),
                "count": sql_summary.get("current_pv_count"),
            },
            "pvalue_context": {
                key: sql_summary.get("pvalue_context", {}).get(key, {})
                for key in (
                    "global_timestep",
                    "category_timestep",
                    "advertiser_timestep",
                    "budget_peer_timestep",
                    "decision_basis",
                )
            },
            "market_position_summary": decision_context["market_position_summary"],
            "traffic_mix_summary": decision_context["traffic_mix_summary"],
        }

        budget_context = {
            "shared_status": shared_status,
            "budget_pacing_summary": decision_context["budget_pacing_summary"],
            "score_and_cpa_summary": decision_context["score_and_cpa_summary"],
            "memory_diagnosis": decision_context["memory_diagnosis"],
            "market_budget_pacing_by_scope": {
                key: sql_summary.get("market_context", {}).get(key, {}).get("budget_pacing", {})
                for key in ("global", "category", "advertiser", "budget_peer")
            },
            "score_status": score_status,
        }

        success_context = {
            "shared_status": shared_status,
            "success_context": sql_summary.get("success_context", {}),
            "success_reference_by_scope": success_refs,
            "market_position_summary": decision_context["market_position_summary"],
            "traffic_mix_summary": decision_context["traffic_mix_summary"],
        }

        market_prompt = f"""
# ROLE
You are MarketValueAgent. Analyze traffic value and historical market price only. Do not decide final adjustment bounds.

# INPUT
{json.dumps(market_context, ensure_ascii=False, sort_keys=True)}

# TASK
Compare current traffic value and base_alpha against four historical levels: global, category, advertiser, and budget_peer at the same timestep. Then analyze the four pValue segments. If value is high and base_alpha is below historical winning/market alpha, favor upward direction. If value is low and base_alpha is high, favor downward direction.

Return strict JSON:
{{
  "agent": "market_value",
  "direction_signal": "up|down|hold|mixed",
  "confidence": "low|medium|high",
  "market_position": "underbid|overbid|near_market|mixed",
  "traffic_value_status": "high|medium|low|mixed",
  "segment_suggestions": {{
    "lower_quartile": "up|down|hold",
    "standard_value": "up|down|hold",
    "strong_value": "up|down|hold",
    "premium_value": "up|down|hold"
  }},
  "core_evidence": ["short numeric evidence"],
  "strategy_advice": "clear advice, no final adjustment number"
}}
"""

        budget_prompt = f"""
# ROLE
You are BudgetControlAgent. Analyze pacing, CPA, score risk, and memory only.

# INPUT
{json.dumps(budget_context, ensure_ascii=False, sort_keys=True)}

# TASK
Judge whether the advertiser is spending too fast or too slow, whether CPA/score risk requires conservative control, and what adjustment direction constraints should apply. Use memory only as diagnostic evidence.

Return strict JSON:
{{
  "agent": "budget_control",
  "risk_level": "low|medium|high",
  "pacing_status": "ahead|on_track|behind|severely_behind|unknown",
  "direction_constraint": "prefer_up|prefer_down|prefer_hold|allow_both",
  "suggested_adjustment_bounds": {{"lower": -1.0, "upper": 1.0}},
  "core_evidence": ["short numeric evidence"],
  "strategy_advice": "clear pacing and CPA advice"
}}
"""

        success_prompt = f"""
# ROLE
You are SuccessExperienceAgent. Analyze winning experience and feasible alpha for winning auctions.

# INPUT
{json.dumps(success_context, ensure_ascii=False, sort_keys=True)}

# TASK
Estimate what alpha can win current-like traffic. Compare advertiser historical successful alpha, same timestep global/category successful alpha, and similar-pValue successful alpha. Explain whether base_alpha is below, near, or above feasible winning references.

Return strict JSON:
{{
  "agent": "success_experience",
  "winning_feasibility": "easy|moderate|hard|too_expensive|unknown",
  "alpha_gap": "base_alpha_below_win_reference|near_reference|above_reference|mixed",
  "win_alpha_reference": {{
    "advertiser_typical": 0.0,
    "category_timestep": 0.0,
    "global_timestep": 0.0,
    "similar_pvalue": 0.0
  }},
  "core_evidence": ["short numeric evidence"],
  "strategy_advice": "clear winning-alpha advice, no final adjustment number"
}}
"""

        analysis_timer = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            market_future = executor.submit(
                self._ask_analysis_json,
                "market_value",
                market_prompt,
                {"strategy_advice": "Use market references conservatively."},
            )
            budget_future = executor.submit(
                self._ask_analysis_json,
                "budget_control",
                budget_prompt,
                {"strategy_advice": "Prefer hold if pacing and CPA evidence is unclear."},
            )
            success_future = executor.submit(
                self._ask_analysis_json,
                "success_experience",
                success_prompt,
                {"strategy_advice": "Do not chase winning alpha without CPA and pacing support."},
            )
            market_analysis = market_future.result()
            budget_analysis = budget_future.result()
            success_analysis = success_future.result()
        analysis_elapsed = time.perf_counter() - analysis_timer

        self._print_agent_output("MarketValueAgent", market_analysis)
        self._print_agent_output("BudgetControlAgent", budget_analysis)
        self._print_agent_output("SuccessExperienceAgent", success_analysis)

        final_context = {
            "shared_status": shared_status,
            "market_value_analysis": market_analysis,
            "budget_control_analysis": budget_analysis,
            "success_experience_analysis": success_analysis,
            "decision_memory": decision_memory or {},
            "traffic_mix_summary": decision_context["traffic_mix_summary"],
            "score_and_cpa_summary": decision_context["score_and_cpa_summary"],
            "guardrail_notice": {
                "global_adjustment_range": [MIN_ADJUSTMENT, MAX_ADJUSTMENT],
                "normal_changes": "usually keep additive alpha adjustment in [-1, 1]",
                "large_positive_allowed_when": "severe underbid or low-spend recovery with safe CPA and reliable references",
                "large_negative_allowed_when": "CPA/budget risk plus overbid evidence",
                "segment_multiplier_range": [0.95, 1.05],
            },
        }

        final_prompt = f"""
# ROLE
You are FinalStrategyAgent. Integrate three parallel expert analyses and choose the final bidding strategy.

# INPUT
{json.dumps(final_context, ensure_ascii=False, sort_keys=True)}

# TASK
Choose one additive alpha adjustment and pValue segment multipliers. Balance market value, budget/CPA control, and winning-alpha experience. Keep explanation short and mention which agents drove the decision.

Return strict JSON only:
{{
  "direction": "up|down|hold",
  "adjustment_range": "one of {ADJUSTMENT_RANGES}",
  "adjustment": 0.0,
  "pvalue_multipliers": {{
    "lower_quartile": 1.0,
    "standard_value": 1.0,
    "strong_value": 1.0,
    "premium_value": 1.0
  }},
  "agent_consensus": {{
    "market_value": "up|down|hold|mixed",
    "budget_control": "prefer_up|prefer_down|prefer_hold|allow_both",
    "success_experience": "up|down|hold|mixed"
  }},
  "reason": "concise attribution in <= 70 words"
}}
"""

        repair_note = ""
        last_error = ""
        final_timer = time.perf_counter()
        for attempt in range(self.MAX_REPAIR_ATTEMPTS + 1):
            try:
                data = self.llm_client.ask_json(final_prompt + repair_note)
                raw_adjustment = float(data["adjustment"])
                if not math.isfinite(raw_adjustment):
                    raise ValueError(f"adjustment must be finite, got {raw_adjustment!r}")
                if raw_adjustment < MIN_ADJUSTMENT or raw_adjustment > MAX_ADJUSTMENT:
                    raise ValueError(
                        f"adjustment {raw_adjustment} is outside global range [{MIN_ADJUSTMENT}, {MAX_ADJUSTMENT}]"
                    )
                raw_adjustment = round(raw_adjustment, 4)
                adjustment, guardrail = contextual_adjustment_guardrail(
                    raw_adjustment=raw_adjustment,
                    base_alpha=base_alpha,
                    sql_summary=sql_summary,
                    score_status=score_status,
                    decision_memory=decision_memory or {},
                    budget=budget,
                    remaining_budget=remaining_budget,
                    time_step_index=time_step_index,
                )
                direction = str(data.get("direction", "")).lower()
                if direction not in {"up", "down", "hold"} or guardrail["applied"]:
                    direction = "hold" if abs(adjustment) < 0.03 else ("up" if adjustment > 0 else "down")
                adjustment_range = str(data.get("adjustment_range") or adjustment_to_range(adjustment))
                if guardrail["applied"] or adjustment_range not in ADJUSTMENT_RANGES:
                    adjustment_range = adjustment_to_range(adjustment)
                multipliers = self._normalize_multipliers(data.get("pvalue_multipliers"), sql_summary)
                consensus = data.get("agent_consensus", {})
                self._print_agent_output(
                    "FinalStrategyAgent",
                    {
                        "direction": data.get("direction"),
                        "adjustment": raw_adjustment,
                        "adjustment_range": data.get("adjustment_range"),
                        "pvalue_multipliers": data.get("pvalue_multipliers"),
                        "agent_consensus": consensus,
                        "reason": data.get("reason"),
                    },
                )
                reason = str(data.get("reason", "FinalStrategyAgent integrated three analyses."))[:300]
                reason = (
                    f"{reason} Consensus={json.dumps(consensus, ensure_ascii=False, sort_keys=True)}"
                )[:360]
                if guardrail["applied"]:
                    reason = (
                        reason
                        + f" Guardrail {guardrail['mode']}: raw {guardrail['raw_adjustment']:.2f} -> {guardrail['bounded_adjustment']:.2f}."
                    )[:360]
                source = "multi_llm+guardrail" if guardrail["applied"] else "multi_llm"
                if attempt > 0:
                    source += "+repair"
                print(
                    "[timing] llm_agents "
                    f"timestep={time_step_index} "
                    f"analysis_parallel={analysis_elapsed:.2f}s "
                    f"final_llm={time.perf_counter() - final_timer:.2f}s "
                    f"total_decide={time.perf_counter() - step_timer:.2f}s",
                    flush=True,
                )
                return AdjustmentDecision(direction, adjustment_range, adjustment, reason, multipliers, source)
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.MAX_REPAIR_ATTEMPTS:
                    break
                repair_note = f"""

# PREVIOUS FINAL OUTPUT WAS INVALID - REPAIR REQUIRED
Your previous JSON could not be used: {last_error}
Return fresh strict JSON. adjustment must be finite and in [{MIN_ADJUSTMENT}, {MAX_ADJUSTMENT}]. Multipliers must be in [0.95, 1.05].
"""

        fallback_multipliers = self._default_multipliers(sql_summary)
        return AdjustmentDecision(
            "hold",
            adjustment_to_range(0.0),
            0.0,
            f"Final multi-agent LLM decision invalid after repair; safe hold fallback. Last error: {last_error}"[:360],
            fallback_multipliers,
            "multi_llm_repair_fallback",
        )

        import numpy as np

        spent = budget - remaining_budget
        historical_conversions = float(
            sum(np.sum(result[:, 1]) for result in history["historyImpressionResult"])
        )
        phase_policy = delivery_phase_policy(time_step_index)
        success_refs = success_reference_by_scope(sql_summary)
        score_status = (current_state or {}).get("score_status", {})
        decision_context = build_decision_brief(
            base_alpha=base_alpha,
            sql_summary=sql_summary,
            success_refs=success_refs,
            score_status=score_status,
            decision_memory=decision_memory or {},
            budget=budget,
            remaining_budget=remaining_budget,
            time_step_index=time_step_index,
        )
        period7_evidence = build_compact_period7_evidence(sql_summary, base_alpha)
        gave_state_features = dict(current_state or {})
        gave_state_features.pop("score_status", None)
        gave_state_features.pop("score_optimization_hint", None)
        prompt = f"""
# YOUR ROLE
You are an expert auto-bidding controller for RTB advertising. First judge market position and delivery opportunity, then judge CPA/score risk. Output one additive alpha adjustment, pValue segment multipliers, and one concise reason.

# PRIORITIZED DECISION BRIEF
{json.dumps(decision_context["decision_brief"], ensure_ascii=False, sort_keys=True)}

# MARKET POSITION SUMMARY - PRIMARY EVIDENCE
{json.dumps(decision_context["market_position_summary"], ensure_ascii=False, sort_keys=True)}

# BUDGET PACING SUMMARY
{json.dumps(decision_context["budget_pacing_summary"], ensure_ascii=False, sort_keys=True)}

# SCORE AND CPA SUMMARY
{json.dumps(decision_context["score_and_cpa_summary"], ensure_ascii=False, sort_keys=True)}

# TRAFFIC MIX SUMMARY
{json.dumps(decision_context["traffic_mix_summary"], ensure_ascii=False, sort_keys=True)}

# MEMORY DIAGNOSIS - USE AS DIAGNOSIS, NOT AS AN OVERRIDING COMMAND
{json.dumps(decision_context["memory_diagnosis"], ensure_ascii=False, sort_keys=True)}

# GAVE REFERENCE
- base_alpha: {base_alpha}
- base_bid_formula: bid = base_alpha * pValue

# ENVIRONMENT STATUS
- timeStepIndex: {time_step_index}
- budget: {budget}
- remaining_budget: {remaining_budget}
- budget_left_ratio: {safe_div(remaining_budget, budget, 0.0)}
- cpa_constraint: {cpa}
- spent_so_far: {spent}
- conversions_so_far: {historical_conversions}
- realized_cpa_so_far: {safe_div(spent, historical_conversions, 0.0)}

# DELIVERY PHASE POLICY
{json.dumps(phase_policy, ensure_ascii=False, sort_keys=True)}

# CURRENT GAVE STATE FEATURES
{json.dumps(gave_state_features, ensure_ascii=False, sort_keys=True)}

# SUCCESSFUL-BID ALPHA REFERENCES BY SCOPE
{json.dumps(success_refs, ensure_ascii=False, sort_keys=True)}

# COMPACT PERIOD-7 SUPPORTING EVIDENCE
{json.dumps(period7_evidence, ensure_ascii=False, sort_keys=True)}

Read order matters: treat PRIORITIZED DECISION BRIEF, MARKET POSITION SUMMARY, BUDGET PACING, SCORE/CPA, and TRAFFIC MIX as the main evidence. Use compact experience-period evidence only to verify details. Do not let risk words in memory override clear market evidence.

# TASK
Choose an additive adjustment for the next bidding factor. The final bidding factor will be:
final_alpha = max(base_alpha + adjustment, 0)
The adjustment is an absolute alpha delta, not a percentage or ratio. It may be in [{MIN_ADJUSTMENT}, {MAX_ADJUSTMENT}], but local guardrails will enforce context-aware bounds:
- normal evidence: usually keep adjustment in [-1, 1]
- risk-control evidence: if CPA/budget risk and base_alpha is above reliable market references, strong negative adjustment is allowed
- recovery evidence: every advertiser is expected to spend its available budget; if time_pacing_status is severely_behind_time_pacing, positive adjustment up to +4 is allowed and delivery recovery should not be blocked by noisy CPA alone
- exploration evidence: if severe underbid, behind pacing, and CPA is safe, positive adjustment up to +4 is allowed
Choose direction from ["up", "down", "hold"].
Also output pValue segment multipliers. These multipliers will be applied locally to each traffic row:
final_bid = final_alpha * pValue * multiplier_by_pValue_band.
Use these hard policy rules:
1. Do not make the decision from a single timestep label. Use segment_counts, segment_shares, segment_mix_summary, and dominant_segment to understand the current traffic mix. Treat lower_quartile as the lowest relative pValue band, not automatically bad traffic.
2. Segment names: premium_value means p75+ traffic by advertiser-won history or current timestep; strong_value means p50-p75 tier; standard_value means p25-p50 tier; lower_quartile means below both p25 references. The current-timestep references use current test-period pValue only, which is visible at bidding time.
3. Always compare all three market levels: advertiser_timestep_wins, category_timestep_wins, and global_timestep_wins. Market position should be judged before CPA risk.
4. If advertiser sample count is below 1000 or category sample count is below 10000, explicitly treat that layer as weaker evidence and use the provided sample counts/warnings.
5. Use SUCCESSFUL-BID ALPHA REFERENCES BY SCOPE as three-layer references. Compare avg/p50/p75/p90 winning_alpha, win_count, confidence, and conversion_rate; avoid blindly chasing the largest reference.
6. Choose up/down/hold and all segment multipliers from the full market, budget, CPA, GAVE state, memory, and segment evidence. There is no forced monotonic multiplier order; decide the values yourself from evidence.
7. Each multiplier must be in [0.95, 1.05]. Output multipliers only for segments that exist in the current timestep. Omit any segment whose corresponding current_has_*_traffic flag is false.
8. Follow the DELIVERY PHASE POLICY. In cold_start especially, low realized CPA or few conversions alone is not enough evidence for aggressive upward bidding; use experience-period budget_pacing and market references to justify increases.
9. Use MEMORY DIAGNOSIS to avoid repeated one-direction adjustments from the same reason. It is diagnostic context, not a hard command.
10. If memory_risk_flags includes expensive traffic or repeated upward adjustments with expensive traffic, do not keep raising bids merely because reference winning alpha is higher. In that case, prioritize CPA control: hold/down alpha, reduce lower_quartile/standard_value multipliers, and only keep a small premium_value multiplier when premium evidence is reliable.
11. Optimize score, not CPA alone. If expected_score_so_far_beta2 would likely fall because volume drops faster than CPA improves, avoid broad alpha cuts; prefer segment multipliers or hold. If score1/CPA risk is severe, preserve score through selective spend rather than only chasing conversions.
12. Budget delivery is mandatory: each advertiser should aim to use the full budget by the end of the episode. When low_spend_recovery_required is true, do not repeatedly hold/down because of noisy CPA alone; recover delivery using market references and segment selectivity.
13. Use |adjustment| > 1 only for clear regime evidence. Large positive moves require severe underbid, reliable market references, and behind time or experience-period pacing; large negative moves require clear CPA risk plus overbid evidence.
Return strict JSON only:
{{
  "direction": "up|down|hold",
  "adjustment_range": "one of {ADJUSTMENT_RANGES}",
  "adjustment": 0.0,
  "pvalue_multipliers": {{
    "lower_quartile": 1.0,
    "standard_value": 1.0,
    "strong_value": 1.0,
    "premium_value": 1.0
  }},
  "reason": "concise attribution in <= 60 words; avoid repeating raw numbers already shown in the input"
}}
"""
        repair_note = ""
        last_error = ""
        for attempt in range(self.MAX_REPAIR_ATTEMPTS + 1):
            active_prompt = prompt + repair_note
            try:
                data = self.llm_client.ask_json(active_prompt)
                raw_adjustment = float(data["adjustment"])
                if not math.isfinite(raw_adjustment):
                    raise ValueError(f"adjustment must be finite, got {raw_adjustment!r}")
                if raw_adjustment < MIN_ADJUSTMENT or raw_adjustment > MAX_ADJUSTMENT:
                    raise ValueError(
                        f"adjustment {raw_adjustment} is outside global range [{MIN_ADJUSTMENT}, {MAX_ADJUSTMENT}]"
                    )
                raw_adjustment = round(raw_adjustment, 4)
                adjustment, guardrail = contextual_adjustment_guardrail(
                    raw_adjustment=raw_adjustment,
                    base_alpha=base_alpha,
                    sql_summary=sql_summary,
                    score_status=score_status,
                    decision_memory=decision_memory or {},
                    budget=budget,
                    remaining_budget=remaining_budget,
                    time_step_index=time_step_index,
                )
                if not math.isfinite(adjustment) or adjustment < MIN_ADJUSTMENT or adjustment > MAX_ADJUSTMENT:
                    raise ValueError(
                        f"guardrail produced invalid adjustment {adjustment}; global range is "
                        f"[{MIN_ADJUSTMENT}, {MAX_ADJUSTMENT}]"
                    )
                direction = str(data.get("direction", "")).lower()
                if direction not in {"up", "down", "hold"} or guardrail["applied"]:
                    direction = "hold" if abs(adjustment) < 0.03 else ("up" if adjustment > 0 else "down")
                adjustment_range = str(data.get("adjustment_range") or adjustment_to_range(adjustment))
                if guardrail["applied"] or adjustment_range not in ADJUSTMENT_RANGES:
                    adjustment_range = adjustment_to_range(adjustment)
                multipliers = self._normalize_multipliers(data.get("pvalue_multipliers"), sql_summary)
                reason = str(data.get("reason", "LLM selected adjustment from GAVE and SQL signals."))[:360]
                if attempt > 0:
                    reason = (reason + f" Repaired after invalid prior output: {last_error}")[:360]
                if guardrail["applied"]:
                    guardrail_note = (
                        f" Guardrail {guardrail['mode']}: raw {guardrail['raw_adjustment']:.2f} -> "
                        f"{guardrail['bounded_adjustment']:.2f} within "
                        f"[{guardrail['lower_bound']:.2f}, {guardrail['upper_bound']:.2f}]."
                    )
                    reason = (reason + guardrail_note)[:360]
                source = "llm+guardrail" if guardrail["applied"] else "llm"
                if attempt > 0:
                    source += "+repair"
                return AdjustmentDecision(direction, adjustment_range, adjustment, reason, multipliers, source)
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.MAX_REPAIR_ATTEMPTS:
                    break
                repair_note = f"""

# PREVIOUS OUTPUT WAS INVALID - REPAIR REQUIRED
Your previous JSON could not be used: {last_error}
Return a fresh strict JSON decision now. The adjustment must be finite and inside [{MIN_ADJUSTMENT}, {MAX_ADJUSTMENT}].
Do not repeat the invalid value. Keep multipliers inside [0.95, 1.05].
"""
                print(
                    f"[demo] LLM decision invalid at timestep {time_step_index}: {last_error}. "
                    f"Requesting repaired decision ({attempt + 1}/{self.MAX_REPAIR_ATTEMPTS})...",
                    flush=True,
                )

        fallback_multipliers = self._default_multipliers(sql_summary)
        return AdjustmentDecision(
            "hold",
            adjustment_to_range(0.0),
            0.0,
            f"LLM decision invalid after repair attempts; using safe hold fallback. Last error: {last_error}"[:360],
            fallback_multipliers,
            "llm_repair_fallback",
        )

    def _default_multipliers(self, sql_summary: Dict[str, Any]) -> Dict[str, float]:
        decision_basis = sql_summary.get("pvalue_context", {}).get("decision_basis", {})
        multipliers = {}
        for key in self.MULTIPLIER_KEYS:
            if decision_basis.get(f"current_has_{key}_traffic"):
                multipliers[key] = 1.0
        if not multipliers:
            multipliers["standard_value"] = 1.0
        return multipliers

    def _normalize_multipliers(self, raw: Any, sql_summary: Dict[str, Any]) -> Dict[str, float]:
        multipliers = self._default_multipliers(sql_summary)
        if isinstance(raw, dict):
            for key in list(multipliers.keys()):
                if key in raw:
                    multipliers[key] = round(clip(as_float(raw[key], multipliers[key]), 0.95, 1.05), 4)
        return multipliers


class FinalBidAgent:
    """本地执行器：把 AdjustmentDecision 转换成实际的逐 PV bid。

English:
Local executor that converts an AdjustmentDecision into actual per-PV bids."""

    def __init__(self, min_alpha: float = 0.0):
        """设置 final alpha 的下界。

English:
Set a lower bound for final alpha."""
        self.min_alpha = min_alpha

    def finalize(
        self,
        base_alpha: float,
        base_bids: Any,
        decision: AdjustmentDecision,
        sql_summary: Optional[Dict[str, Any]] = None,
        budget: Optional[float] = None,
        remaining_budget: Optional[float] = None,
        cpa: Optional[float] = None,
        time_step_index: Optional[int] = None,
        p_values: Optional[Any] = None,
    ) -> Tuple[float, Any, str, Dict[str, Any]]:
        """生成 final alpha、最终 bid 数组、来源标签和分层策略元数据。

English:
Produce final alpha, final bid array, source label and segment strategy metadata."""
        import numpy as np

        final_alpha = max(base_alpha + decision.adjustment, self.min_alpha, 0.0)
        if sql_summary is None:
            raise RuntimeError("FinalBidAgent requires sql_summary for segmented bidding.")
        segment_strategy = self._build_segment_strategy(sql_summary, decision.pvalue_multipliers)
        if p_values is not None:
            bid, bid_summary = self._segment_bids(final_alpha, p_values, segment_strategy)
            segment_strategy["bid_summary"] = bid_summary
            return final_alpha, bid, "local_segment_executor", segment_strategy
        scale = safe_div(final_alpha, base_alpha, default=0.0) if base_alpha != 0 else 0.0
        return final_alpha, np.maximum(base_bids * scale, 0.0), "local_segment_executor", segment_strategy

    def _build_segment_strategy(self, sql_summary: Dict[str, Any], multipliers: Dict[str, float]) -> Dict[str, Any]:
        """构建 pValue 分层阈值，并附加 LLM 选择的 multiplier。

English:
Build pValue segment thresholds and attach the multipliers selected by LLM."""
        pvalue_context = sql_summary.get("pvalue_context", {})
        thresholds_by_scope = {}
        for scope in ("advertiser_timestep", "category_timestep", "global_timestep", "budget_peer_timestep"):
            history = pvalue_context.get(scope, {}).get("history", {})
            thresholds_by_scope[scope] = {
                "count": int(as_float(history.get("count"), 0.0)),
                "p25": as_float(history.get("p25")),
                "p50": as_float(history.get("p50")),
                "p75": as_float(history.get("p75")),
                "p90": as_float(history.get("p90")),
            }
        global_timestep_history = pvalue_context.get("global_timestep", {}).get("history", {})
        thresholds_by_scope["global_timestep"] = {
            "count": int(as_float(global_timestep_history.get("count"), 0.0)),
            "p25": as_float(global_timestep_history.get("p25")),
            "p50": as_float(global_timestep_history.get("p50")),
            "p75": as_float(global_timestep_history.get("p75")),
            "p90": as_float(global_timestep_history.get("p90")),
        }
        decision_basis = pvalue_context.get("decision_basis", {})
        thresholds = {
            "advertiser_won_p25": as_float(decision_basis.get("advertiser_won_p25_threshold")),
            "advertiser_won_p50": as_float(decision_basis.get("advertiser_won_p50_threshold")),
            "advertiser_won_p75": as_float(decision_basis.get("advertiser_won_p75_threshold")),
            "current_timestep_p25": as_float(decision_basis.get("current_timestep_p25_threshold")),
            "current_timestep_p50": as_float(decision_basis.get("current_timestep_p50_threshold")),
            "current_timestep_p75": as_float(decision_basis.get("current_timestep_p75_threshold")),
            "advertiser_overall_p25": as_float(decision_basis.get("advertiser_overall_p25_threshold")),
            "advertiser_overall_p50": as_float(decision_basis.get("advertiser_overall_p50_threshold")),
            "advertiser_overall_p75": as_float(decision_basis.get("advertiser_overall_p75_threshold")),
            "global_timestep_p25": as_float(decision_basis.get("global_timestep_p25_threshold")),
            "global_timestep_p50": as_float(decision_basis.get("global_timestep_p50_threshold")),
            "global_timestep_p75": as_float(decision_basis.get("global_timestep_p75_threshold")),
            "effective_standard_value_threshold": as_float(decision_basis.get("effective_standard_value_threshold")),
            "effective_strong_value_threshold": as_float(decision_basis.get("effective_strong_value_threshold")),
            "effective_premium_value_threshold": as_float(decision_basis.get("effective_premium_value_threshold")),
        }
        return {
            "type": "four_segment_multiplier",
            "threshold_scope": "advertiser_won_or_current_timestep_p25_p50_p75",
            "thresholds": thresholds,
            "thresholds_by_scope": thresholds_by_scope,
            "segment_mix": {
                "counts": decision_basis.get("segment_counts", {}),
                "shares": decision_basis.get("segment_shares", {}),
                "dominant_segment": decision_basis.get("dominant_segment", decision_basis.get("segment", "unknown")),
                "summary": decision_basis.get("segment_mix_summary", ""),
            },
            "multipliers": multipliers,
            "rule": "premium_value if pValue >= advertiser_won_p75 OR pValue >= current_timestep_p75; else strong_value if pValue >= advertiser_won_p50 OR pValue >= current_timestep_p50; else standard_value if pValue >= advertiser_won_p25 OR pValue >= current_timestep_p25; else lower_quartile. final_bid = final_alpha * pValue * multiplier_by_segment",
        }

    def _segment_bids(self, final_alpha: float, p_values: Any, segment_strategy: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """把分层 multiplier 应用到每个 PV，并返回 bid 数组和分层计数。

English:
Apply segment multipliers to each PV and return bid array plus segment counts."""
        import numpy as np

        thresholds = segment_strategy["thresholds"]
        multipliers = segment_strategy["multipliers"]
        advertiser_p25 = thresholds.get("advertiser_won_p25", thresholds.get("advertiser_overall_p25", 0.0))
        advertiser_p50 = thresholds.get("advertiser_won_p50", thresholds["advertiser_overall_p50"])
        advertiser_p75 = thresholds.get("advertiser_won_p75", thresholds["advertiser_overall_p75"])
        current_timestep_p25 = thresholds.get("current_timestep_p25", 0.0)
        current_timestep_p50 = thresholds.get("current_timestep_p50", thresholds["global_timestep_p50"])
        current_timestep_p75 = thresholds.get("current_timestep_p75", thresholds["global_timestep_p75"])
        multiplier_array = np.ones(p_values.shape[0], dtype=float)
        premium_mask = np.zeros(p_values.shape[0], dtype=bool)
        strong_or_premium_mask = np.zeros(p_values.shape[0], dtype=bool)
        standard_or_above_mask = np.zeros(p_values.shape[0], dtype=bool)
        if advertiser_p75 > 0:
            premium_mask = premium_mask | (p_values >= advertiser_p75)
        if current_timestep_p75 > 0:
            premium_mask = premium_mask | (p_values >= current_timestep_p75)
        if advertiser_p50 > 0:
            strong_or_premium_mask = strong_or_premium_mask | (p_values >= advertiser_p50)
        if current_timestep_p50 > 0:
            strong_or_premium_mask = strong_or_premium_mask | (p_values >= current_timestep_p50)
        if advertiser_p25 > 0:
            standard_or_above_mask = standard_or_above_mask | (p_values >= advertiser_p25)
        if current_timestep_p25 > 0:
            standard_or_above_mask = standard_or_above_mask | (p_values >= current_timestep_p25)
        strong_mask = strong_or_premium_mask & ~premium_mask
        standard_mask = standard_or_above_mask & ~(premium_mask | strong_mask)
        lower_mask = ~(premium_mask | strong_mask | standard_mask)
        masks = {
            "lower_quartile": lower_mask,
            "standard_value": standard_mask,
            "strong_value": strong_mask,
            "premium_value": premium_mask,
        }
        default_multiplier = multipliers.get("standard_value", 1.0)
        for key, mask in masks.items():
            multiplier_array[mask] = multipliers.get(key, default_multiplier)
        bids = np.maximum(final_alpha * p_values * multiplier_array, 0.0)
        total = max(int(p_values.shape[0]), 1)
        bid_summary = {
            "lower_quartile_count": int(np.sum(masks["lower_quartile"])),
            "standard_value_count": int(np.sum(masks["standard_value"])),
            "strong_value_count": int(np.sum(masks["strong_value"])),
            "premium_value_count": int(np.sum(masks["premium_value"])),
            "lower_quartile_share": float(np.sum(masks["lower_quartile"]) / total),
            "standard_value_share": float(np.sum(masks["standard_value"]) / total),
            "strong_value_share": float(np.sum(masks["strong_value"]) / total),
            "premium_value_share": float(np.sum(masks["premium_value"]) / total),
            "avg_multiplier": float(np.mean(multiplier_array)) if multiplier_array.shape[0] else 0.0,
            "avg_bid": float(np.mean(bids)) if bids.shape[0] else 0.0,
            "max_bid": float(np.max(bids)) if bids.shape[0] else 0.0,
        }
        return bids, bid_summary

def adjustment_to_range(adjustment: float) -> str:
    """把连续调价值映射为展示区间。

English:
Map a continuous adjustment value into a display bucket."""
    adjustment = clip(adjustment, MIN_ADJUSTMENT, MAX_ADJUSTMENT)
    if adjustment >= MAX_ADJUSTMENT - 0.1:
        return ADJUSTMENT_RANGES[-1]
    index = int(math.floor((adjustment - MIN_ADJUSTMENT) / 0.1))
    index = max(0, min(index, len(ADJUSTMENT_RANGES) - 1))
    return ADJUSTMENT_RANGES[index]


def simulate_ad_bidding(p_values: Any, p_value_sigmas: Any, bids: Any, least_winning_costs: Any) -> Tuple[Any, Any, Any, Any]:
    """离线拍卖模拟器：判断竞胜、成本，并采样转化。

English:
Offline auction simulator: decide wins/costs and sample conversions."""
    import numpy as np

    tick_status = bids >= least_winning_costs
    tick_cost = least_winning_costs * tick_status
    values = np.random.normal(loc=p_values, scale=p_value_sigmas) * tick_status
    tick_value = np.clip(values, 0, 1)
    tick_conversion = np.random.binomial(n=1, p=tick_value)
    return tick_value, tick_cost, tick_status, tick_conversion


def get_score(reward: float, cpa: float, cpa_constraint: float, beta: int) -> float:
    """竞赛评分：当实际 CPA 超过约束时，对转化数施加惩罚。

English:
Competition score: conversions penalized when realized CPA exceeds constraint."""
    penalty = 1.0
    if cpa > cpa_constraint:
        penalty = pow(cpa_constraint / (cpa + 1e-10), beta)
    return penalty * reward


def build_score_status(
    cost_so_far: float,
    conversions_so_far: float,
    expected_conversions_so_far: float,
    cpa_constraint: float,
) -> Dict[str, float]:
    """构建累计 score/CPA 诊断信息，供 LLM prompt 使用。

English:
Build cumulative score/CPA diagnostics for the LLM prompt."""
    realized_cpa = safe_div(cost_so_far, conversions_so_far, 0.0)
    expected_cpa = safe_div(cost_so_far, expected_conversions_so_far, 0.0)
    return {
        "cost_so_far": cost_so_far,
        "conversions_so_far": conversions_so_far,
        "expected_conversions_so_far": expected_conversions_so_far,
        "realized_cpa_so_far": realized_cpa,
        "expected_cpa_so_far": expected_cpa,
        "score_so_far_beta2": get_score(conversions_so_far, realized_cpa, cpa_constraint, beta=2),
        "score_so_far_beta5": get_score(conversions_so_far, realized_cpa, cpa_constraint, beta=5),
        "expected_score_so_far_beta2": get_score(expected_conversions_so_far, expected_cpa, cpa_constraint, beta=2),
        "expected_score_so_far_beta5": get_score(expected_conversions_so_far, expected_cpa, cpa_constraint, beta=5),
        "cpa_constraint": cpa_constraint,
        "cpa_pressure_ratio": safe_div(expected_cpa, cpa_constraint, 0.0),
    }


def success_reference_by_scope(sql_summary: Dict[str, Any]) -> Dict[str, Any]:
    """从 sql_summary 中提取紧凑的 winning-alpha 参考表。

English:
Extract a compact winning-alpha reference table from sql_summary."""
    success_context = sql_summary.get("success_context", {})
    references = {}
    for scope, item in success_context.get("win_reference_by_scope", {}).items():
        references[scope] = {
            "win_count": int(as_float(item.get("win_count"))),
            "avg_winning_alpha": as_float(item.get("avg_winning_alpha")),
            "p50_winning_alpha": as_float(item.get("p50_winning_alpha")),
            "p75_winning_alpha": as_float(item.get("p75_winning_alpha")),
            "p90_winning_alpha": as_float(item.get("p90_winning_alpha")),
            "avg_least_winning_cost": as_float(item.get("avg_least_winning_cost")),
            "conversion_rate": as_float(item.get("conversion_rate")),
            "confidence": item.get("confidence", "none"),
        }
    return references


def compact_sql_trace_summary(sql_summary: Dict[str, Any]) -> Dict[str, Any]:
    """在写入 decision_trace.csv 前压缩庞大的 SQL/KB summary。

English:
Shrink the large SQL/KB summary before writing it to decision_trace.csv."""
    pvalue_context = sql_summary.get("pvalue_context", {})
    market_context = sql_summary.get("market_context", {})
    success_context = sql_summary.get("success_context", {})

    def compact_history(scope: str) -> Dict[str, Any]:
        item = pvalue_context.get(scope, {})
        history = item.get("history", {})
        won_history = item.get("won_history", {})
        return {
            "history_count": int(as_float(history.get("count"))),
            "history_p50": as_float(history.get("p50")),
            "history_p75": as_float(history.get("p75")),
            "history_p90": as_float(history.get("p90")),
            "won_count": int(as_float(won_history.get("count"))),
            "won_p50": as_float(won_history.get("p50")),
            "won_p75": as_float(won_history.get("p75")),
            "current_mean_rank": item.get("current_mean_rank", {}),
        }

    def compact_market(scope: str) -> Dict[str, Any]:
        item = market_context.get(scope, {})
        pacing = item.get("budget_pacing", {}) if isinstance(item.get("budget_pacing"), dict) else {}
        return {
            "traffic_num": int(as_float(item.get("traffic_num"))),
            "avg_pvalue": as_float(item.get("avg_pvalue")),
            "avg_bid": as_float(item.get("avg_bid")),
            "avg_market_price": as_float(item.get("avg_market_price")),
            "avg_winning_market_price": as_float(item.get("avg_winning_market_price")),
            "win_rate": as_float(item.get("win_rate")),
            "conversion_rate": as_float(item.get("conversion_rate")),
            "cpa_proxy": as_float(item.get("cpa_proxy")),
            "budget_spend_ratio": as_float(
                pacing.get("advertiser_weighted_budget_spend_ratio", item.get("row_weighted_budget_spend_ratio"))
            ),
        }

    return {
        "current": {
            "pvalue_mean": as_float(sql_summary.get("current_pvalue_mean")),
            "pv_count": int(as_float(sql_summary.get("current_pv_count"))),
            "p25": as_float(sql_summary.get("current_pvalue_p25")),
            "p50": as_float(sql_summary.get("current_pvalue_p50")),
            "p75": as_float(sql_summary.get("current_pvalue_p75")),
            "p90": as_float(sql_summary.get("current_pvalue_p90")),
        },
        "decision_basis": pvalue_context.get("decision_basis", {}),
        "pvalue_scopes": {
            scope: compact_history(scope)
            for scope in ("advertiser_timestep", "category_timestep", "global_timestep", "budget_peer_timestep")
        },
        "market_scopes": {
            scope: compact_market(scope)
            for scope in ("advertiser", "category", "global", "budget_peer")
        },
        "success_refs": success_reference_by_scope(sql_summary),
        "success_reference_alpha_summary": success_context.get("reference_alpha_summary", {}),
    }


def reliable_success_alpha_values(success_refs: Dict[str, Any], quantile_key: str = "p75_winning_alpha") -> List[float]:
    """从中高置信度范围收集可用 winning-alpha 参考。

English:
Collect usable winning-alpha references from medium/high confidence scopes."""
    values: List[float] = []
    for scope in ("category_timestep_wins", "global_timestep_wins", "advertiser_timestep_wins"):
        ref = success_refs.get(scope, {})
        confidence = str(ref.get("confidence", "none")).lower()
        win_count = int(as_float(ref.get("win_count")))
        if confidence not in {"high", "medium"} or win_count <= 0:
            continue
        value = as_float(ref.get(quantile_key))
        if value <= 0:
            value = as_float(ref.get("avg_winning_alpha"))
        if value > 0:
            values.append(value)
    return values


def contextual_adjustment_guardrail(
    raw_adjustment: float,
    base_alpha: float,
    sql_summary: Dict[str, Any],
    score_status: Dict[str, Any],
    decision_memory: Dict[str, Any],
    budget: float,
    remaining_budget: float,
    time_step_index: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """根据预算节奏、CPA 风险和市场位置约束 LLM 调价。

English:
Bound the LLM adjustment according to pacing, CPA risk and market-position signals."""
    success_refs = success_reference_by_scope(sql_summary)
    market_position = build_market_position_summary(base_alpha, sql_summary, success_refs)
    budget_pacing = build_budget_pacing_summary(budget, remaining_budget, sql_summary, time_step_index)
    risk_flags = decision_memory.get("memory_risk_flags", []) if isinstance(decision_memory, dict) else []
    cpa_pressure = as_float(score_status.get("cpa_pressure_ratio"))
    remaining_ratio = safe_div(remaining_budget, budget, 0.0)
    avg_market_ratio = as_float(market_position.get("avg_base_to_winning_alpha_ratio"))
    time_pacing_status = str(budget_pacing.get("time_pacing_status", "unknown"))
    mode = "normal_small_adjustment"
    lower = -1.0
    upper = 1.0
    target_alpha = 0.0
    reasons = []

    risk_signal = (
        cpa_pressure > 1.10
        or "recent_expected_cpa_above_constraint" in risk_flags
        or "episode_expected_cpa_above_constraint" in risk_flags
        or "recent_cost_spike" in risk_flags
        or remaining_ratio < 0.20
    )
    market_overbid_signal = (
        market_position.get("market_position") == "overbid_vs_period7_winning_alpha"
        or avg_market_ratio > 1.15
    )
    market_underbid_signal = market_position.get("market_position") == "underbid_vs_period7_winning_alpha"
    pacing_behind = budget_pacing.get("pacing_status") == "behind_period7_pacing"
    severe_time_pacing_behind = time_pacing_status == "severely_behind_time_pacing"
    time_pacing_behind = severe_time_pacing_behind or time_pacing_status == "behind_time_pacing"
    expensive_upward_memory = "repeated_upward_adjustments_with_expensive_traffic" in risk_flags

    if severe_time_pacing_behind and remaining_ratio > 0.15:
        lower = -0.25
        upper = MAX_ADJUSTMENT if (cpa_pressure <= 2.0 or market_underbid_signal) else 2.0
        mode = "low_spend_recovery"
        reasons.append("budget spend is severely behind elapsed time; recovery applies to every advertiser")
    elif time_pacing_behind and market_underbid_signal and remaining_ratio > 0.10 and cpa_pressure <= 1.30:
        lower = -0.5
        upper = min(MAX_ADJUSTMENT, 3.0)
        mode = "pacing_recovery_limited_up"
        reasons.append("budget spend is behind elapsed time and market references indicate underbidding")
    elif risk_signal and market_overbid_signal:
        refs = reliable_success_alpha_values(success_refs, "p75_winning_alpha")
        if refs:
            target_alpha = min(refs)
            if base_alpha > target_alpha:
                lower = max(MIN_ADJUSTMENT, target_alpha - base_alpha)
                required_brake = min(-1.0, (target_alpha - base_alpha) * 0.5)
                upper = min(0.0, max(lower, required_brake))
            else:
                lower = -2.0
                upper = 0.0
        else:
            lower = -3.0
            upper = 0.0
        mode = "risk_control_strong_down"
        reasons.append("CPA/budget risk and base_alpha above reliable experience-period market reference")
    elif risk_signal or expensive_upward_memory:
        lower = -2.0
        upper = 0.25
        mode = "risk_control_limited_up"
        reasons.append("risk signal present; allow stronger down but cap upward move")
    elif market_underbid_signal and pacing_behind and cpa_pressure <= 1.05:
        lower = -0.5
        upper = 2.0
        mode = "exploration_limited_up"
        reasons.append("underbid with behind pacing and no CPA pressure")

    bounded = round(clip(raw_adjustment, lower, upper), 4)
    return bounded, {
        "mode": mode,
        "raw_adjustment": raw_adjustment,
        "bounded_adjustment": bounded,
        "lower_bound": lower,
        "upper_bound": upper,
        "target_alpha": target_alpha,
        "market_position": market_position.get("market_position"),
        "avg_base_to_winning_alpha_ratio": avg_market_ratio,
        "budget_pacing": budget_pacing.get("pacing_status"),
        "time_pacing_status": time_pacing_status,
        "cpa_pressure_ratio": cpa_pressure,
        "risk_flags": risk_flags,
        "applied": abs(bounded - raw_adjustment) > 1e-9,
        "reasons": reasons,
    }


def attach_current_segment_mix(sql_summary: Dict[str, Any], p_values: Any) -> None:
    """附加当前 PV 在 lower/standard/strong/premium 各价值层的数量和占比。

English:
Attach current PV counts/shares for lower/standard/strong/premium segments."""
    import numpy as np

    decision_basis = sql_summary.get("pvalue_context", {}).get("decision_basis", {})
    p_values = np.asarray(p_values, dtype=float)
    total = int(p_values.shape[0])
    premium_mask = np.zeros(total, dtype=bool)
    strong_or_premium_mask = np.zeros(total, dtype=bool)
    standard_or_above_mask = np.zeros(total, dtype=bool)
    for key in ("advertiser_won_p75_threshold", "current_timestep_p75_threshold"):
        threshold = as_float(decision_basis.get(key))
        if threshold > 0:
            premium_mask |= p_values >= threshold
    for key in ("advertiser_won_p50_threshold", "current_timestep_p50_threshold"):
        threshold = as_float(decision_basis.get(key))
        if threshold > 0:
            strong_or_premium_mask |= p_values >= threshold
    for key in ("advertiser_won_p25_threshold", "current_timestep_p25_threshold"):
        threshold = as_float(decision_basis.get(key))
        if threshold > 0:
            standard_or_above_mask |= p_values >= threshold
    strong_mask = strong_or_premium_mask & ~premium_mask
    standard_mask = standard_or_above_mask & ~(premium_mask | strong_mask)
    lower_mask = ~(premium_mask | strong_mask | standard_mask)
    counts = {
        "lower_quartile": int(np.sum(lower_mask)),
        "standard_value": int(np.sum(standard_mask)),
        "strong_value": int(np.sum(strong_mask)),
        "premium_value": int(np.sum(premium_mask)),
    }
    denominator = max(total, 1)
    shares = {key: value / denominator for key, value in counts.items()}
    nonzero_segments = [key for key, value in counts.items() if value > 0]
    dominant_segment = max(counts, key=counts.get) if total else "standard_value"
    decision_basis.update(
        {
            "segment": nonzero_segments[0] if len(nonzero_segments) == 1 else "mixed",
            "dominant_segment": dominant_segment,
            "segment_counts": counts,
            "segment_shares": shares,
            "segment_mix_summary": (
                f"lower_quartile={shares['lower_quartile']:.2%}, standard_value={shares['standard_value']:.2%}, "
                f"strong_value={shares['strong_value']:.2%}, premium_value={shares['premium_value']:.2%}; "
                f"dominant={dominant_segment}"
            ),
            "current_has_lower_quartile_traffic": counts["lower_quartile"] > 0,
            "current_has_standard_value_traffic": counts["standard_value"] > 0,
            "current_has_strong_value_traffic": counts["strong_value"] > 0,
            "current_has_premium_value_traffic": counts["premium_value"] > 0,
            "current_has_mid_value_traffic": counts["strong_value"] > 0,
            "current_has_high_value_traffic": counts["premium_value"] > 0,
        }
    )


def compact_market_scope(scope_data: Dict[str, Any], base_alpha: float, current_pvalue_mean: float) -> Dict[str, Any]:
    """压缩市场证据，形成适合 prompt 的市场位置摘要。

English:
Compress market evidence for prompt-friendly market-position summaries."""
    budget_pacing = scope_data.get("budget_pacing", {}) if isinstance(scope_data.get("budget_pacing", {}), dict) else {}
    base_bid_at_mean = base_alpha * current_pvalue_mean
    avg_winning_market_price = as_float(scope_data.get("avg_winning_market_price"))
    return {
        "traffic_num": int(as_float(scope_data.get("traffic_num"))),
        "win_rate": as_float(scope_data.get("win_rate")),
        "conversion_rate": as_float(scope_data.get("conversion_rate")),
        "cpa_proxy": as_float(scope_data.get("cpa_proxy")),
        "avg_pvalue": as_float(scope_data.get("avg_pvalue")),
        "avg_bid": as_float(scope_data.get("avg_bid")),
        "bid_p50": as_float(scope_data.get("bid_p50")),
        "bid_p75": as_float(scope_data.get("bid_p75")),
        "avg_market_price": as_float(scope_data.get("avg_market_price")),
        "avg_winning_market_price": avg_winning_market_price,
        "market_price_p50": as_float(scope_data.get("market_price_p50")),
        "market_price_p75": as_float(scope_data.get("market_price_p75")),
        "base_bid_at_current_mean_pvalue": base_bid_at_mean,
        "base_bid_to_avg_winning_market_price": safe_div(base_bid_at_mean, avg_winning_market_price, 0.0),
        "budget_spend_ratio_reference": as_float(
            budget_pacing.get("advertiser_weighted_budget_spend_ratio", scope_data.get("row_weighted_budget_spend_ratio"))
        ),
        "budget_left_ratio_reference": as_float(
            budget_pacing.get("advertiser_weighted_budget_left_ratio", scope_data.get("row_weighted_budget_left_ratio"))
        ),
    }


def build_market_position_summary(
    base_alpha: float,
    sql_summary: Dict[str, Any],
    success_refs: Dict[str, Any],
) -> Dict[str, Any]:
    """比较 base_alpha 与历史 winning-alpha、市场价参考之间的位置。

English:
Compare base_alpha against historical winning-alpha and market-price references."""
    current_pvalue_mean = as_float(sql_summary.get("current_pvalue_mean"))
    scope_rows = {}
    usable_ratios = []
    for scope in ("advertiser_timestep_wins", "category_timestep_wins", "global_timestep_wins"):
        ref = success_refs.get(scope, {})
        avg_alpha = as_float(ref.get("avg_winning_alpha"))
        ratio = safe_div(base_alpha, avg_alpha, 0.0)
        scope_rows[scope] = {
            "win_count": int(as_float(ref.get("win_count"))),
            "confidence": ref.get("confidence", "none"),
            "avg_winning_alpha": avg_alpha,
            "p50_winning_alpha": as_float(ref.get("p50_winning_alpha")),
            "p75_winning_alpha": as_float(ref.get("p75_winning_alpha")),
            "base_alpha_to_avg_winning_alpha": ratio,
            "conversion_rate": as_float(ref.get("conversion_rate")),
        }
        if avg_alpha > 0 and as_float(ref.get("win_count")) > 0:
            usable_ratios.append(ratio)

    avg_ratio = safe_div(sum(usable_ratios), len(usable_ratios), 0.0)
    if not usable_ratios:
        market_position = "unknown"
    elif avg_ratio < 0.85:
        market_position = "underbid_vs_period7_winning_alpha"
    elif avg_ratio > 1.15:
        market_position = "overbid_vs_period7_winning_alpha"
    else:
        market_position = "near_period7_winning_alpha"

    market_context = sql_summary.get("market_context", {})
    market_scopes = {
        scope: compact_market_scope(market_context.get(scope, {}), base_alpha, current_pvalue_mean)
        for scope in ("advertiser", "category", "global")
    }
    return {
        "market_position": market_position,
        "avg_base_to_winning_alpha_ratio": avg_ratio,
        "success_alpha_by_scope": scope_rows,
        "market_price_by_scope": market_scopes,
    }


def build_budget_pacing_summary(
    budget: float,
    remaining_budget: float,
    sql_summary: Dict[str, Any],
    time_step_index: Optional[int] = None,
) -> Dict[str, Any]:
    """总结当前消耗相对时间进度和经验参考是超前还是滞后。

English:
Summarize whether current spend is ahead/behind time and experience references."""
    current_spend_ratio = safe_div(budget - remaining_budget, budget, 0.0)
    current_budget_left_ratio = safe_div(remaining_budget, budget, 0.0)
    if time_step_index is None:
        elapsed_time_ratio = 0.0
        time_left_ratio = 0.0
        time_pacing_status = "unknown"
        spend_vs_elapsed_gap = 0.0
    else:
        elapsed_time_ratio = clip((float(time_step_index) + 1.0) / DEFAULT_TOTAL_TIMESTEPS, 0.0, 1.0)
        time_left_ratio = max(0.0, 1.0 - elapsed_time_ratio)
        spend_vs_elapsed_gap = elapsed_time_ratio - current_spend_ratio
        if elapsed_time_ratio <= 0:
            time_pacing_status = "unknown"
        elif current_spend_ratio < elapsed_time_ratio * 0.45 and spend_vs_elapsed_gap > 0.10:
            time_pacing_status = "severely_behind_time_pacing"
        elif current_spend_ratio < elapsed_time_ratio * 0.75 and spend_vs_elapsed_gap > 0.05:
            time_pacing_status = "behind_time_pacing"
        elif current_spend_ratio > elapsed_time_ratio * 1.25 and current_spend_ratio - elapsed_time_ratio > 0.05:
            time_pacing_status = "ahead_of_time_pacing"
        else:
            time_pacing_status = "near_time_pacing"
    market_context = sql_summary.get("market_context", {})
    references = {}
    usable_refs = []
    for scope in ("advertiser", "category", "global"):
        scope_data = market_context.get(scope, {})
        budget_pacing = scope_data.get("budget_pacing", {}) if isinstance(scope_data.get("budget_pacing", {}), dict) else {}
        ref = as_float(
            budget_pacing.get("advertiser_weighted_budget_spend_ratio", scope_data.get("row_weighted_budget_spend_ratio"))
        )
        references[scope] = ref
        if ref > 0:
            usable_refs.append(ref)
    reference_spend_ratio = safe_div(sum(usable_refs), len(usable_refs), 0.0)
    if reference_spend_ratio <= 0:
        pacing_status = "unknown"
    elif current_spend_ratio < reference_spend_ratio * 0.75:
        pacing_status = "behind_period7_pacing"
    elif current_spend_ratio > reference_spend_ratio * 1.25:
        pacing_status = "ahead_of_period7_pacing"
    else:
        pacing_status = "near_period7_pacing"
    return {
        "current_spend_ratio": current_spend_ratio,
        "current_budget_left_ratio": current_budget_left_ratio,
        "elapsed_time_ratio": elapsed_time_ratio,
        "time_left_ratio": time_left_ratio,
        "spend_vs_elapsed_time_gap": spend_vs_elapsed_gap,
        "time_pacing_status": time_pacing_status,
        "must_spend_budget_objective": "spend_remaining_budget_by_episode_end",
        "low_spend_recovery_required": time_pacing_status == "severely_behind_time_pacing" and current_budget_left_ratio > 0.15,
        "period7_spend_ratio_reference_avg": reference_spend_ratio,
        "period7_spend_ratio_by_scope": references,
        "pacing_status": pacing_status,
    }


def build_compact_period7_evidence(sql_summary: Dict[str, Any], base_alpha: float) -> Dict[str, Any]:
    """从 SQL/KB summary 中保留对 prompt 有用的历史证据。

English:
Keep only prompt-useful historical evidence from the SQL/KB summary."""
    pvalue_context = sql_summary.get("pvalue_context", {})
    decision_basis = pvalue_context.get("decision_basis", {})
    thresholds_by_scope = {}
    for scope in ("advertiser", "category", "global", "global_timestep"):
        history = pvalue_context.get(scope, {}).get("history", {})
        thresholds_by_scope[scope] = {
            "count": int(as_float(history.get("count"))),
            "p50": as_float(history.get("p50")),
            "p75": as_float(history.get("p75")),
            "p90": as_float(history.get("p90")),
        }
    return {
        "current_traffic": {
            "pv_count": int(as_float(sql_summary.get("current_pv_count"))),
            "pvalue_mean": as_float(sql_summary.get("current_pvalue_mean")),
            "pvalue_p25": as_float(sql_summary.get("current_pvalue_p25")),
            "pvalue_p50": as_float(sql_summary.get("current_pvalue_p50")),
            "pvalue_p75": as_float(sql_summary.get("current_pvalue_p75")),
            "pvalue_p90": as_float(sql_summary.get("current_pvalue_p90")),
            "pvalue_max": as_float(sql_summary.get("current_pvalue_max")),
        },
        "segment_basis": {
            "segment": decision_basis.get("segment"),
            "dominant_segment": decision_basis.get("dominant_segment"),
            "segment_counts": decision_basis.get("segment_counts", {}),
            "segment_shares": decision_basis.get("segment_shares", {}),
            "segment_mix_summary": decision_basis.get("segment_mix_summary", ""),
            "current_has_mid_value_traffic": decision_basis.get("current_has_mid_value_traffic", False),
            "current_has_high_value_traffic": decision_basis.get("current_has_high_value_traffic", False),
            "current_has_lower_quartile_traffic": decision_basis.get("current_has_lower_quartile_traffic", False),
            "current_has_standard_value_traffic": decision_basis.get("current_has_standard_value_traffic", False),
            "current_has_strong_value_traffic": decision_basis.get("current_has_strong_value_traffic", False),
            "current_has_premium_value_traffic": decision_basis.get("current_has_premium_value_traffic", False),
            "advertiser_won_p25_threshold": as_float(decision_basis.get("advertiser_won_p25_threshold")),
            "advertiser_won_p50_threshold": as_float(decision_basis.get("advertiser_won_p50_threshold")),
            "advertiser_won_p75_threshold": as_float(decision_basis.get("advertiser_won_p75_threshold")),
            "advertiser_won_sample_count": int(as_float(decision_basis.get("advertiser_won_sample_count"))),
            "current_timestep_p25_threshold": as_float(decision_basis.get("current_timestep_p25_threshold")),
            "current_timestep_p50_threshold": as_float(decision_basis.get("current_timestep_p50_threshold")),
            "current_timestep_p75_threshold": as_float(decision_basis.get("current_timestep_p75_threshold")),
            "effective_standard_value_threshold": as_float(decision_basis.get("effective_standard_value_threshold")),
            "effective_strong_value_threshold": as_float(decision_basis.get("effective_strong_value_threshold")),
            "effective_premium_value_threshold": as_float(decision_basis.get("effective_premium_value_threshold")),
        },
        "pvalue_thresholds_by_scope": thresholds_by_scope,
        "sample_warnings": decision_basis.get("sample_warnings", {}),
        "base_alpha": base_alpha,
    }


def build_decision_brief(
    base_alpha: float,
    sql_summary: Dict[str, Any],
    success_refs: Dict[str, Any],
    score_status: Dict[str, Any],
    decision_memory: Dict[str, Any],
    budget: float,
    remaining_budget: float,
    time_step_index: Optional[int] = None,
) -> Dict[str, Any]:
    """把市场、节奏、CPA 和记忆信号合并为最终 agent 的紧凑简报。

English:
Merge market, pacing, CPA and memory signals into a compact final-agent brief."""
    market_position = build_market_position_summary(base_alpha, sql_summary, success_refs)
    budget_pacing = build_budget_pacing_summary(budget, remaining_budget, sql_summary, time_step_index)
    pvalue_basis = sql_summary.get("pvalue_context", {}).get("decision_basis", {})
    cpa_pressure = as_float(score_status.get("cpa_pressure_ratio"))
    if cpa_pressure <= 0:
        cpa_status = "no_spend_or_no_expected_conversion_signal"
    elif cpa_pressure <= 0.9:
        cpa_status = "safe"
    elif cpa_pressure <= 1.1:
        cpa_status = "watch"
    else:
        cpa_status = "breached"

    risk_flags = decision_memory.get("memory_risk_flags", []) if isinstance(decision_memory, dict) else []
    if "repeated_downward_adjustments_without_score_gain" in risk_flags:
        score_tradeoff = "broad_down_adjustments_have_not_improved_expected_score"
        recommended_bias = "hold_or_segment_selective_rebound"
    elif cpa_status == "breached" and budget_pacing["pacing_status"] == "ahead_of_period7_pacing":
        score_tradeoff = "score1_cpa_risk_is_high; reduce inefficient spend selectively"
        recommended_bias = "selective_down"
    elif (
        market_position["market_position"] == "underbid_vs_period7_winning_alpha"
        and (
            budget_pacing["pacing_status"] == "behind_period7_pacing"
            or budget_pacing["time_pacing_status"] in {"behind_time_pacing", "severely_behind_time_pacing"}
        )
        and cpa_status in {"safe", "watch", "no_spend_or_no_expected_conversion_signal"}
    ):
        score_tradeoff = "volume_loss_may_hurt_score_more_than_small_cpa_risk"
        recommended_bias = "cautious_up"
    elif cpa_status == "breached":
        score_tradeoff = "cpa_penalty_exists; avoid broad volume chase but do not cut if score keeps falling"
        recommended_bias = "hold_or_selective_down"
    else:
        score_tradeoff = "balanced; use market position and segment mix to decide"
        recommended_bias = "hold_selective"

    return {
        "decision_brief": {
            "market_position": market_position["market_position"],
            "cpa_status": cpa_status,
            "budget_pacing": budget_pacing["pacing_status"],
            "time_pacing": budget_pacing["time_pacing_status"],
            "must_spend_budget": True,
            "low_spend_recovery_required": budget_pacing["low_spend_recovery_required"],
            "traffic_mix": pvalue_basis.get("segment_mix_summary", ""),
            "score_tradeoff": score_tradeoff,
            "recommended_bias": recommended_bias,
        },
        "market_position_summary": market_position,
        "budget_pacing_summary": budget_pacing,
        "score_and_cpa_summary": score_status,
        "memory_diagnosis": {
            "risk_flags": risk_flags,
            "recent_5_summary": decision_memory.get("recent_5_summary", "") if isinstance(decision_memory, dict) else "",
            "followup_advice": decision_memory.get("followup_advice", "") if isinstance(decision_memory, dict) else "",
        },
        "traffic_mix_summary": {
            "segment": pvalue_basis.get("segment", "unknown"),
            "dominant_segment": pvalue_basis.get("dominant_segment", "unknown"),
            "segment_counts": pvalue_basis.get("segment_counts", {}),
            "segment_shares": pvalue_basis.get("segment_shares", {}),
            "segment_mix_summary": pvalue_basis.get("segment_mix_summary", ""),
        },
    }


def empty_history() -> Dict[str, list]:
    """创建每个广告主的 rollout 历史，供 GAVE 和规则护栏使用。

English:
Create the per-advertiser rollout history consumed by GAVE and rule guards."""
    return {
        "historyBids": [],
        "historyAuctionResult": [],
        "historyImpressionResult": [],
        "historyLeastWinningCost": [],
        "historyPValueInfo": [],
    }


def update_history(history: Dict[str, list], p_values: Any, p_value_sigmas: Any, bid: Any, tick_status: Any, tick_cost: Any, tick_conversion: Any, least_winning_cost: Any) -> None:
    """把一个 timestep 的模拟竞价结果追加到 rollout 历史中。

English:
Append one timestep of simulated auction outcomes to the rollout history."""
    import numpy as np

    history["historyPValueInfo"].append(np.array([(p_values[i], p_value_sigmas[i]) for i in range(p_values.shape[0])]))
    history["historyBids"].append(bid)
    history["historyLeastWinningCost"].append(least_winning_cost)
    history["historyAuctionResult"].append(
        np.array([(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
    )
    history["historyImpressionResult"].append(
        np.array([(tick_conversion[i], tick_conversion[i]) for i in range(p_values.shape[0])])
    )


def enforce_budget(bid: Any, p_values: Any, p_value_sigmas: Any, least_winning_cost: Any, remaining_budget: float) -> Tuple[Any, Any, Any, Any]:
    """模拟出价，并随机移除部分竞胜，直到 timestep 成本不超过剩余预算。

English:
Simulate bidding and randomly drop wins until the timestep cost fits the remaining budget."""
    import numpy as np

    tick_value, tick_cost, tick_status, tick_conversion = simulate_ad_bidding(
        p_values, p_value_sigmas, bid, least_winning_cost
    )
    over_cost_ratio = max((float(np.sum(tick_cost)) - remaining_budget) / (float(np.sum(tick_cost)) + 1e-4), 0)
    while over_cost_ratio > 0:
        pv_index = np.where(tick_status == 1)[0]
        if pv_index.shape[0] == 0:
            break
        dropped_count = int(math.ceil(pv_index.shape[0] * over_cost_ratio))
        dropped_pv_index = np.random.choice(pv_index, dropped_count, replace=False)
        bid[dropped_pv_index] = 0
        tick_value, tick_cost, tick_status, tick_conversion = simulate_ad_bidding(
            p_values, p_value_sigmas, bid, least_winning_cost
        )
        over_cost_ratio = max((float(np.sum(tick_cost)) - remaining_budget) / (float(np.sum(tick_cost)) + 1e-4), 0)
    return tick_value, tick_cost, tick_status, tick_conversion


def evaluate(
    strategy: str,
    data_loader: TrafficDataLoader,
    max_keys: Optional[int],
    gave_tool: GaveTool,
    sql_tool: Optional[SqlInsightTool],
    adjustment_agent: Optional[AdjustmentAnalysisAgent],
    final_agent: Optional[FinalBidAgent],
    seed: int,
    print_trace: bool = True,
    live_trace_path: Optional[Path] = None,
    llm_decision_interval: int = 1,
    llm_skip_policy: str = "reuse_last",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """对 GAVE 或 LLM-agent 策略运行一次完整离线评估。

该函数会遍历广告主 episode 和 timestep，让 GAVE 给出基础出价；
如果是 agent 策略，则进一步调用 LLM 决策链做调价；随后模拟竞胜、成本和转化，
最终返回逐广告主结果、period 汇总指标和逐 timestep 决策 trace。

English:
Run one complete offline evaluation pass for GAVE or an LLM-agent strategy.

The function iterates advertiser episodes and timesteps, asks GAVE for base bids,
optionally asks the LLM agent chain for an adjustment, simulates auction wins/costs,
and returns row-level results, a period-level summary, and decision traces."""
    import numpy as np

    # 规范化 interval 设置，方便后续逻辑直接使用。
    # Normalize interval settings so downstream logic can assume valid values.
    llm_decision_interval = max(int(llm_decision_interval), 1)
    if llm_skip_policy not in {"reuse_last", "gave_between"}:
        raise ValueError(f"Unsupported llm_skip_policy: {llm_skip_policy}")
    set_seed(seed)
    results = []
    traces = []
    overall_score = 0.0
    overall_score1 = 0.0
    overall_conversion = 0.0
    exceed_count = 0
    total_cost_sum = 0.0
    evaluated_keys = list(data_loader.iter_keys(max_keys))

    for key in evaluated_keys:
        # 每个 key 对应测试 period 内的一个广告主 episode。
        # Each key is one advertiser episode within the selected test period.
        delivery_period_index, advertiser_number = key
        num_steps, p_values_list, p_value_sigmas_list, least_costs_list, budget, cpa, category = data_loader.mock_data(key)
        budget *= 1.0
        remaining_budget = budget
        rewards = np.zeros(num_steps)
        expected_rewards = np.zeros(num_steps)
        history = empty_history()
        episode_decision_memory: List[Dict[str, Any]] = []
        last_interval_decision: Optional[AdjustmentDecision] = None
        last_interval_sql_summary: Optional[Dict[str, Any]] = None

        for time_step_index in range(num_steps):
            # 一个 timestep 内包含该广告主面对的大量 PV/曝光机会。
            # One timestep contains many PV/impression opportunities for the advertiser.
            step_timer = time.perf_counter()
            trace_row: Optional[Dict[str, Any]] = None
            p_values = p_values_list[time_step_index]
            p_value_sigmas = p_value_sigmas_list[time_step_index]
            least_winning_cost = least_costs_list[time_step_index]

            if remaining_budget < 0.1:
                # 当剩余预算几乎耗尽时停止出价。
                # Stop bidding once the episode is effectively out of budget.
                bid = np.zeros(p_values.shape[0])
                base_alpha = 0.0
                final_alpha = 0.0
            elif strategy == "gave":
                # Baseline 路径：直接使用 GAVE alpha，不做 SQL/LLM 调价。
                # Baseline path: use GAVE alpha directly, without SQL or LLM adjustment.
                base_alpha, bid = gave_tool.bid(time_step_index, p_values, budget, cpa, remaining_budget, history)
                final_alpha = base_alpha
                final_source = "gave"
            else:
                assert sql_tool is not None and adjustment_agent is not None and final_agent is not None
                # Agent 路径：每 N 个 timestep 调用一次 LLM，中间步复用策略或回退到 GAVE。
                # Agent path: call LLM every N timesteps, then either reuse strategy or fall back to GAVE between calls.
                should_call_llm = (
                    llm_decision_interval <= 1
                    or time_step_index % llm_decision_interval == 0
                    or last_interval_decision is None
                    or last_interval_sql_summary is None
                )
                sql_summary: Dict[str, Any] = {}
                segment_strategy: Dict[str, Any] = {}
                if should_call_llm:
                    # GAVE 推理和 KB 查询互不依赖，因此并行执行。
                    # GAVE inference and KB lookup are independent, so run them in parallel.
                    sql_gave_timer = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        gave_future = executor.submit(
                            gave_tool.bid, time_step_index, p_values, budget, cpa, remaining_budget, history
                        )
                        sql_future = executor.submit(
                            sql_tool.summarize,
                            advertiser_number,
                            category,
                            time_step_index,
                            float(np.mean(p_values)) if len(p_values) else 0.0,
                            int(len(p_values)),
                            float(budget),
                            float(np.min(p_values)) if len(p_values) else 0.0,
                            float(np.quantile(p_values, 0.25)) if len(p_values) else 0.0,
                            float(np.quantile(p_values, 0.50)) if len(p_values) else 0.0,
                            float(np.quantile(p_values, 0.75)) if len(p_values) else 0.0,
                            float(np.quantile(p_values, 0.90)) if len(p_values) else 0.0,
                            float(np.max(p_values)) if len(p_values) else 0.0,
                        )
                        base_alpha, base_bids = gave_future.result()
                        sql_summary = sql_future.result()
                    sql_gave_elapsed = time.perf_counter() - sql_gave_timer
                    print(
                        "[timing] step_prepare "
                        f"advertiser={advertiser_number} timestep={time_step_index} "
                        f"gave_plus_sql={sql_gave_elapsed:.2f}s call_llm=True",
                        flush=True,
                    )
                    attach_current_segment_mix(sql_summary, p_values)
                else:
                    # 跳过 LLM 的 timestep 仍需基于当前状态/历史重新计算 GAVE base alpha。
                    # Skipped timesteps still need a fresh GAVE base alpha for current state/history.
                    sql_gave_timer = time.perf_counter()
                    base_alpha, base_bids = gave_tool.bid(time_step_index, p_values, budget, cpa, remaining_budget, history)
                    sql_gave_elapsed = time.perf_counter() - sql_gave_timer
                    print(
                        "[timing] step_prepare "
                        f"advertiser={advertiser_number} timestep={time_step_index} "
                        f"gave_only={sql_gave_elapsed:.2f}s call_llm=False policy={llm_skip_policy}",
                        flush=True,
                    )
                current_state = state_to_dict(
                    # 获得当前基础动作后，重建状态用于 prompt 和规则护栏。
                    # Rebuild the state for prompts/guards after obtaining current base action.
                    build_gave_state(
                        time_step_index=time_step_index,
                        p_values=p_values,
                        budget=budget,
                        remaining_budget=remaining_budget,
                        history=history,
                    )
                )
                cost_so_far = budget - remaining_budget
                conversions_so_far = float(np.sum(rewards[:time_step_index]))
                expected_conversions_so_far = float(np.sum(expected_rewards[:time_step_index]))
                current_state["score_status"] = build_score_status(
                    cost_so_far=cost_so_far,
                    conversions_so_far=conversions_so_far,
                    expected_conversions_so_far=expected_conversions_so_far,
                    cpa_constraint=cpa,
                )
                current_state["score_optimization_hint"] = (
                    "Primary score rewards conversions with beta=2 CPA penalty; score1 uses beta=5 and punishes CPA "
                    "overrun more strongly. Prefer changes that improve score without causing severe score1 loss."
                )
                decision_memory = build_decision_memory(episode_decision_memory)
                decide_timer = time.perf_counter()
                if should_call_llm:
                    # 调价 agent 会运行市场、预算、成功经验三个子 agent，并合并结果。
                    # The adjustment agent runs market, budget, and success sub-agents, then merges them.
                    decision = adjustment_agent.decide(
                        base_alpha=base_alpha,
                        sql_summary=sql_summary,
                        time_step_index=time_step_index,
                        budget=budget,
                        cpa=cpa,
                        remaining_budget=remaining_budget,
                        history=history,
                        current_state=current_state,
                        decision_memory=decision_memory,
                    )
                    last_interval_decision = decision
                    last_interval_sql_summary = sql_summary
                elif llm_skip_policy == "gave_between":
                    # 在两次 LLM 调用之间，该策略忽略上次策略，直接使用原始 GAVE。
                    # Between LLM calls, this policy ignores the last strategy and uses raw GAVE.
                    decision = AdjustmentDecision(
                        "hold",
                        adjustment_to_range(0.0),
                        0.0,
                        f"Skipped LLM on interval={llm_decision_interval}; used raw GAVE bid for this timestep.",
                        {
                            "lower_quartile": 1.0,
                            "standard_value": 1.0,
                            "strong_value": 1.0,
                            "premium_value": 1.0,
                        },
                        "interval_skip_gave",
                    )
                else:
                    # 在两次 LLM 调用之间，该策略复用上一次 LLM 的调价和 multiplier。
                    # Between LLM calls, this policy reuses the previous LLM adjustment/multipliers.
                    assert last_interval_decision is not None and last_interval_sql_summary is not None
                    decision = AdjustmentDecision(
                        last_interval_decision.direction,
                        last_interval_decision.adjustment_range,
                        last_interval_decision.adjustment,
                        (
                            f"Skipped LLM on interval={llm_decision_interval}; reused previous LLM strategy. "
                            + last_interval_decision.reason
                        )[:360],
                        dict(last_interval_decision.pvalue_multipliers),
                        "interval_reuse_last_llm",
                    )
                    sql_summary = last_interval_sql_summary
                decide_elapsed = time.perf_counter() - decide_timer
                if decision.adjustment < MIN_ADJUSTMENT or decision.adjustment > MAX_ADJUSTMENT:
                    print(
                        f"[demo] Invalid adjustment after LLM repair at advertiser={advertiser_number}, "
                        f"timeStep={time_step_index}: {decision.adjustment}. Using safe hold fallback.",
                        flush=True,
                    )
                    decision = AdjustmentDecision(
                        "hold",
                        adjustment_to_range(0.0),
                        0.0,
                        "Invalid adjustment after repair; safe hold fallback used before bidding.",
                        decision.pvalue_multipliers,
                        "local_safety_fallback",
                    )
                finalize_timer = time.perf_counter()
                if not should_call_llm and llm_skip_policy == "gave_between":
                    # raw-GAVE 间隔路径不执行分层 bid 逻辑。
                    # Raw-GAVE interval path bypasses segmented bid execution.
                    final_alpha = base_alpha
                    bid = base_bids
                    final_source = "gave_between_llm_interval"
                    segment_strategy = {
                        "type": "raw_gave_between_llm_calls",
                        "multipliers": decision.pvalue_multipliers,
                        "bid_summary": {"note": "raw GAVE bid used; no segmented executor on skipped step"},
                    }
                else:
                    # 常规 agent 路径把 alpha 调整和分层 multiplier 转为逐 PV bid。
                    # Normal agent path converts alpha adjustment and segment multipliers into PV-level bids.
                    final_alpha, bid, final_source, segment_strategy = final_agent.finalize(
                        base_alpha,
                        base_bids,
                        decision,
                        sql_summary=sql_summary,
                        budget=budget,
                        remaining_budget=remaining_budget,
                        cpa=cpa,
                        time_step_index=time_step_index,
                        p_values=p_values,
                    )
                finalize_elapsed = time.perf_counter() - finalize_timer
                if final_alpha < 0:
                    print(
                        f"[demo] Invalid final_alpha at advertiser={advertiser_number}, timeStep={time_step_index}: "
                        f"{final_alpha}. Replacing bid with zero for this step.",
                        flush=True,
                    )
                    final_alpha = 0.0
                    bid = np.zeros(p_values.shape[0])
                    final_source = "local_safety_zero_bid"
                trace_row = {
                    # trace 行是解释每次 agent 决策的主要审计记录。
                    # Trace rows are the main audit artifact for explaining each agent decision.
                    "deliveryPeriodIndex": delivery_period_index,
                    "advertiserNumber": advertiser_number,
                    "timeStepIndex": time_step_index,
                    "base_alpha": round(base_alpha, 8),
                    "adjustment": decision.adjustment,
                    "direction": decision.direction,
                    "adjustment_range": decision.adjustment_range,
                    "final_alpha": round(final_alpha, 8),
                    "reason": decision.reason,
                    "strategy": strategy,
                    "decision_source": decision.source,
                    "final_source": final_source,
                    "segment_strategy_json": json.dumps(segment_strategy, ensure_ascii=False, sort_keys=True),
                    "value_segment": sql_summary.get("pvalue_context", {})
                    .get("decision_basis", {})
                    .get("segment", "unknown"),
                    "pvalue_basis_scope": sql_summary.get("pvalue_context", {})
                    .get("decision_basis", {})
                    .get("scope_policy", "all_scopes"),
                    "success_ref_alpha": sql_summary.get("success_context", {})
                    .get("reference_alpha_summary", {})
                    .get("advertiser_timestep_wins", 0.0),
                    "success_ref_by_scope_json": json.dumps(
                        success_reference_by_scope(sql_summary), ensure_ascii=False, sort_keys=True
                    ),
                    "phase_json": json.dumps(delivery_phase_policy(time_step_index), ensure_ascii=False, sort_keys=True),
                    "state_json": json.dumps(current_state, ensure_ascii=False, sort_keys=True),
                    "decision_memory_json": json.dumps(decision_memory, ensure_ascii=False, sort_keys=True),
                    "sql_summary_json": json.dumps(compact_sql_trace_summary(sql_summary), ensure_ascii=False, sort_keys=True) if sql_summary else "{}",
                    "outcome_json": "{}",
                }
                print(
                    "[timing] step_decision "
                    f"advertiser={advertiser_number} timestep={time_step_index} "
                    f"decide={decide_elapsed:.2f}s finalize={finalize_elapsed:.2f}s",
                    flush=True,
                )
                traces.append(trace_row)

            _, tick_cost, tick_status, tick_conversion = enforce_budget(
                # 用测试 period 的 leastWinningCost 评估当前 bid 向量是否竞胜。
                # Evaluate the chosen bid vector against the test-period leastWinningCost.
                bid, p_values, p_value_sigmas, least_winning_cost, remaining_budget
            )
            remaining_budget -= float(np.sum(tick_cost))
            rewards[time_step_index] = float(np.sum(tick_conversion))
            expected_rewards[time_step_index] = float(np.sum(p_values[tick_status]))
            update_history(history, p_values, p_value_sigmas, bid, tick_status, tick_cost, tick_conversion, least_winning_cost)
            if trace_row is not None:
                # 模拟结果产生后，把真实成本/转化反馈写入记忆和实时 trace。
                # After outcome simulation, attach realized cost/conversion feedback for memory and live trace.
                step_cost = float(np.sum(tick_cost))
                step_conversions = float(np.sum(tick_conversion))
                step_expected_conversions = float(expected_rewards[time_step_index])
                step_wins = int(np.sum(tick_status))
                cumulative_cost = budget - remaining_budget
                cumulative_conversions = float(np.sum(rewards[: time_step_index + 1]))
                cumulative_expected_conversions = float(np.sum(expected_rewards[: time_step_index + 1]))
                cpa_so_far_after = safe_div(cumulative_cost, cumulative_conversions, 0.0)
                expected_cpa_so_far_after = safe_div(cumulative_cost, cumulative_expected_conversions, 0.0)
                outcome = {
                    "cost": step_cost,
                    "win_count": step_wins,
                    "conversions": step_conversions,
                    "expected_conversions": step_expected_conversions,
                    "expected_cpa": safe_div(step_cost, step_expected_conversions, 0.0),
                    "remaining_budget_after": remaining_budget,
                    "remaining_budget_ratio_after": safe_div(remaining_budget, budget, 0.0),
                    "cumulative_cost_after": cumulative_cost,
                    "cumulative_conversions_after": cumulative_conversions,
                    "cumulative_expected_conversions_after": cumulative_expected_conversions,
                    "cpa_so_far_after": cpa_so_far_after,
                    "expected_cpa_so_far_after": expected_cpa_so_far_after,
                    "score_so_far_after": get_score(cumulative_conversions, cpa_so_far_after, cpa, beta=2),
                    "score1_so_far_after": get_score(cumulative_conversions, cpa_so_far_after, cpa, beta=5),
                    "expected_score_so_far_after": get_score(
                        cumulative_expected_conversions, expected_cpa_so_far_after, cpa, beta=2
                    ),
                    "expected_score1_so_far_after": get_score(
                        cumulative_expected_conversions, expected_cpa_so_far_after, cpa, beta=5
                    ),
                }
                trace_row["outcome_json"] = json.dumps(outcome, ensure_ascii=False, sort_keys=True)
                episode_decision_memory.append(
                    {
                        "timeStepIndex": time_step_index,
                        "value_segment": trace_row["value_segment"],
                        "base_alpha": base_alpha,
                        "adjustment": decision.adjustment,
                        "final_alpha": final_alpha,
                        "multipliers": segment_strategy.get("multipliers", {}),
                        "reason": decision.reason,
                        "cost": step_cost,
                        "win_count": step_wins,
                        "conversions": step_conversions,
                        "expected_conversions": step_expected_conversions,
                        "remaining_budget_ratio_after": outcome["remaining_budget_ratio_after"],
                        "cpa_so_far_after": outcome["cpa_so_far_after"],
                        "expected_cpa_so_far_after": outcome["expected_cpa_so_far_after"],
                        "score_so_far_after": outcome["score_so_far_after"],
                        "score1_so_far_after": outcome["score1_so_far_after"],
                        "expected_score_so_far_after": outcome["expected_score_so_far_after"],
                        "expected_score1_so_far_after": outcome["expected_score1_so_far_after"],
                        "cpa_constraint": cpa,
                    }
                )
                if print_trace:
                    print_decision_trace(trace_row)
                if live_trace_path is not None:
                    trace_write_timer = time.perf_counter()
                    append_csv_row(live_trace_path, trace_row, DECISION_TRACE_FIELDNAMES)
                    print(
                        "[timing] step_total "
                        f"advertiser={advertiser_number} timestep={time_step_index} "
                        f"total={time.perf_counter() - step_timer:.2f}s "
                        f"trace_write={time.perf_counter() - trace_write_timer:.2f}s",
                        flush=True,
                    )

        all_reward = float(np.sum(rewards))
        # 将广告主 episode 转换为 GAVE 环境使用的同款 score 指标。
        # Convert the advertiser episode into the same score metrics used by the GAVE environment.
        all_cost = budget - remaining_budget
        cpa_real = all_cost / (all_reward + 1e-10)
        score = get_score(all_reward, cpa_real, cpa, beta=2)
        score1 = get_score(all_reward, cpa_real, cpa, beta=5)
        exceeded = int(cpa_real > cpa)
        overall_score += score
        overall_score1 += score1
        overall_conversion += all_reward
        exceed_count += exceeded
        total_cost_sum += all_cost
        results.append(
            {
                "strategy": strategy,
                "deliveryPeriodIndex": delivery_period_index,
                "advertiserNumber": advertiser_number,
                "score": score,
                "score1": score1,
                "conversion": all_reward,
                "exceed": exceeded,
                "total_cost": all_cost,
                "cpa_real": cpa_real,
                "cpa_constraint": cpa,
                "budget": budget,
                "final_remaining_budget": remaining_budget,
            }
        )

    key_count = max(len(evaluated_keys), 1)
    summary = {
        "strategy": strategy,
        "period_score": overall_score / key_count,
        "period_score1": overall_score1 / key_count,
        "period_conversion": overall_conversion / key_count,
        "exceed_rate": exceed_count / key_count,
        "total_cost": total_cost_sum,
        "evaluated_keys": len(evaluated_keys),
    }
    return results, summary, traces


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """写入完整 CSV；即使没有数据行也会写 header。

English:
Write a complete CSV file, including a header even when there are no rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


DECISION_TRACE_FIELDNAMES = [
    "deliveryPeriodIndex",
    "advertiserNumber",
    "timeStepIndex",
    "base_alpha",
    "adjustment",
    "direction",
    "adjustment_range",
    "final_alpha",
    "reason",
    "strategy",
    "decision_source",
    "final_source",
    "segment_strategy_json",
    "value_segment",
    "pvalue_basis_scope",
    "success_ref_alpha",
    "success_ref_by_scope_json",
    "phase_json",
    "state_json",
    "decision_memory_json",
    "sql_summary_json",
    "outcome_json",
]


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """立即追加一行 trace，保证长时间 LLM 运行中断后仍有可恢复输出。

English:
Append one trace row immediately so long LLM runs still leave recoverable output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """把 CSV 读成字典列表；文件缺失或为空时返回空列表。

English:
Read a CSV into dictionaries, returning an empty list for missing/empty files."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_or_run_gave_baseline(
    args: argparse.Namespace,
    checkpoint: Path,
    model_param: Dict[str, Any],
    data_loader: TrafficDataLoader,
    max_keys: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    """加载缓存的 GAVE baseline 指标；没有缓存时运行一次确定性 baseline。

English:
Load cached GAVE baseline metrics or run the deterministic baseline once."""
    result_cache, summary_cache = baseline_cache_paths(args, checkpoint, model_param)
    if result_cache.exists() and summary_cache.exists() and not args.force_baseline:
        print(f"[demo] Loading cached GAVE baseline: {result_cache}", flush=True)
        rows = read_csv_rows(result_cache)
        with open(summary_cache, "r", encoding="utf-8") as file:
            summary = json.load(file)
        return rows, summary, True

    print("[demo] Running GAVE baseline evaluation; result will be cached for identical settings.", flush=True)
    baseline_tool = GaveTool(checkpoint, model_param)
    baseline_results, baseline_summary, _ = evaluate(
        strategy="gave",
        data_loader=data_loader,
        max_keys=max_keys,
        gave_tool=baseline_tool,
        sql_tool=None,
        adjustment_agent=None,
        final_agent=None,
        seed=args.seed,
        print_trace=False,
    )
    result_cache.parent.mkdir(parents=True, exist_ok=True)
    write_csv(result_cache, baseline_results)
    with open(summary_cache, "w", encoding="utf-8") as file:
        json.dump(baseline_summary, file, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[demo] Cached GAVE baseline: {result_cache}", flush=True)
    return baseline_results, baseline_summary, False


def segment_multipliers_for_pvalues(p_values: Any, segment_strategy: Dict[str, Any]) -> Any:
    """从分层策略 JSON 中重建逐 PV 的 segment multiplier。

English:
Reconstruct PV-level segment multipliers from the segment strategy JSON."""
    import numpy as np

    thresholds = segment_strategy.get("thresholds", {})
    multipliers = segment_strategy.get("multipliers", {}) or {"standard_value": 1.0}
    advertiser_p25 = as_float(thresholds.get("advertiser_won_p25", thresholds.get("advertiser_overall_p25")))
    advertiser_p50 = as_float(thresholds.get("advertiser_won_p50", thresholds.get("advertiser_overall_p50")))
    advertiser_p75 = as_float(thresholds.get("advertiser_won_p75", thresholds.get("advertiser_overall_p75")))
    current_timestep_p25 = as_float(thresholds.get("current_timestep_p25", thresholds.get("global_timestep_p25")))
    current_timestep_p50 = as_float(thresholds.get("current_timestep_p50", thresholds.get("global_timestep_p50")))
    current_timestep_p75 = as_float(thresholds.get("current_timestep_p75", thresholds.get("global_timestep_p75")))
    premium_mask = np.zeros(p_values.shape[0], dtype=bool)
    strong_or_premium_mask = np.zeros(p_values.shape[0], dtype=bool)
    standard_or_above_mask = np.zeros(p_values.shape[0], dtype=bool)
    if advertiser_p75 > 0:
        premium_mask = premium_mask | (p_values >= advertiser_p75)
    if current_timestep_p75 > 0:
        premium_mask = premium_mask | (p_values >= current_timestep_p75)
    if advertiser_p50 > 0:
        strong_or_premium_mask = strong_or_premium_mask | (p_values >= advertiser_p50)
    if current_timestep_p50 > 0:
        strong_or_premium_mask = strong_or_premium_mask | (p_values >= current_timestep_p50)
    if advertiser_p25 > 0:
        standard_or_above_mask = standard_or_above_mask | (p_values >= advertiser_p25)
    if current_timestep_p25 > 0:
        standard_or_above_mask = standard_or_above_mask | (p_values >= current_timestep_p25)
    strong_mask = strong_or_premium_mask & ~premium_mask
    standard_mask = standard_or_above_mask & ~(premium_mask | strong_mask)
    lower_mask = ~(premium_mask | strong_mask | standard_mask)
    multiplier_array = np.ones(p_values.shape[0], dtype=float)
    default_multiplier = as_float(multipliers.get("standard_value"), 1.0)
    multiplier_array[lower_mask] = as_float(multipliers.get("lower_quartile"), default_multiplier)
    multiplier_array[standard_mask] = default_multiplier
    multiplier_array[strong_mask] = as_float(multipliers.get("strong_value"), default_multiplier)
    multiplier_array[premium_mask] = as_float(multipliers.get("premium_value"), default_multiplier)
    return multiplier_array


def write_partial_counterfactual_comparison(path: Path, trace_rows: List[Dict[str, Any]], data_loader: TrafficDataLoader) -> None:
    """不重新采样随机转化，估计 traced timesteps 上 base 与 agent 的期望结果。

English:
Estimate base-vs-agent expected outcomes for traced timesteps without resimulating randomness."""
    import numpy as np
    import pandas as pd

    metric_rows = []
    for trace in trace_rows:
        try:
            key = (float(trace["deliveryPeriodIndex"]), float(trace["advertiserNumber"]))
            time_step_index = int(float(trace["timeStepIndex"]))
            data = data_loader.test_dict[key]
            step_data = data.loc[data["timeStepIndex"].astype(int) == time_step_index]
            if step_data.empty:
                continue
            p_values = step_data["pValue"].to_numpy(dtype=float)
            least_winning_cost = step_data["leastWinningCost"].to_numpy(dtype=float)
            cpa = float(step_data["CPAConstraint"].iloc[0])
            base_alpha = as_float(trace.get("base_alpha"))
            final_alpha = as_float(trace.get("final_alpha"))
            trace_strategy = str(trace.get("strategy") or "agent")
            segment_strategy = json.loads(trace.get("segment_strategy_json", "{}") or "{}")
            outcome = json.loads(trace.get("outcome_json", "{}") or "{}")

            base_wins = base_alpha * p_values >= least_winning_cost
            agent_multipliers = segment_multipliers_for_pvalues(p_values, segment_strategy)
            agent_wins = final_alpha * p_values * agent_multipliers >= least_winning_cost
            metric_rows.extend(
                [
                    {
                        "advertiserNumber": key[1],
                        "strategy": "base_counterfactual",
                        "cpa_constraint": cpa,
                        "traced_steps": 1,
                        "win_count": int(np.sum(base_wins)),
                        "total_cost": float(np.sum(least_winning_cost[base_wins])),
                        "simulated_conversion": 0.0,
                        "expected_conversion": float(np.sum(p_values[base_wins])),
                    },
                    {
                        "advertiserNumber": key[1],
                        "strategy": f"{trace_strategy}_counterfactual",
                        "cpa_constraint": cpa,
                        "traced_steps": 1,
                        "win_count": int(np.sum(agent_wins)),
                        "total_cost": float(np.sum(least_winning_cost[agent_wins])),
                        "simulated_conversion": 0.0,
                        "expected_conversion": float(np.sum(p_values[agent_wins])),
                    },
                    {
                        "advertiserNumber": key[1],
                        "strategy": f"{trace_strategy}_actual_trace",
                        "cpa_constraint": cpa,
                        "traced_steps": 1,
                        "win_count": int(as_float(outcome.get("win_count"))),
                        "total_cost": as_float(outcome.get("cost")),
                        "simulated_conversion": as_float(outcome.get("conversions")),
                        "expected_conversion": as_float(outcome.get("expected_conversions")),
                    },
                ]
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    if not metric_rows:
        write_csv(path, [], [
            "scope", "advertiserNumber", "strategy", "traced_steps", "win_count", "total_cost",
            "simulated_conversion", "expected_conversion", "expected_cpa", "expected_score",
            "expected_score1",
        ])
        return

    metrics = pd.DataFrame(metric_rows)
    summary_rows: List[Dict[str, Any]] = []
    for (advertiser, strategy), group in metrics.groupby(["advertiserNumber", "strategy"], sort=True):
        cost = float(group["total_cost"].sum())
        expected_conversion = float(group["expected_conversion"].sum())
        simulated_conversion = float(group["simulated_conversion"].sum())
        cpa_constraint = float(group["cpa_constraint"].iloc[0])
        expected_cpa = safe_div(cost, expected_conversion, 0.0)
        summary_rows.append(
            {
                "scope": "advertiser",
                "advertiserNumber": advertiser,
                "strategy": strategy,
                "traced_steps": int(group["traced_steps"].sum()),
                "win_count": int(group["win_count"].sum()),
                "total_cost": cost,
                "simulated_conversion": simulated_conversion,
                "expected_conversion": expected_conversion,
                "expected_cpa": expected_cpa,
                "expected_score": get_score(expected_conversion, expected_cpa, cpa_constraint, beta=2),
                "expected_score1": get_score(expected_conversion, expected_cpa, cpa_constraint, beta=5),
            }
        )

    for strategy in sorted(metrics["strategy"].unique()):
        rows = [row for row in summary_rows if row["strategy"] == strategy and row["scope"] == "advertiser"]
        summary_rows.append(
            {
                "scope": "all_sum",
                "advertiserNumber": "all",
                "strategy": strategy,
                "traced_steps": sum(int(row["traced_steps"]) for row in rows),
                "win_count": sum(int(row["win_count"]) for row in rows),
                "total_cost": sum(float(row["total_cost"]) for row in rows),
                "simulated_conversion": sum(float(row["simulated_conversion"]) for row in rows),
                "expected_conversion": sum(float(row["expected_conversion"]) for row in rows),
                "expected_cpa": safe_div(
                    sum(float(row["total_cost"]) for row in rows),
                    sum(float(row["expected_conversion"]) for row in rows),
                    0.0,
                ),
                "expected_score": sum(float(row["expected_score"]) for row in rows),
                "expected_score1": sum(float(row["expected_score1"]) for row in rows),
            }
        )

    write_csv(
        path,
        summary_rows,
        [
            "scope",
            "advertiserNumber",
            "strategy",
            "traced_steps",
            "win_count",
            "total_cost",
            "simulated_conversion",
            "expected_conversion",
            "expected_cpa",
            "expected_score",
            "expected_score1",
        ],
    )


def print_decision_trace(trace_row: Dict[str, Any]) -> None:
    """向终端打印一条紧凑、可读的决策 trace。

English:
Print one compact human-readable decision trace to the terminal."""
    segment_strategy = json.loads(trace_row.get("segment_strategy_json", "{}") or "{}")
    phase = json.loads(trace_row.get("phase_json", "{}") or "{}")
    memory = json.loads(trace_row.get("decision_memory_json", "{}") or "{}")
    outcome = json.loads(trace_row.get("outcome_json", "{}") or "{}")
    success_refs = json.loads(trace_row.get("success_ref_by_scope_json", "{}") or "{}")
    multipliers = segment_strategy.get("multipliers", {})
    bid_summary = segment_strategy.get("bid_summary", {})
    display = {
        "period": trace_row["deliveryPeriodIndex"],
        "advertiser": trace_row["advertiserNumber"],
        "timeStep": trace_row["timeStepIndex"],
        "value_segment": trace_row["value_segment"],
        "phase": phase.get("phase", "unknown"),
        "scope": trace_row["pvalue_basis_scope"],
        "threshold_scope": segment_strategy.get("threshold_scope", "unknown"),
        "base_alpha": trace_row["base_alpha"],
        "adjustment": trace_row["adjustment"],
        "direction": trace_row["direction"],
        "final_alpha": trace_row["final_alpha"],
        "multipliers": multipliers,
        "bid_segment_counts": {
            "lower_quartile": bid_summary.get("lower_quartile_count", 0),
            "standard_value": bid_summary.get("standard_value_count", 0),
            "strong_value": bid_summary.get("strong_value_count", 0),
            "premium_value": bid_summary.get("premium_value_count", 0),
        },
        "success_ref_alpha": trace_row["success_ref_alpha"],
        "success_ref_by_scope": success_refs,
        "outcome": {
            "cost": outcome.get("cost", 0.0),
            "win_count": outcome.get("win_count", 0),
            "conversions": outcome.get("conversions", 0.0),
            "expected_conversions": outcome.get("expected_conversions", 0.0),
            "expected_cpa": outcome.get("expected_cpa", 0.0),
            "remaining_budget_ratio_after": outcome.get("remaining_budget_ratio_after", 0.0),
            "cpa_so_far_after": outcome.get("cpa_so_far_after", 0.0),
            "expected_cpa_so_far_after": outcome.get("expected_cpa_so_far_after", 0.0),
            "score_so_far_after": outcome.get("score_so_far_after", 0.0),
            "expected_score_so_far_after": outcome.get("expected_score_so_far_after", 0.0),
        },
        "memory_risk_flags": memory.get("memory_risk_flags", []),
        "memory_advice": memory.get("followup_advice", ""),
        "reason": trace_row["reason"],
    }
    print("[decision_trace] " + json.dumps(display, ensure_ascii=False), flush=True)


def validate_outputs(sql_tool: SqlInsightTool, args: argparse.Namespace, traces: List[Dict[str, Any]]) -> None:
    """对缓存行数和策略边界做轻量级 sanity check。

English:
Run lightweight sanity checks for cache row counts and strategy bounds."""
    actual_rows = sql_tool.row_count()
    expected_rows = sql_tool.source_rows_loaded()
    if actual_rows != expected_rows:
        raise RuntimeError(f"SQLite row count mismatch: sqlite={actual_rows}, loaded_source_rows={expected_rows}")
    for trace in traces:
        adjustment = float(trace["adjustment"])
        final_alpha = float(trace["final_alpha"])
        if adjustment < MIN_ADJUSTMENT or adjustment > MAX_ADJUSTMENT:
            print(f"[demo] Warning: trace contains out-of-range adjustment after run: {adjustment}", flush=True)
        if final_alpha < 0:
            print(f"[demo] Warning: trace contains negative final_alpha after run: {final_alpha}", flush=True)


def parse_args() -> argparse.Namespace:
    """定义数据选择、LLM 调用、GAVE 参数和输出行为相关 CLI 选项。

English:
Define CLI options for data selection, LLM calls, GAVE params, and output behavior."""
    parser = argparse.ArgumentParser(description="V2 multi-LLM auto-bidding agent demo.")
    parser.add_argument("--mode", choices=["rtbagent-llm", "llm"], default="llm")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-keys", type=parse_max_keys, default=parse_max_keys("8"))
    parser.add_argument("--max-rows", type=int, default=None, help="Evaluate only the first N traffic rows from test CSV.")
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY_CSV)
    parser.add_argument("--sql-csv", type=Path, default=DEFAULT_SQL_CSV)
    parser.add_argument(
        "--experience-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Traffic CSVs used as experience SQL sources. Defaults to --sql-csv, or all non-test final periods when --test-period is set.",
    )
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--test-period", type=int, default=13, help="Use final_aigb traffic/period-N.csv as test and all other final periods as experience.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_FULL_GAVE_CHECKPOINT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--llm-config-json", type=str, default=None)
    parser.add_argument("--llm-config-file", type=Path, default=None)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-retry-delay", type=float, default=5.0)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm-model", type=str, default="qwen3.5-flash")
    parser.add_argument("--min-agent-alpha", type=float, default=0.0)
    parser.add_argument(
        "--insight-backend",
        choices=["kb", "sql"],
        default="kb",
        help="Use preaggregated experience knowledge base (`kb`) or raw SQLite summaries (`sql`).",
    )
    parser.add_argument(
        "--force-kb-build",
        action="store_true",
        help="Rebuild the preaggregated experience knowledge base before evaluation.",
    )
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Re-run GAVE baseline instead of loading the cached baseline for identical settings.",
    )
    parser.add_argument(
        "--llm-decision-interval",
        type=int,
        default=4,
        help="Call LLM once every N timesteps for agent strategies. Default 4.",
    )
    parser.add_argument(
        "--interval-policy",
        choices=["both", "reuse_last", "gave_between"],
        default="both",
        help="How to handle timesteps between LLM calls. `both` runs both strategies for comparison.",
    )
    parser.add_argument("--no-print-trace", action="store_true", help="Disable per-decision console trace output.")
    parser.add_argument(
        "--no-print-agent-outputs",
        action="store_true",
        help="Disable console-only multi-agent LLM output printing. These outputs are never written to trace CSV.",
    )
    parser.add_argument(
        "--full-sql-cache",
        action="store_true",
        help="Compatibility flag; the demo now always builds SQLite from all experience CSVs so global references are correct.",
    )

    parser.add_argument("--step-num", type=int, default=1000)
    parser.add_argument("--save-step", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--time-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--expectile", type=float, default=0.99)
    parser.add_argument("--loss-report", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=16)
    parser.add_argument("--n-inner", type=int, default=1024)
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--n-position", type=int, default=1024)
    parser.add_argument("--resid-pdrop", type=float, default=0.1)
    parser.add_argument("--attn-pdrop", type=float, default=0.1)
    parser.add_argument("--budget-rate", type=float, default=1.0)
    parser.add_argument(
        "--gave-target-return",
        type=float,
        default=0.01,
        help=(
            "Normalized GAVE target return used at inference. Training divides score/reward by 2000, "
            "so values around 0.005-0.02 are in-distribution. Use 4 to reproduce the original default."
        ),
    )
    args = parser.parse_args()
    if args.save_step is None:
        args.save_step = args.step_num
    return args


def apply_test_period_defaults(args: argparse.Namespace) -> None:
    """把 --test-period 转换为具体测试 CSV 和经验 period CSV 路径。

English:
Convert --test-period into concrete test CSV and experience-period CSV paths."""
    if args.test_period is None:
        args.experience_csvs = args.experience_csvs or [args.sql_csv]
        return

    available_periods = available_final_periods()
    if args.test_period not in available_periods:
        raise FileNotFoundError(
            f"--test-period {args.test_period} is not available under {FINAL_AIGB_TRAFFIC_DIR}. "
            f"Available periods: {available_periods}"
        )
    args.test_csv = final_period_csv(args.test_period)
    if args.experience_csvs is None:
        args.experience_csvs = [final_period_csv(period) for period in available_periods if period != args.test_period]
    if args.experience_csvs:
        args.sql_csv = args.experience_csvs[0]


def main() -> None:
    """串联配置、权重加载、KB 构建、评估和结果输出。

English:
Wire together config, checkpoint loading, KB creation, evaluation, and outputs."""
    args = parse_args()
    apply_test_period_defaults(args)
    require_runtime_dependencies()
    add_gave_to_path()
    llm_client = None
    if args.mode == "rtbagent-llm":
        llm_config = load_llm_config(args)
        if not llm_config:
            raise RuntimeError(
                "Mode `rtbagent-llm` requires LLM config via --llm-config-json, --llm-config-file, "
                "or AUTOBID_LLM_CONFIG. Example: "
                "'{\"api_type\":\"openai\",\"api_key\":\"sk-...\",\"base_url\":\"https://api.openai.com/v1\","
                "\"model\":\"gpt-4o-mini\"}'"
            )
        llm_client = RTBAgentLLMClient(llm_config=llm_config, timeout=args.llm_timeout)
    elif args.mode == "llm":
        llm_client = OpenAICompatibleLLMClient(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            timeout=args.llm_timeout,
            retries=args.llm_retries,
            retry_delay=args.llm_retry_delay,
        )

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        trajectory_tag = file_fingerprint(args.trajectory_csv)
        checkpoint_dir = (
            WORKSPACE_ROOT
            / "autobid_demo"
            / "checkpoints"
            / f"gave_h{args.hidden_size}_l{args.n_layer}_steps{args.step_num}_seed{args.seed}_data{trajectory_tag}"
        )
    checkpoint_dir = checkpoint_dir.resolve()
    model_param = build_model_param(args, checkpoint_dir)
    set_seed(args.seed)
    checkpoint = train_or_load_gave(args, model_param, checkpoint_dir)

    output_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    live_trace_path = output_dir / "decision_trace_live.csv"
    partial_comparison_path = output_dir / "partial_counterfactual_comparison.csv"
    print(f"[demo] trajectory_csv: {args.trajectory_csv}")
    print(f"[demo] test_csv: {args.test_csv}")
    print("[demo] experience_csvs:")
    for experience_csv in args.experience_csvs:
        print(f"[demo]   {experience_csv}")
    print(f"[demo] checkpoint_dir: {checkpoint_dir}")
    data_loader = TrafficDataLoader(
        args.test_csv,
        max_keys=args.max_keys,
        chunksize=args.csv_chunksize,
        max_rows=args.max_rows,
    )
    max_keys = args.max_keys
    print(f"[demo] Evaluating {len(list(data_loader.iter_keys(max_keys)))} advertiser keys from {args.test_csv}")
    sql_data_tag = files_fingerprint(args.experience_csvs)
    sql_schema_tag = SqlInsightTool.CACHE_SCHEMA_VERSION
    sql_cache_path = DEFAULT_CACHE_DIR / f"experience_v{sql_schema_tag}_{sql_data_tag}_full.sqlite"
    raw_sql_tool = SqlInsightTool(args.experience_csvs, sql_cache_path, chunksize=args.csv_chunksize)
    if args.insight_backend == "kb":
        kb_cache_path = DEFAULT_CACHE_DIR / f"experience_kb_v{ExperienceKnowledgeTool.KB_SCHEMA_VERSION}_{sql_data_tag}.sqlite"
        sql_tool = ExperienceKnowledgeTool(raw_sql_tool, kb_cache_path, force_rebuild=args.force_kb_build)
        print(f"[demo] insight_backend: kb ({kb_cache_path})", flush=True)
    else:
        sql_tool = raw_sql_tool
        print(f"[demo] insight_backend: sql ({sql_cache_path})", flush=True)

    baseline_results, baseline_summary, baseline_from_cache = load_or_run_gave_baseline(
        args=args,
        checkpoint=checkpoint,
        model_param=model_param,
        data_loader=data_loader,
        max_keys=max_keys,
    )
    print(f"[demo] GAVE baseline source: {'cache' if baseline_from_cache else 'fresh_eval'}", flush=True)
    write_csv(output_dir / "baseline_gave_results.csv", baseline_results)

    if args.mode == "llm":
        agent_strategy_prefix = "llm_agent"
    else:
        agent_strategy_prefix = "rtbagent_llm_agent"
    if args.interval_policy == "both":
        interval_policies = ["reuse_last", "gave_between"]
    else:
        interval_policies = [args.interval_policy]

    traces: List[Dict[str, Any]] = []
    agent_results: List[Dict[str, Any]] = []
    agent_summaries: List[Dict[str, Any]] = []
    run_error: Optional[BaseException] = None
    try:
        for interval_policy in interval_policies:
            agent_tool = GaveTool(checkpoint, model_param)
            if args.llm_decision_interval <= 1:
                strategy_name = f"{agent_strategy_prefix}_every_step"
            else:
                strategy_name = f"{agent_strategy_prefix}_{interval_policy}{args.llm_decision_interval}"
            print(
                f"[demo] Running agent strategy: {strategy_name} "
                f"(interval={args.llm_decision_interval}, policy={interval_policy})",
                flush=True,
            )
            current_results, current_summary, current_traces = evaluate(
                strategy=strategy_name,
                data_loader=data_loader,
                max_keys=max_keys,
                gave_tool=agent_tool,
                sql_tool=sql_tool,
                adjustment_agent=AdjustmentAnalysisAgent(
                    llm_client=llm_client,
                    print_agent_outputs=not args.no_print_agent_outputs,
                ),
                final_agent=FinalBidAgent(min_alpha=args.min_agent_alpha),
                seed=args.seed,
                print_trace=not args.no_print_trace,
                live_trace_path=live_trace_path,
                llm_decision_interval=args.llm_decision_interval,
                llm_skip_policy=interval_policy,
            )
            validate_outputs(sql_tool, args, current_traces)
            agent_results.extend(current_results)
            agent_summaries.append(current_summary)
            traces.extend(current_traces)
        validate_outputs(sql_tool, args, traces)
        write_csv(output_dir / "agent_results.csv", agent_results)
        write_csv(
            output_dir / "comparison.csv",
            [baseline_summary] + agent_summaries,
            ["strategy", "period_score", "period_score1", "period_conversion", "exceed_rate", "total_cost", "evaluated_keys"],
        )
        write_csv(output_dir / "decision_trace.csv", traces, DECISION_TRACE_FIELDNAMES)
    except (KeyboardInterrupt, Exception) as exc:
        run_error = exc
        print(f"[demo] Interrupted or failed during agent evaluation: {type(exc).__name__}: {exc}", flush=True)
    finally:
        live_rows = read_csv_rows(live_trace_path)
        if live_rows:
            if not traces:
                traces = live_rows
            write_csv(output_dir / "decision_trace.csv", live_rows, DECISION_TRACE_FIELDNAMES)
            write_partial_counterfactual_comparison(partial_comparison_path, live_rows, data_loader)
            print(f"[demo] Live trace saved: {live_trace_path}", flush=True)
            print(f"[demo] Partial counterfactual comparison saved: {partial_comparison_path}", flush=True)
        else:
            write_csv(output_dir / "decision_trace.csv", [], DECISION_TRACE_FIELDNAMES)
            write_partial_counterfactual_comparison(partial_comparison_path, [], data_loader)

    print("[demo] Done." if run_error is None else "[demo] Stopped before full completion.")
    print(f"[demo] Outputs: {output_dir}")
    if run_error is None:
        print("[demo] Comparison:")
        for row in [baseline_summary] + agent_summaries:
            print(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        print(f"[demo] Partial outputs are available despite {type(run_error).__name__}.")


if __name__ == "__main__":
    main()
