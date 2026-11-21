# AuctionNet 四算法竞价 Agent

当前可复现的方案：每个广告主在 AuctionNet 官方 48 步环境运行 Episode 0（冷启动并生成经验卡）和 Episode 1（使用冻结的上一轮经验）。每个 Episode 比较 Agent 与官方 BCQ、OnlineLP、IQL、MOPO 四个基线。Agent 每步分析市场价值、预算和经验，调用四个算法并融合出价，经过护栏后交给官方环境执行。每四步生成一张经验卡。

## 本仓库包含

- `autobid_agent/`：Agent、四个工具适配、三项分析 skill、记忆、模型客户端及出价约束。
- `run_autonomous_agent_48x2.py`：当前 48 广告主 × 2 Episode × 5 策略实验入口；可以从空输出目录运行，也可以续跑本仓库自己生成的完成标记。
- `run_autonomous_agent_12x7.py`、`run_autonomous_agent_smoke.py`、`run_auctionnet_agent_vs_gave_episode.py`、`run_auctionnet_official_baselines.py`、`bidding_agent_demo_v2.py`：当前入口所需的实验、官方策略适配及 KB 查询代码。后两个文件含旧 GAVE 实验实现，但本实验入口不加载 GAVE 权重。
- `cache/experience_kb_v1_3fa1b08629.sqlite`：先前从历史训练轨迹聚合的 KB；当前实验默认使用它，不含原始大体量竞价数据。
- `tests/test_autonomous_agent.py`：Agent 核心行为测试。
- `AUTONOMOUS_AGENT.md`：架构、护栏、记忆和 trace 说明。

## 本仓库外的官方依赖

在仓库根目录运行 `./setup_auctionnet.ps1`。脚本克隆 [AuctionNet 官方仓库](https://github.com/alimama-tech/AuctionNet)，固定到 `c20de631a40e24669d1d6a8e2acefd8f492f172d`，并检查官方模型权重。官方代码和 48 个广告主使用的模型均从该仓库取得，不用自制线性模型替代。`third_party/AuctionNet/` 被 Git 忽略。

使用 Python 3.9 环境，先安装官方 `third_party/AuctionNet/requirements.txt` 中兼容的包，以及本实验所需的 `openai`。本地原实验使用 `.conda/auctionnet/python.exe`。PyTorch 等依赖应按机器系统和 CUDA/CPU 情况安装。

设置阿里云百炼密钥，然后执行：

```powershell
$env:DASHSCOPE_API_KEY = '<your-key>'
python -m unittest discover -s tests -p test_autonomous_agent.py
python run_autonomous_agent_48x2.py --output-dir outputs/my_48x2
```

入口固定使用 `qwen3.7-plus-2026-05-26`、seed 2026、50 万 PV、官方 beta=2 score。运行前需要账号有该模型的调用权限。LLM 决策会产生 API 费用；四个官方算法基线本地推理不调用 LLM。完整 480 次策略运行耗时且可能因 API 或网络中断；完成单元会记录在 `completed.json` 中，使用相同输出目录重启会跳过已完成单元。不要把 `outputs/` 和密钥上传。

本包没有包含之前未完成的大规模实验结果，也没有把旧实验进度当作新实验的已完成结果。若代码、KB 或官方版本改变，请使用新的输出目录。详细逐步 trace、模型调用、经验卡、最终分数保存在 `outputs/my_48x2/`。

## 上传到自己的 GitHub

在这个目录运行 `git init`，然后创建一个空 GitHub 仓库，再运行：

```powershell
git add .
git commit -m "Package AuctionNet four-tool bidding agent"
git branch -M main
git remote add origin https://github.com/leah374/<repository-name>.git
git push -u origin main
```

请先在 GitHub 建仓库，并将 `<repository-name>` 换成实际仓库名。每次代码迭代完成、测试通过后再提交和推送；实验产物默认只留在本地。
