# 流水线监控

六阶段流水线一次完整运行需要数天，且无人值守。本页说明如何持续观察它的状态，
以及各类告警分别意味着什么。

## 两个脚本的分工

| 脚本 | 职责 | 适合场景 |
|---|---|---|
| `scripts/snapshot_progress.py` | 刷新 `reports/training-progress.md` 表格 | 想看当前进度数字时 |
| `scripts/monitor_pipeline.py` | 检测**变化与异常**，产出结构化事件 | 无人值守、需要知道「发生了什么」时 |

`snapshot_progress.py` 只回答「现在跑到哪了」；`monitor_pipeline.py` 回答「和上次比，
有什么新情况需要我知道」。持续监控用后者。

## 基本用法

```bash
cd /Users/wlh/Documents/【实践】minimind

# 单次检查，输出 JSON
.venv/bin/python scripts/monitor_pipeline.py

# 只看文件不打印（供定时任务使用）
.venv/bin/python scripts/monitor_pipeline.py --quiet

# 调整停滞阈值（默认 15 分钟）
.venv/bin/python scripts/monitor_pipeline.py --stall-minutes 20
```

### 退出码

| 码 | 含义 |
|---|---|
| `0` | 无异常 |
| `1` | warn 级事件，需要关注但不紧急 |
| `2` | critical 级事件，流水线中断或 loss 非法 |

退出码让上层自动化不必解析 JSON 就能判断是否需要报警。

## 事件类型

| 事件 | 级别 | 含义 |
|---|---|---|
| `stage_started` | info | 某阶段从 pending/ready 变为 active |
| `stage_finished` | info | 某阶段最终 checkpoint 已落盘 |
| `pipeline_finished` | info | 六个阶段全部完成 |
| `pipeline_stalled` | warn | 有 active 阶段但日志长时间无新记录 |
| `pipeline_dead` | critical | 流水线主进程不存在，但仍有未完成阶段 |
| `loss_invalid` | critical | 最近日志出现 NaN / inf loss |

### 关于 `pipeline_stalled` 的误报

本机（MacBook）会休眠，休眠期间训练进程被挂起、日志自然停止刷新，醒来后会继续推进。
这与真正的卡死在日志上无法区分，所以该事件被定为 **warn 而非 critical**，措辞也是
"需确认进程是否仍在推进"。判断方法：再看一次日志步数是否增长。若恢复增长即为休眠，无需处理。

该事件内置 60 分钟报警冷却，避免休眠期间反复触发。

## 状态与事件文件

均在 `artifacts/` 下，Git 忽略：

- `artifacts/monitor-state.json` —— 最近一次观测的完整快照。脚本通过对比它与当前状态
  来识别变化，因此**删除它会丢失「上次是什么状态」的记忆**，下一次运行将不产生
  任何 `stage_started` / `stage_finished` 事件。
- `artifacts/monitor-events.jsonl` —— 追加写入的事件历史，可追溯每次状态迁移的时间。

## 定时运行

监控已配置为每小时自动执行一次，由 `scripts/monitor_pipeline.py` 提供数据。
定时任务本身只做观测，不修改训练过程；是否在里程碑处补写报告由运行时判断决定。

手动查看最近事件：

```bash
cd /Users/wlh/Documents/【实践】minimind
tail -n 10 artifacts/monitor-events.jsonl
```

## 设计约束

- **只读**：脚本不启动、不停止、不重启任何训练进程，也不写入 checkpoint 或训练日志。
- **不重复报警**：依赖 `monitor-state.json` 中的状态快照与冷却时间戳。
- **不制造结论**：loss 数值异常只报告「非有限值」这类确定性问题；趋势判断留给人或报告环节，
  避免用启发式阈值产生误导性结论。
