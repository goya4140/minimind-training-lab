# 视频评估的全量证据重放

## 为什么需要单独重放

固定训练代码 `53f9b9f` 的 `scripts/evaluate_video_omni.py` 会对 QIVD 前 100 条和受控集 160 条
生成正常/倒序回答，全部参与汇总评分；`generation_summary` 只将 `cases[:12]` 写入 `qualitative`。
因此原汇总不能提供 260 条完整生成记录。不能把后来重新生成的回答说成原执行的逐样本输出。

`video_evidence_replay.py` 是独立的补充证据运行，不改已固定的训练源码、配置、数据或基线结果。
它加载原 evaluator，在 `generation_summary` 调用处复制全部输入 case 到 JSONL，再原样返回
原函数的汇总。题目、样本顺序、生成参数和评分算法不变；运行时间与原执行不同，后续报告要区分。

## 门禁与执行

必须等原六阶段训练与评估流水线完成后再执行。工具检查完成事件、固定源与依赖、视频 SFT 归档
checkpoint 路径和 SHA-256，拒绝未完成阶段、歧义归档、全局可变权重和重复 eval ID。要求显式
选择本项目授权 GPU，不在训练期间争用 GPU，也不启用监控。

在服务器项目根目录运行（以下是待执行命令，不代表已有结果）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python reporting/video_evidence_replay.py \
  --run-dir artifacts/runs/mm64-cuda-20261024T075332Z \
  --eval-id fixed-video-full-output-v1-final
```

所有结果保存在 run 下 `supplemental-evaluations/<eval-id>/`：独立 manifest、原 evaluator 与
留存工具副本、260 条 JSONL（每条含正常与倒序答案）、原格式 summary 及校验和。每条记录保留
参考、类别、问题、原始回答与原评分字段，不额外宣称包含原脚本未生成的 token IDs。两组样本数
与 ID 唯一性检查失败时不宣告成功；异常保留失败 manifest 和已有记录，不覆盖旧证据。

工具测试只验证留存不改变 summary、完整计数、拒绝重复 ID 和禁止覆盖；不证明模型评估已经完成，
也不证明原指标的科学有效性。实际 GPU 重放仍待全部训练和固定评估完成后执行。

## 指标解释边界

- 倒序 loss 和生成评分均沿用原始视频参考，不是倒序视频新正确答案下的准确率。
- 受控集的正确答案会反转；重放不悄悄改参考或评分，也不引入新调参目标。
- QIVD 中有些题目不依赖帧顺序，倒序后分数不变并不自动说明模型失败。
- 回答变化率只说明生成变化；正常高分且倒序答案语义正确，才是更有力的顺序相关行为证据。
- 视频 loss 是脚本的 batch loss 算术平均，不应改称全体 assistant token 的加权 NLL。
- 视频脚本的纯文本回归生成上限为 48 token，独立 LLM 脚本上限为 80 token；即使提示相同，
  也不应将生成长度、复读比例和速度当作严格同条件比较。文本留出 loss 可另按其实现口径比较。

这里只修正文档解读与证据留存，不改变原固定实验协议。若以后要改评分、生成上限或新增倒序正确
参考评估，应注册独立版本，保留原结果，不因测试分数而回调本次训练超参数。
