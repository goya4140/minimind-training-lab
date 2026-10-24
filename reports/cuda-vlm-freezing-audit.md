# VLM 冻结策略：实际权重对照

这不是能力评估或阶段完成声明。VLM SFT 在后台继续，本页只核验一个已保留快照。
run 为 `mm64-cuda-vlmfix-v3-20261024T170500Z`，训练 commit `9eafe5a548df7d5eae7ce2efe420e512a52d647e`。

## 设计与实测是否一致

固定实现 `MiniMindVLM.set_alignment_trainable()` 先冻结全部参数，再开放 Projector；
`set_instruction_trainable()` 在此前提下开放 Decoder 第 0 与第 7 层。
CPU 读取父 LLM SFT 最终归档、当前 Alignment 最终归档和 SFT step 10,000 的固定恢复快照，
用 `torch.equal` 比较同名、同 shape/dtype 的完整张量，不抽样、不只看范数。

| 比较 | 权重项数（state keys） | 结果 |
|---|---:|---|
| Alignment 语言主干 vs 父 LLM SFT | 91 | 全部相等，语言权重未改变 |
| SFT Projector vs Alignment | 6 | 全部已变化 |
| SFT 首尾 Decoder 层 vs Alignment | 22 | 全部已变化 |
| SFT 其他语言权重 vs Alignment | 69 | 全部相等 |

69 个未变化项包括中间 6 层、Embedding、共享 LM Head 和最终 Norm。
state keys 不是独立参数数量，共享权重会出现多个名字；不要把 97 个 state keys 写成 97 个参数。
快照还通过恢复完整性检查：配置与 SFT YAML 一致，模型与 optimizer 有限，RNG 和更新边界完整，
history 仅属于本 run。具体逐项名称、SHA 和恢复审计见[原始审计摘要](vlm-freezing-step10000-20261025.json)。

## 为什么要做这个检查

“配置说冻结”与“实际参数没有变化”是两个证据层次。对照已完成阶段与中间快照，
可核实本次实际运行没有意外更新被声明冻结的语言层，同时确实更新了允许训练的组件。
冻结语言参数也不意味着切断梯度：回答 loss 仍可通过冻结层的计算传播到可训练 Projector。

张量最终相等不等价于逐步梯度追踪，不证明每个训练时刻都没有暂时变化，也不保证未来快照。
检查未加载或比较外部 SigLIP2 权重，不把这里的语言层结果当作视觉编码器冻结证明；
该部分另由固定实现、资产指纹与先前真实 CUDA 探针的梯度检查提供不同范围的证据。
没有把此审计变成新模型训练或基于测试结果调参。

## 来源、保留与复现

服务器根目录：`/root/autodl-tmp/wlh_goya/MiniMind/experiments/vlm-slot-v3`。
固定快照目录：
`artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/audits/sft-freezing-20261024T174800Z/`。
训练器采用原子替换保存恢复文件，审计先硬链接固定一个 inode，再只读比较；旧快照不会随新保存变化。
快照及完整 audit 保留服务器，公开摘要不包含张量数值、数据、optimizer 内容或凭据。
快照 SHA-256：`49c0fc734ff5a9364e2bdf240108a663f19f0521a600ff3f0580d7d8a1f30716`。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src \
  .venv/bin/python reporting/audit_vlm_freezing.py \
  --run-dir artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z \
  --sft-snapshot artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/audits/sft-freezing-20261024T174800Z/vlm-sft-cuda.resume.pt
```

工具验证父来源、归档事件/SHA、快照所属 run 和固定配置，最后复查训练源码指纹；
输出审计 JSON 到标准输出，不写权重、不访问 GPU、不重启训练。38 项 reporting 测试通过，
其中 4 项新测试验证权重项的组件分类，真实张量对照另由上面的实际执行证明。
未来完成 SFT 后仍需最终审计，不能将本次中间结果替代最终状态。
