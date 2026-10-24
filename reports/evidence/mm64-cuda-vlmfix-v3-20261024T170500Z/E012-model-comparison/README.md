# E012：可追溯的跨检查点评估比较

![同固定语料PPL](language-ppl.png)

![视频正常与倒序F1](video-frame-order-f1.png)

这两张是归档指标生成的数据图，不是终端截图。已逐张检查标题、数值、坐标、图例与限制脚注完整可读。
数据来自[comparison-source.json](comparison-source.json)，PNG哈希与渲染器信息见
[figure-manifest.json](figure-manifest.json)。没有误差条：只代表一次训练轨迹，未估计多seed或抽样不确定性。

`reporting/model_results_comparison.py extract`先验证六阶段归档与父运行血缘，再读取6个训练摘要和
4个固定评估JSON，检查精确归档路径与SHA。四个语言检查点的token/byte分母必须一致，
PPL=exp(NLL)、BPB的分母公式也检查一致；这些检查不是语义去重或输入独立性的证明。
公开source只保留汇总数字和原文件SHA，不含完整原日志或逐样本输出。

PPL图从0起轴，同一2048条留出语料上较低更好，不是总体能力排名。
视频图比较同一集合的正常/倒序词面F1；两种集合的均值不可互换，不是语义准确率。
倒序仍使用原视频参考。相等的柱子忠实显示零差，不加视觉偏移或虚构误差条。

服务器本run的`curves/final-model-comparison-source/`保存同一JSON，
`curves/final-model-comparison-figures/`保存原PNG及清单。复现另用新输出目录，拒绝覆盖：

```bash
.venv/bin/python reporting/model_results_comparison.py extract \
  --run-dir artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z \
  --output artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/curves/comparison-new-source
.venv/bin/python reporting/model_results_comparison.py plot \
  --source artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/curves/comparison-new-source/comparison-source.json \
  --output artifacts/runs/mm64-cuda-vlmfix-v3-20261024T170500Z/curves/comparison-new-figures
```

工具4项synthetic测试检查分母、有限值、PPL/BPB算术、完整模型顺序；不代表模型性能通过。
实际发布另检查10个来源SHA、2个图像SHA和数值对应关系。
[综合学习报告](../../../cuda-three-model-synthesis.md)提供设计、实现与失败解读。
