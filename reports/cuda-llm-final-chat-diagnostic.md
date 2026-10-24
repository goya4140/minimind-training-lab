# CUDA LLM 最终检查点：固定对话诊断

2026-10-25（北京时间）。本页是已完成 LLM 阶段的补充评估，不是六阶段实验完成声明。
训练 run 为 `mm64-cuda-20261024T075332Z`，训练代码固定为
`53f9b9f46560bee446d1c74e89cac991b71a8053`。后续多模态修复运行继承这些权重，不重复训练 LLM。

## 协议与边界

使用此前冻结的 [chat-diagnostic-v1](../reporting/protocols/chat-diagnostic-v1.json)，
以及其[来源更正](../reporting/protocols/chat-diagnostic-v1-provenance.json)。8 题由项目助手起草，
不是人工参与者数据、外部 benchmark 或经过语义去重的独立测试集。原协议中的 Human-authored
描述不准确，已通过独立更正文件和注册事件纠正，未偷偷改写冻结题目或答案。

推理使用项目 tokenizer 的 chat template，单条 user 消息并附 assistant generation prefix；
训练 SFT 同样使用该 tokenizer 的 chat template。CPU FP32、greedy、最多 80 个新 token、遇 EOS
停止、不用 KV cache，线程环境 OMP/MKL/OPENBLAS 均为 4；未占用训练 GPU。
这不是原有 raw-prefix 固定生成的重跑，也不替代原有通用文本留出集评估。

每题严格二值评分，只去除首尾空白；JSON 题要求恰好包含指定字段和值，拒绝代码围栏、重复键和
非 JSON 常量。额外解释即使包含正确答案也不通过。全部 8 题均纳入，不根据结果扩大答案集或改提示词。

## 结果

| 最终阶段 | 严格通过 | EOS 停止 | 达 80 token 上限且未 EOS |
|---|---:|---:|---:|
| LLM Pretrain | 0/8 | 2/8 | 6/8 |
| LLM SFT | 0/8 | 3/8 | 5/8 |

两个阶段 C01–C08 全部不通过；C06 原因为 `invalid_or_nonunique_json`，其余为
`unmatched_reference`。下列是人工定性阅读，不是新增语义评分，也没有改变上述严格分数：

| 题号／任务 | Pretrain 实际输出概况 | SFT 实际输出与局限 |
|---|---|---|
| C01 首都，仅城市名 | 重复 `Alibaba Cloud`，EOS 停止 | `中国的首都是北京。`，事实正确但不满足仅城市名 |
| C02 2+3，仅数字 | 重复连字符，到上限 | 解释初始化变量、循环等步骤，未直接给出 5，到上限 |
| C03 原样复制 | 重复连字符，到上限 | 追问行数并重复示例说明，未原样输出，到上限 |
| C04 cat 翻译 | 英文回文函数指令，EOS 停止 | 先称译为 `cat` 再称为“猫”，多余解释且前后矛盾，EOS 停止 |
| C05 排序，仅逗号分隔数字 | 回文脚本指令与重复 substantial，到上限 | 转为 Python sorted 代码教程，到上限 |
| C06 指定 JSON 对象 | header/markdown 复读，到上限 | Python json 教程与重复注释，不是要求的 JSON，到上限 |
| C07 情感，仅标签 | select/success 和大量换行，到上限 | 判断正面但附解释并有重复，EOS 停止 |
| C08 从上下文取号码 | 重复连字符，到上限 | 先把小周号码称为 73，再重复 26，未直接取值，到上限 |

SFT 在这些实例中更倾向中文对话/教程，部分内容相关，但严格短答、复制、结构化输出与停止行为
仍有明显问题。不能将 0/8 解读为“完全没有语义能力”，也不能据这些定性变化声称广泛能力提升。
Pretrain 本就未训练为对话助手，套用 chat template 的表现不等于其通用语言建模能力。

相同通用文本留出集的既有固定评估为 Pretrain PPL 6.2227168、SFT PPL 6.7796409，
详见[原运行结果与故障记录](cuda-vlm-slot-truncation-failure.md)。PPL 和本页严格短答评分测量不同能力，
不能互相替代。未依据本页结果修改训练设置，后续若改变训练或协议应另立实验。

## 证据与复现

服务器根目录：`/root/autodl-tmp/wlh_goya/MiniMind`。
以下目录均相对于 `artifacts/runs/mm64-cuda-20261024T075332Z/supplemental-evaluations/`：

- `chat-v1-llm-pretrain-final-20261025`：UTC 2026-10-24 17:12:57–17:13:17。
- `chat-v1-llm-sft-final-20261025`：UTC 2026-10-24 17:13:22–17:13:42。

每个目录包含完整 `predictions.jsonl`（原提示词、渲染后提示词、输入/生成 token IDs、实际输出、
EOS、逐题评分与耗时）、`summary.json`、`manifest.json`、冻结协议及来源更正、当次 evaluator 源码。
完整逐样本输出仅留服务器，本仓库只提交上述摘要。耗时是无预热单次 CPU 请求耗时，不作硬件基准。
最终权重经原阶段归档事件和完成事件绑定；严格加载模型结构，评估前后运行源码指纹检查通过。
两个 summary SHA 相同是因为它们保存的严格评分结果相同，不表示预测或权重相同。

| 文件 | Pretrain SHA-256 | SFT SHA-256 |
|---|---|---|
| checkpoint | `5d93e0b34336130dc7395e6fb9fc0028c969cdce0c417b92aa8e1818d8724d60` | `61aff7c70d0e6d110b07d5a92f82e2ba0e7667aba6468404a94be585861910a4` |
| manifest.json | `e095bbcc097e912a1495a5e5d0519334e5cc7c3cb2b277213d41e46616487721` | `6c753f5a493137e2dfc93fea5ebfb3b976cf54d404872af7551b77d157c78530` |
| predictions.jsonl | `ca62c66ab5c298912c08e3fc6adfd1b73b44b076377744b8d946598256551573` | `3a0e486b277d8f380f5ce1771c937a687d88ac7148acd9071cc77d0a8bf48498` |
| summary.json | `bd6a2a13c0182f0eacd8b0f09e512600786b924d8ac44003f6f5470055314fa0` | `bd6a2a13c0182f0eacd8b0f09e512600786b924d8ac44003f6f5470055314fa0` |

协议 SHA-256：`af230c71f939b0c663a6dbfa681d1046c6d34c9fe89c5e8c7f94e4c2b6ccb98c`；
来源更正：`5ca23302f5d5baa1fd3064f93aeb386128c7d70d1e9e09584ee53eff1348cd81`；
evaluator：`a24ba6d9917eb8a9f6410d9ea4a9605ef34d444659a073795a64f3271945c7f2`。

复现命令见 [工具说明](../reporting/README.md) 与 `reporting/chat_diagnostic.py --help`。
新执行必须使用新 eval ID，不覆盖上述目录；不能把 step 12,000 的早期探针当作最终结果。
已补齐[两个最终阶段的真实终端截图及来源索引](evidence/mm64-cuda-20261024T075332Z/E003-chat-final/README.md)，
展示完整逐题评分表和相同 C01 的完整实际回答。截图不替代原始预测文件和上述指纹。
