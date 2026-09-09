# 冻结外部组件

VLM/Omni 训练中的视觉、语音和 codec 模块不是本项目从零训练的参数；它们提供连续特征、
离散音频 codes 或音色条件，并在训练中保持冻结。`scripts/fetch_models.py` 固定 revision，
对大权重验证字节数与 SHA-256。

| 组件 | Hub repo | Revision | 主要权重 | License |
|---|---|---|---|---|
| SigLIP2 P32 256 | `jingyaogong/siglip2-base-p32-256-ve` | `9465d1dc…` | 189,129,296 bytes；`c1e9cc19…` | Apache-2.0 |
| SenseVoiceSmall | `jingyaogong/SenseVoiceSmall` | `964f07d8…` | 468,291,478 bytes；`21881197…` | Apache-2.0 |
| Mimi | `jingyaogong/mimi` | `b4e362bb…` | 192,346,842 bytes；`7542ee03…` | CC-BY-4.0 |
| CAM++ | `jingyaogong/campplus` | `77bb7d92…` | 14,173,135 bytes；`55ffb1a5…` | Apache-2.0 |

这里的“从零训练”特指 64M LLM、Vision/Audio Projector 和 Thinker–Talker 可训练主体从本项目
定义的初始化与阶段 checkpoint 演进；不声称从零训练上述通用 encoder/codec。

## 本地加载验证

- SenseVoiceSmall encoder：221,136,992 参数；1 秒静音经 frontend 得到 `[1, 17, 560]`，
  encoder 输出 `[1, 17, 512]`，全部为有限值；
- Mimi：79,308,609 参数；配置支持最多 32 个量化器。使用真实训练数据的 8 个 active
  codebook、100 帧成功解码为 192,000 个 24 kHz 样本（8 秒），全部为有限值；
- FunASR 的音频加载链依赖 `torchaudio`，已纳入 `omni` 可选依赖并写入 lockfile。

这证明本地冻结组件可加载并与数据格式衔接，不代表可训练主体已经具备语音理解或合成能力。
