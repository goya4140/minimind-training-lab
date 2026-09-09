# 冻结外部组件

VLM/Video-Omni 训练中的视觉编码器不是本项目从零训练的参数；它提供连续视觉特征并在训练中
保持冻结。`scripts/fetch_models.py` 固定 revision，
对大权重验证字节数与 SHA-256。

| 组件 | Hub repo | Revision | 主要权重 | License |
|---|---|---|---|---|
| SigLIP2 P32 256 | `jingyaogong/siglip2-base-p32-256-ve` | `9465d1dc…` | 189,129,296 bytes；`c1e9cc19…` | Apache-2.0 |
这里的“从零训练”特指 64M LLM、VLM projector、Video-Omni temporal adapter 与 video
projector 从本项目定义的初始化与阶段 checkpoint 演进；不声称从零训练 SigLIP2。

下载器对三个文件都做字节数与 SHA-256 双重校验：

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 410 | `5ad8dda7d55541c7749f9b1cc43fe8eb8c70d8664588d89f710242ce06b3167e` |
| `preprocessor_config.json` | 394 | `d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff` |
| `model.safetensors` | 189,129,296 | `c1e9cc19ed6704b87353ee00b9ff5d6191886d741898339984364f789c62810d` |

配置与预处理文件虽小，却决定 hidden size、patch size、图像缩放和归一化；仅验证大权重不足以复现
相同视觉输入。

## 本地加载验证

- SigLIP2 可加载并接受 `[8, 3, 256, 256]` 的真实 QIVD 视频帧；
- patch features 经时序模块得到固定数量 video tokens，并能完成有限 loss 的端到端前向；
- 正常帧和倒序帧的 video embedding 不完全相同，证明架构在初始化阶段已经保留顺序信息。

这只证明组件边界和张量链路正确，不代表训练后的模型已经具备视频理解能力。
