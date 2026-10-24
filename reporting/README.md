# 学习报告生成准备

本目录保存中文学习报告的方法部分内容和可重复生成脚本，**不代表最终报告已完成**。
最终交付仍需六阶段训练、固定评估、最终曲线、实际输出、消融分析和完整复现附录。
`methods-content.json` 的 `missing_final_sections` 显式保留这些要求；当前 builder 只接受
`methods-draft` 状态，不能将其切换成最终报告发布器。

## 同源内容与证据

- DOCX 使用详细原理段落，PPTX 使用同一主题的简短说明和原生可编辑表格。
- 每个主题保留具体源文件，代码与配置绑定训练 commit，文档绑定单独的文档 commit。
- 图像 E001 是真实终端窗口截图，插入前核验 SHA-256，不生成模拟终端。
- 文字、表格和讲者备注可编辑；真实截图保持栅格原件，不声称截图中的文字可独立编辑。
- 私有输出和 QA 中间文件都在 `.report-build/`，不提交 Git；最终学习报告另行完成并交付。

## 已验证的构建环境

2026-10-24 使用 Codex bundled runtime `26.1020.11348`：Python + python-docx、
Node + `@oai/artifact-tool`，以及 bundled LibreOffice renderer。字体为本机已存在的
`Arial Unicode MS`，这是设计选择，不是用户指定字体。不要往受管理依赖目录安装替代包或字体。

运行前通过 workspace dependency loader 获取路径，设置 `RUNTIME_PYTHON`、`RUNTIME_NODE`、
`RUNTIME_NODE_MODULES` 和 `ARTIFACT_SKILL_DIR`；后者是 Presentations skill 目录。
按文档/演示技能执行 artifact-operation marker，再进行 authoring。新版本使用新文件名，
不覆盖旧版排版证据。

DOCX 构建命令：

```bash
mkdir -p .report-build
"$RUNTIME_PYTHON" reporting/build_methods_docx.py \
  --output .report-build/methods-new.docx
```

macOS 的 bundled LibreOffice 首次渲染没有找到中文字体，出现缺字；仅生成文件并不能发现此问题。
解决方法是用任务内 fontconfig 显式读取现有系统字体，未修改 bundled runtime，也未安装新字体。
通过 Documents skill 的 `render_docx.py` 渲染并逐页检查：

```bash
FONTCONFIG_FILE="$PWD/reporting/fonts-macos.conf" \
  "$RUNTIME_PYTHON" "$DOCUMENTS_SKILL_DIR/render_docx.py" \
  .report-build/methods-new.docx \
  --output_dir .report-build/docx-new-render --emit_pdf
```

PPTX 构建命令（`node_modules` 链接仅需首次建立）：

```bash
ln -s "$RUNTIME_NODE_MODULES" .report-build/node_modules
cp reporting/build_methods_deck.mjs .report-build/build_methods_deck.mjs
ARTIFACT_RUNTIME_PYTHON="$RUNTIME_PYTHON" \
  "$RUNTIME_NODE" .report-build/build_methods_deck.mjs methods-new
```

应先 export 上述运行变量，或在命令中传入。最终化过程要求验证收据与导出的 PPTX 分目录，
以及 import 子进程可读取 `RUNTIME_NODE_MODULES`。这些工具调用问题均发生在本地报告构建中，
没有改动服务器训练实现、配置或环境。

## 内部验证与剩余工作

当前方法样稿为 9 页 DOCX、9 页 PPTX，所有页面已渲染并逐页视觉检查。PPTX 还通过了包结构、
几何、字体、7 张原生表格所在页和 first-party import 验证。初版 DOCX 的中文缺字、标题边线
残留和单独落页的来源段落已修复。图片缩放不裁剪内容，时间和 run ID 可追溯。

这些只验证方法样稿的实现与排版，不验证最终实验结论，也不构成目标完成。
最终完整报告新增结果后必须重新渲染全部页面，不能沿用此次 QA 声明。
