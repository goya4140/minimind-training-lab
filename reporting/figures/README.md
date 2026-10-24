# 设计图的布局与修复记录

本目录保留架构 PNG 的设计来源，不是待执行的 DOCX/PPTX 交付任务。用户已取消这两类文件制作和交付。

- `design-spec.json`：720×540 pt 画布、字体层级、语义颜色、精确对象文本和预定拓扑。
- `powerpoint-operations.json`：当次原生对象操作记录，包含经预览验证的连接端口映射；仅用于追溯图示来源。
- `connector-correction.json`：Reviewer 发现与最小对象级修复计划。

绘图时检测到 Microsoft PowerPoint 16.99.2，但没有连接的 Office.js 窗格，因此使用独立文件式
OOXML 工作副本。没有改动其他演示文稿或服务器训练代码。每个区域先结构审计再渲染复核，最终
全部四图再次审计并预览。实际 PNG 和中文说明见
[架构图索引](../../reports/figures/minimind-architecture/README.md)。

当次适配器为 Scientific Illustrator 1.5.4，水平连接 site 的几何约定与写入的 OOXML rectangle
端口索引不一致：调用端 2/4 分别按右/左计算，但对应原生索引 1/3 在渲染时表现为左/右。
因此只对水平端口调用参数交换 2↔4，顶部/底部保持 1/3，保留所有原生附着关系。
不要把此映射无条件套用于未来工具版本，必须检查实际预览。
