# RoundoffLab：量化误差，不只看权重

[交互演示](https://gedouluhui.github.io/roundoff-lab/) · [English](README.md) · [完整数值契约](docs/DESIGN.md)

输入权重矩阵 `W[输出通道, 输入通道]` 和代表性输入 `X[样本, 输入通道]`，分别比较量化后的权重误差与 `X @ W.T` 输出误差。生成一个可离线打开、切换策略、检查通道及导出 JSON 的 HTML 报告。

只需要 CPU 和 NumPy，不需要 GPU、模型下载、API 密钥、服务器或模型格式转换。它是单层诊断工具，不是部署量化器，不宣称加速、节约真实显存或改善模型准确率。

## 本分支新增：0.2.0a1 候选验证工作流

0.2.0a1新增冻结候选验证；顶部在线链接仍是v0.1演示，请下载0.2.0a1预发行版或从当前源码安装。旧`analyze`／`demo`和schema-1数据继续兼容。

新增`study`直接接受外部工具**已经反量化**的浮点权重文件，不二次量化，不从名称猜测GPTQ/AWQ身份、位数、尺度数或裁剪数量。W与候选必须采用同一个`[输出通道,输入通道]`坐标顺序；同形状不证明语义对齐。

```sh
python -m pip install .
python examples/build_study.py --out candidate-study
```

打开`candidate-study/index.html`。例子先在discovery输入选定候选，再检查validation输入，切换报告视图不改既定选择。数据与两份导出候选均由RoundoffLab合成产生，不冒充第三方模型结果。

自己的候选清单`candidates.json`形如`{"a":"exports/a.npy","b":"exports/b.npy"}`；相对路径从清单目录解析：

```sh
python -m roundofflab study --weights weights.npy --discovery discovery.npy --validation validation.npy --candidates candidates.json --out result
```

默认依据discovery的`propagated_mse`选择；`--selected external:a`可保持此前已选定的候选。validation不参与选择；不同文件／指纹不证明独立采样，反复看validation再改参数属于数据复用。数值完全相同的两组输入会另给复用警告，没警告也不是无泄漏证明。

新报告分别列出“两次float64前向相减的观测误差”和“直接计算`X @ (Wq-W).T`的传播误差”。例如`W=[[1,.1]]`、`Wq=[[1,0]]`、`X=[[1e16,1]]`，前者可能为零，后者MSE仍为0.01。工具不再将这种舍入掩蔽视作没有影响，但两者都不代表任意精度真值或部署内核表现。

新工作流按固定逐行矩阵向量运算和固定顺序累积，`--batch-rows`只改变临时输出批量，不改变归约路径；这是以吞吐换取批次选型稳定性的参考实现，不宣称更快。最坏样本给出零基行／通道坐标，不嵌入原始激活。

若已经安装PyTorch，运行`python examples/pytorch_export.py --out torch-example`可验证真实`dequantize()`导出，并对照故意漏掉zero point的错误。样本仍为合成数据，不是预训练模型跑分。PyTorch不是运行依赖，不会自动安装。完整契约见[候选验证说明](docs/VALIDATION_STUDY.md)。

## 安装与演示

建议先激活自己的 Python 3.10+ 虚拟环境：

```sh
git clone https://github.com/gedouluhui/roundoff-lab.git
cd roundoff-lab
python -m pip install .
python -m roundofflab demo --out demo-quiet
python -m roundofflab demo --scenario active-outlier --out demo-active
```

分别打开输出目录里的 `report.html`。两个演示使用同一份合成权重，只改变异常大权重所在输入方向是否被激活；这是刻意构造的数值反例，不是实际模型跑分。报告同时给出权重 MSE、输出 MSE、相对输出 L2、逐通道误差及聚合热力图，不把两种不同单位的 MSE 直接比较大小。

## 分析自己的矩阵

将单层的权重与输入分别保存为有限、非空的二维 float32/float64 `.npy` 文件：

```sh
python -m roundofflab analyze --weights weights.npy --inputs inputs.npy --out result
```

输入通道数必须相同；工具比较无偏置线性贡献，不自动加偏置、不量化激活。`.npy` 只读映射并禁用 pickle，不导入或执行模型代码。

默认比较整张量、逐输出通道、行内分组等策略；可用 `--configs policies.json` 提供显式策略列表，格式见英文 README。支持 2–8 位、对称窄范围编码，4 位为 `[-7, 7]`，舍入为 ties-to-even。百分位阈值从权重计算，不是从输入做校准或自动寻优。

文件默认不覆盖；换输出目录，或明确使用 `--force`。不要并发写同一目录。写入前暂存全部新文件并备份旧文件，常规失败会回滚；回滚失败则保留备份并提示位置。发布依赖文件系统硬链接支持，不保证进程中断或断电时的多文件事务；中断后换新目录重做。默认不包含原始矩阵的报告仍可能泄露统计信息，分享前需自行检查。

## 如何理解结果

权重误差较低，不等于给定输入下输出误差较低；单层输出误差较低，也不等于全模型准确率更高。更换输入分布后，结论可能改变。报告的排名反转来自实际计算，平分不算反转，没有“普遍最优策略”。输出参考范数为零时，相对 L2 显示 undefined，而不是冒充零误差百分比。

图像最多聚合为 24×64 个 tile；它展示平均绝对权重误差，不是每一个权重，更不是因果归因。工具不估算部署成本，不提供压缩权重或 GPU kernel。它不会替代 ONNX Runtime、LiteRT 的完整模型量化调试，也不声称发明了输出感知量化。

默认限制权重与输入合计 400 万元素、输出 400 万元素、24 个策略与 4096 个输出通道。元素限制可以通过参数调整，但不是峰值内存或运行时保证。大层请用小型代表性切片，再用独立留出输入验证结论。

## 验证与贡献

```sh
python -m unittest discover -s tests -v
```

测试包含手算舍入、独立标量参考、尾组、零阈值、各向同性输入对照、真实 CLI 往返、覆盖保护与脚本注入字符串。跨平台状态请看真实 CI 回执；测试通过不等于任意模型正确性或真实用户效果已经验证。

欢迎提交能公开的小矩阵反例、清晰的错误提示改进及带独立参考的新策略。请勿在 Issue 上传私有权重、敏感输入或密钥。项目使用 MIT 许可证，开发过程有 AI 辅助。
