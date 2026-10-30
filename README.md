# Industrial Anomaly Research Notes

工业视觉异常检测的探索性代码与负结果记录。**不是已发表论文，不是已验证的新方法，不承诺 EI 录用。** 代码与记录使用 AI 辅助完成。

## 当前结论

研究固定高分辨率精检预算下的区域选择。目前纹理校正残差融合、简单合成纠错收益预测两条候选机制均未获得充分支持，已停止扩展。

- grid 开发实验，16 块选 4 块：粗异常分数宏平均缺陷像素覆盖率约 81.49%，纹理校正融合约 76.34%。覆盖率不是最终检测精度。
- 合成验证的未见模糊类型：净收益预测错误代价 0.48111，粗分数选择 0.46715（越低越好）。只有 8 张背景图，差异区间跨零，不支持显著优劣结论。见 [实验记录](UTILITY_PILOT_FINDINGS.md) 与 [协议](UTILITY_PILOT_PROTOCOL.md)。
- grid 已反复用于开发，不是独立最终测试集；没有多类别确认，不宣称部署收益或新颖性。

## 文件与复现边界

`scripts/` 为实验脚本；`requirements-core.txt` 为核心依赖，`requirements-lock.txt` 是历史环境快照，未验证全新环境安装。

不分发数据集、合成图片、模型权重、原始结果缓存和第三方仓库。文档的 `results/` 路径指原实验本地记录，公开版不包含这些工件，不是完整归档复现包。部分脚本依赖前序实验日期命名的输出，应先阅读输入输出与前置条件，不要逐个盲跑。

### 基础检查

在独立 Python 3.10 环境执行，不修改系统 Python。已测环境：torch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 1.26.4、RTX 4060 8GB。先确认本机驱动兼容和可用空间。

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt
git clone https://github.com/amazon-science/patchcore-inspection.git vendor/patchcore-inspection
git -C vendor/patchcore-inspection checkout --detach fcaa92f124fb1ad74a7acf56726decd4b27cbcad
python -I scripts/test_selection.py
python -I scripts/test_utility_synthesis.py
```

上述测试不读取真实缺陷图片，也不验证模型精度。完整数据实验未做干净环境端到端复跑。实际采用冻结 ResNet50 IMAGENET1K_V1、layer2/3 与 CPU FAISS，不是作者默认 WideResNet 配置的精确复现。

流程：数据准备 → 粗尺度基线 → 区域筛选 → 全分块/稀疏精检 → 合成收益试验。数据脚本可能联网下载，模型脚本会消耗算力并可能下载权重。

## 来源与许可

见 [THIRD_PARTY.md](THIRD_PARTY.md)。原创部分暂未选择开源许可证；公开可阅读不等于授予任意复制、修改或再分发权限。

- [PatchCore 作者代码](https://github.com/amazon-science/patchcore-inspection)
- [HiAD](https://arxiv.org/abs/2508.12931)
- [Tiled Ensemble](https://arxiv.org/abs/2403.04932)

欢迎针对实验协议、实现错误和研究价值反馈。负结果只限定于当前配置，不证明整个问题方向无效。
