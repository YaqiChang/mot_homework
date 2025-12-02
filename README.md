# 项目结构
```
project_root/
│
├── config.py          # 全局参数配置（时间步长、雷达位置、噪声、阈值等）
├── data_gen.py        # 目标轨迹生成 + 量测生成（点目标 / 扩展目标 / 杂波 / 遮蔽）
├── models.py          # 扩展目标状态、Bernoulli 组件、LMB 状态的数据结构
├── tracker_lmb.py     # LMB 跟踪器（预测、更新、track 提取）
├── metrics.py         # RMSE、OSPA 等评估指标
├── utils.py           # 工具函数（坐标转换、聚类、gating 等）
├── plot.py            # 绘图
└── main.py            # 主函数：仿真 → 跟踪 → 评估 → 输出结果
```

# TODOLIST 
- [x] 需要补充结果分析，包括：1）绘图函数，可以不是动态的
- [ ] 2）常见的指标需要分析出来。为什么有这样的结果xxxxx  
- [x] 需要按着题目分析里面那个逻辑（除了上面的术语说明），建议是：先把题目列出来，然后从题目转换成简短的几条有效信息（做数学题时候的列已知条件），然后数学建模，分析xxxxxxx
- [x] 可以考虑加其他算法，但是参数设定有很麻烦，要不就算了。首先接感觉能不能把现有的参数优化以下得到一个更好的结果。---已经更新成GLMB
- [ ] 可以AI解读代码关键部分，制作PPT
修改代码相关的内容加在对应的py文件里面。没用什么包，就matplotlib和numpy

 # 比起初始调整内容说明
调整内容：

  - config.py：将杂波强度降到 0.05/km²（原值在 60km×40km 区域下会生成 ~1.2 万点/
    帧导致图面被杂波淹没），保持“强杂波”但可计算；gating 半径放宽到 500m，分配的
    优势裕度降到 50m，避免交汇时真量测被判模糊。
  - tracker_lmb.py：去掉更新阶段重复的贪心分配循环，保留一次全局最近邻分配，确保
    量测不被重复或丢弃。

  验证结果（运行 python main.py）：

  - Label 对齐：{0: 'A', 1: 'B'}
  - RMSE A: 65.93 m，RMSE B: 42.61 m
  - Mean OSPA: 34.15 m
  - 生成的 traj_result.png、meas_step60.png 已更新，单帧点云不再被杂波刷屏。


# 混淆矩阵分析
传统混淆矩阵用于分类任务，而在多目标跟踪任务中，“混淆” 是指：

真实状态	发生情况	归到什么 cell
真实目标 A	被正确跟踪为 A	TP_AA
真实目标 A	被误跟踪为 B	Confusion AB
真实目标 A	被漏掉（无 track）	FN_A
无真实目标（假目标）	被跟踪器输出为某 label	FP(track)

所以混淆矩阵是一个 (N 真目标 + 1 clutter 行) × (N 估计 + 1 None 列) 的矩阵：

例：A、B 两个真目标：

	A(est)	B(est)	None(漏检)
A(true)	TP_A	Conf_A→B	FN_A
B(true)	Conf_B→A	TP_B	FN_B
Clutter	FP_→A	FP_→B	—

## 实现的混淆矩阵分析包括
在 main.py 运行时直接给出对应数字，同时写进 metrics_summary.txt：

  1. A 被多少帧误认为 B？
      - 使用混淆矩阵 C[0, 1]：
      - 控制台输出：A 被误认为 B 的帧数: ...
  2. B 被误认为 A 的次数？
      - 使用 C[1, 0]：
      - 输出：B 被误认为 A 的帧数: ...
  3. 假目标出现多少？（假 track 总数）
      - 使用 build_id_assignment_series 做逐帧最近邻匹配，对每帧每个 track 判定
        是否匹配到 A/B；
      - 所有被标为 "None" 的 track 计为“假目标一次”（按 time×track 计数）；
      - 输出：假目标（未匹配任何真值）的总次数: ...
  4. A/B 各漏检了多少帧？
      - 来自混淆矩阵的 None 列：
          - A 漏检：C[0, 2]
          - B 漏检：C[1, 2]
      - 输出：
          - A 漏检帧数: ...
          - B 漏检帧数: ...
  5. GLMB 在干扰区是否频繁产生假警？
      - 对所有假目标（上面 "None" 的那些 track），检查它们的位置是否落在干扰区
        （用 in_jam_region(pos, t) 判断）；
      - 统计总次数 false_in_jam：
      - 输出：其中位于干扰区内的假警次数: ...
      - 你可以直接用 false_in_jam / false_total 看假警中有多少发生在干扰区，也可
        以对比时间长度自己判断是否“频繁”。

  另外：

  - 这些统计会被追加写入 metrics_summary.txt，方便你事后看数值：
      - A_as_B, B_as_A
      - A_missed, B_missed
      - False targets total
      - False targets inside jamming area
  - confusion_matrix.png 仍然是 2×3 热力图（True A/B × Pred A/B/None），用来配合
    这些数字做直观分析。
