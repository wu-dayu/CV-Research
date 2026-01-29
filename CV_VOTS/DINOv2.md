| **颜色** | **代表含义**       | **适用场景**                                 |
| ------ | -------------- | ---------------------------------------- |
| **红色** | **核心贡献/核心结论**  | 论文解决的痛点、Abstract和Conclusion的关键句。         |
| **黄色** | **重要定义/概念**    | 第一次出现的专业术语（如：Inductive Bias, Zero-shot）。 |
| **蓝色** | **数学公式/理论依据**  | 损失函数 Loss Function、注意力机制公式。              |
| **绿色** | **实验结果/性能数据**  | SOTA 表现、消融实验的关键数据。                       |
| **紫色** | **值得借鉴的代码/方法** | 实现细节，如“使用了 AdamW 优化器”、“学习率衰减策略”。         |
| **橙色** | **不足/未来工作**    | 作者承认的限制（Limitations），这往往是你的选题切入点。        |
| **灰色** | **背景/引用文献**    | 经典的参考文献，标记以后要去读。                         |
| **青色** | **个人疑问/随笔**    | 自己读不懂的地方，待查阅资料或问导师。                      |
# 📄 [Paper Study] DINOv2 2026-01-25

## 1. 快速预览 (Quick Read - 10min)
- **年份/会刊：** {{2023}} / #Arxiv
- **领域标签：** #CV/{{底层}} 
- **核心痛点 (Motivation)：**
    - 当时的视觉模型依靠“图像-文本对”进行监督，虽然能获得全局语义，但会丢失像素级特征。
    - 前人采用的自监督方法在大数据集训练上极其不稳定。
    - 前人普遍认为自监督学习只需要堆数据量（uncurated data），不重视质量，导致训练效果差。
    - Meta 的团队写这篇论文是为了证明：**自监督学习通过改进“数据精选流水线”和“训练稳定性工程”，可以产生比强监督模型更通用、更强大、且在图像级和像素级任务上全面领先的视觉特征提取器。** 他们想打破“只有靠文本标签才能做大模型”的迷信，确立自监督学习作为**视觉基础模型（Foundation Models）的统治地位。
- **核心贡献 (Key Idea)：**
	- Distillation with No labels 
    - [Summary by Gemini](https://docs.google.com/document/d/1Eez54NqxIUQNmwWN47YwglgrthFHSBG2JgD-OktlYgE/edit?tab=t.0)
    - https://docs.google.com/document/d/1vIcf5c0YGtbBdzWcCtL_Huqd-nIFba1mnBLGx8lB3TA/edit?tab=t.0
- **代码仓库：** [GitHub Link](https://github.com/facebookresearch/dinov2)

---

## 2. 核心架构与数据流 (Architecture & Data Flow)
- **Data Processing 数据处理**：curated dataset
	![[Pasted image 20260125225958.png]]
	Uncurated Data来自网络，
	- **Deduplication:** “We apply the copy detection pipeline of Pizzi et al. (2022) to the uncurated data and remove near-duplicate images.” (Oquab 等, 2024, p. 5)
	- **Retrieval:** “We build our curated pretraining dataset by retrieving images from our uncurated data source that are close to images in our curated sources. In order to do this, we first compute an image embedding using a self-supervised ViT-H/16 network pretrained on ImageNet-22k, and use cosine-similarity as a distance measure between images.” (Oquab 等, 2024, p. 5)
- **Discriminative Self-supervised Pre-training**
	- 使用传统的ViT结构，Model Distillation **Teacher Model的参数**是Student历史参数的动量移动平均(Teacher 参数=Student EMA)$$\theta_t^{(k)}=m\theta_t^{(k-1)}+(1-m)\theta_s^{(k-1)}$$
		- **训练阶段**：双模型（Teacher + Student）并行，Teacher 负责产生稳定目标，Student 负责梯度更新。**student model观察被masked的图像或crops，teacher model观察完整的图像并产生目标分布(Target Distribution)，引导student区模仿。**
		- **部署/推理阶段**：只保留**一个模型**（即 Student/EMA 权重）。你不需要再跑两个网络，这保证了推理时的速度和 ViT 原始架构完全一致。
	- **核心损失函数**： DINO(侧重全局特征)与iBOT(侧重局部特征)
		![[Pasted image 20260125231206.png]]
		- https://chatgpt.com/s/t_69763553fe5c8191a0257983b19631e9 ChatGPT对两种Loss公式的解析，目前看不懂
	- **大规模训练的技术改进**
		- DINO和iBOT使用不同的MLP投影头
		- KoLeo正则项，“encourages a uniform span of the features within a batch” 
			- **物理意义**：这个正则项会计算 batch 内特征点之间的最小距离，并鼓励它们**“散开”** 。它能促使特征在空间中呈均匀分布，防止所有特征聚在一起，从而显著提升了**图像检索**任务的性能 。
		- 大规模使用[[Stochastic Depth]] [[随机深度]]，通过随机跳过一些block来防止过拟合和训练崩溃 “This saves memory and compute in proportion approximately equal to the drop rate” 
---

## 3. 深度技术检查清单 (Direction-Specific Checklist)

### 🟢 分类/骨干网络 (Classification/Backbone)
- [ ] **Basic Block:** 最小重复单元长什么样？(残差结构/Transformer Block/Ghost Block?)
- [ ] **特征提取:** 它是如何权衡局部特征（Conv）和全局特征（Attention）的？
- [ ] **降采样:** 图像分辨率是如何一步步缩小的？(Stride Conv / Pooling / Patch Merging?)
- [ ] **性能指标:** 参数量 (Params) 和计算量 (FLOPs) 处于什么量级？
---

## 4. 数学表达与代码复现 (Math & Code)
- **核心公式：**
	  $$L_{total} = \lambda_1 L_{DINO} + \lambda_2 L_{iBOT}+\lambda_3 L_{Kleo}$$
- **代码核心逻辑 (GitHub Snippets)：**
	/dinov2/layers
	/dinov2/loss
	/dinov2/models

---

## 5. 本科生专项：基础补课与疑问 (To-Learn)

- [ ] **基础概念补课 (用 Obsidian 双链链接)：**
	- [自监督学习的特征投影机制](https://docs.google.com/document/d/190AH7nvk6jmrxfZ9nGKrI2dvHh-W5rjbeOtlY0dkMu0/edit?tab=t.0) By Gemini
		- 预训练阶段（无 Metadata）：模型在做“找共同点”的游戏。它投影的对象是匿名特征空间（Prototypes）。它不识别语义，只识别结构和纹理的相似性。
		- 下游阶段（微调/线性探测）：人类介入，把模型自发学到的“模式”与“单词（Labels）”建立映射关系。
	- 训练最开始如何保证其收敛 [自监督学习的收敛机制](https://docs.google.com/document/d/1iG26S3eN9P2b2HY24EDAm3puQpwH_exGuHSXgH5fMtg/edit?tab=t.0)Inductive Bias; EMA; Anti-Collapse
	- **Sinkhorn-Knopp centering**发生在Teacher 的输出层 (Heads)，保证训练稳定、不坍缩保证了分类的多样性
	- **KoLeo Regularizer**作用于最终的特征向量 (Embeddings)，提升特征的检索性能、精细度，保证了特征的区分度
	- [[Fine-tuning]]
		![[Pasted image 20260125163126.png]]![[Pasted image 20260125163155.png]]
	- [[Supervised Learning]]
		https://docs.google.com/document/d/1mVU8pfZTWFncAt0f4uLVKKtnLVU0N9p-vhP-iw_9YwI/edit?tab=t.0
	- [[Distillation]]
		“Knowledge distillation (Hinton et al., 2014) aims at reproducing the output of a large model with a smaller model by minimizing some distance between both outputs for a set of given inputs.” (Oquab 等, 2024, p. 7)
	- [[EMA]]
		![[Pasted image 20260125230619.png]]
- [ ] **遇到的坑：**

---

## 6. 关联阅读与总结 (Summary & Links)

- **上一代工作：** [[ViT---Vision Transformer]] 
- **核心对比：** 
	- Supervised vs Self-supervised: ViT的训练依赖数据标注，有监督学习，而DINOv2不然。导致Loss函数不同。DINOv2不需要任何标签，学到的特征更全面，更具通用性。
	- 训练架构不同，前者没有self distillation