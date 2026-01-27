 
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
# 📄 [Paper Study] ViT---Vision Transformer 2026-01-19

## 1. 快速预览 (Quick Read - 10min)
- **年份/会刊：** {{2021}} / #ICLR #Arxiv 
- **领域标签：** #CV/{{底层}}
- **核心痛点 (Motivation)：**
    - Transformer在computer vision中的应用十分有限
    - “In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place.” 
- **核心贡献 (Key Idea)：**
    - “We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.” 
    - Vision Transformer (ViT) 可以取得很好的结果，同时需要的训练计算资源比CNN更少
- **代码仓库：** [[GitHub Link](https://github.com/google-research/vision_transformer)]
- **是否值得精读：** 🟢 必读

---

## 2. 核心架构与数据流 (Architecture & Data Flow)
> **提示：** 利用 Zotero 的截图功能，将模型结构图、流程图贴在此处。

- **模型结构图：**
  ![[插入架构截图]]
- **张量变化 (Tensor Shapes - 重点)：**
    ![[f796881e1fc38b3c38033a109b088d40.jpg]]
    
---

## 3. 深度技术检查清单 (Direction-Specific Checklist)

### 🟢 分类/骨干网络 (Classification/Backbone)
- [ ] **Basic Block:** 最小重复单元长什么样？(残差结构/Transformer Block/Ghost Block?)
- [ ] **特征提取:** 它是如何权衡局部特征（Conv）和全局特征（Attention）的？
	Only focus on global characteristics
- [ ] **降采样:** 图像分辨率是如何一步步缩小的？(Stride Conv / Pooling / Patch Merging?)
- [ ] **性能指标:** 参数量 (Params) 和计算量 (FLOPs) 处于什么量级？
- [ ] **Patch Size 是多少？** (论文默认是 16x16，这就是标题的由来)
	Depends on the model, for ViT-L it's 16 * 16
- [ ] **为什么需要 [CLS] Token？** (提示：它是为了最后做分类用的“总结性向量”)
- [ ] **模型在什么数据集上表现好？** (关注预训练数据集 JFT-300M，理解为什么 ViT 比较“吃”数据)
	ImageNet, CIFAR-100, VTAB, etc.
- [ ] **Inductive Bias (归纳偏置) 是什么？** (这是论文的核心论点：CNN 有平移不变性等先验知识，而 Transformer 没有，所以需要更多数据去学习)
![[Pasted image 20260120135733.png]]
- “We find that some heads attend to most of the image already in the lowest layers, showing that the ability to integrate information globally is indeed used by the model.” 

- “Further, the attention distance increases with network depth. Globally, we find that the model attends to image regions that are semantically relevant for classification” 
直觉上说CNN更符合理解图片的直观，卷积核扫过相邻的小区域，再检查总体周围的区域，与人类观察图片的方法类似。而Transformer则关注global chatacteristics.
---

## 4. 数学表达与代码复现 (Math & Code)
- **核心公式：**
- ![[插入架构截图]]
  ![[Pasted image 20260119144955.png]]
	**Eq.1: Patch Embedding:** 
		给Transformer Encoder准备输入序列，$x_{class}$ 对应0号patch ([CLS] token), 是手动加入的，不带图片信息的空向量。$x_p^n E$ 把切好的每一个patch通过矩阵E变换成向量。$E_{pos}$是位置编码，通过深度学习得到，能分辨patch之间的内在关联
		$z_0$ 是一个N+1 by D的矩阵 输入Transformer Encoder.
		$x_{class}$ is a learned input, taken as classification.
	**Eq.2 MSA 自注意力**, 表现在Transformer Encoder示意图的下半部分 **计算L次**
		[[LN]]: Layer Norm 归一化，让数据分布更稳定，避免梯度爆炸
		[[MSA]]: Multi-Head Self Attention: 每个向量和其他向量计算关联，0号patch在此“吸收”其他patch的信息
		[[Residual Connection]]: $z_l-1$ 残差链接，新信息与旧信息叠加，避免丢失信息。
	**Eq.3 MLP 特征进化**， 变现在示意图的上半部分，归一化后多层感知机，线性变换的基础上引入激活函数，最后残差连接[[Residual Connection]]。**计算L次**
	**Eq.4 最终输出** 取出 $z_l^0$ 进行归一化后作为“image representation $y$ ”输出  
	**MLP Head**，将Encoder的输出与分类的总数对应，改变vector size。例如分5类则768-->5. MLP Head后接softmax实现分类的概率计算。


- **代码核心逻辑 (GitHub Snippets)：**
- Colab code PyTorch version: [Colab Link](https://colab.research.google.com/drive/13BRAMPXb3QG062Lzi4i5BJqTSgS8uXtj#scrollTo=TBZ3pQXMskB8)

---

## 5. 本科生专项：基础补课与疑问 (To-Learn)

- [ ] **基础概念补课 (用 Obsidian 双链链接)：**  
    - [[Multihead Attention]] [[Attention]]-> 我的理解
	    - ![[Pasted image 20260121152120.png]]
	    - MHA和MHSA中的可学习参数
		    - 将输入映射到Query Key Value空间的三个矩阵, h为注意力头数$$W^Q\quad W^K\quad W^V \in (d_{model},d_{k}=d_{model}/h)$$
		    - 输出矩阵$$W^O\in (d_{model}, d_{model})$$$$Output=Concat(\sum_i head_i)W^O$$
- [ ] **遇到的坑：**
    
- [ ] 准备问师兄的问题：
    
    1. $E_{pos}$是什么 取代玛里寻找答案
---
## 6. 关联阅读与总结 (Summary & Links)

- **上一代工作：** [[之前的经典论文名]]
    
- **核心对比：** 与 [[SOTA方法名]] 相比，本作在 XX 场景下效果更好。
    
- **我的评价：** 这篇论文的思路是否可以借鉴到我现在的实验中？
    


