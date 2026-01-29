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
# 📄 [Paper Study] DINOv3 2026-01-27

## 1. 快速预览 (Quick Read - 10min)
- **年份/会刊：** {{2025}} /  #Arxiv
- **领域标签：** #CV/{{底层}} 
- **核心痛点 (Motivation)：**[DINOv2的weakness](https://docs.google.com/document/d/1CaADcs6hwx3n9SCqUfuVeMieiPUqMDzkomSHWTPnHe0/edit?tab=t.0)
- **核心贡献 (Key Idea)：**
- **代码仓库：** [GitHub Link](https://github.com/facebookresearch/dinov3)
- **是否值得精读：** 🟢 必读 

---

## 2. 论文内容整理
- 3.1 数据准备
	- 数据分类
		1. **基于聚类的筛选（Clustering-based）**：通过聚类算法平衡不同类别的分布，提升数据的**多样性（Diversity）**。这有助于模型学习更广泛的视觉概念，而不被某些常见类别淹没 
		2. **基于检索的筛选（Retrieval-based）**：通过检索与特定任务相关的图像，提升数据集的**有用性（Usefulness）**。这确保了模型能接触到大量与实际应用场景（如常见物体、场景）相关的图像
		3. “we use raw publicly available computer vision datasets including ImageNet1k (Deng et al., 2009), ImageNet22k (Russakovsky et al., 2015), and Mapillary Street-level Sequences (Warburg et al., 2020). This final part allows us to optimize our model’s performance, following Oquab et al. (2024).” 
	- 数据混合
		Batch的组成：**Batch 组成**：在训练过程中，**10%** 的 Batch 是纯 ImageNet-1k 数据（同质化 Batch），其余 **90%** 则是混合了所有数据源的异构 Batch 。
	- **Batch 的真实结构（张量维度）** by Gemini
		如果你设置 `Batch Size = B`（比如 $B=64$），那么在一个训练步骤中：
		- 原始图片数量：$B$ 张。
		- **送到 GPU 的张量（Tensor）**：
			- **Global Tensor**: 形状为 $(2B, 3, 224, 224)$。因为每张原始图贡献了 2 个全局块。
			- **Local Tensor**: 形状为 $(8B, 3, 96, 96)$。因为每张原始图贡献了 8 个局部块。
			 注：对于Gram Loss，模型**可能**保存了高/低分辨率的两套Global Tensor
			 **总结：** 在代码层面，通常会把所有的 Crops 拼接在一起。所以一个 Batch 实际上是由 **$B \times (2+8)$** 个图像块组成的巨大张量。
- 3.2 自监督大规模训练
	1. 学习目标的融合与改进 (Learning Objective)
		研究者并没有完全推翻 DINOv2 的架构，而是进行了精细的优化 ：
		- **混合损失函数**：继续使用 **DINO loss**（全局图像级目标）和 **iBOT loss**（局部 Patch 级目标）的组合，以平衡语义理解和细节感知 。
		- **Sinkhorn-Knopp 替代 Centering**：在两个损失函数中都引入了 SwAV 中的 Sinkhorn-Knopp 算法，这比 DINO 原本的 Centering 方法更能稳定大规模训练的聚类过程 。
		- **层归一化 (Layer Norm) 的位置优化**：在 Backbone 输出后、Head 输入前添加了专门的 Layer Norm，实验证明这能显著提升 ImageNet kNN 分类的稳定性，并增强稠密预测任务（如分割）的表现 。
		- Pretraining 阶段的损失函数$$L_{pre}=L_{DINO}+L_{iBOT}+0.1*L_{DKleo}$$
	2. Update Model Architecture
		- A custom variant of [[RoPE]]
		- [[RoPE-box Jittering]]
	3. Optimization
		- Constant learning rate, weight decay, and teacher EMA momentum.
		- [[Multi-crop Strategy]]
			这是 DINO 系列（v2/v3）增强特征鲁棒性的核心手段。
			- **Global/Local Crops 是干什么的**：
			    - **Global Crops（2个）**：大尺度的切片（覆盖图片的大部分，如 256x256）。目的是让模型学习**整体语义**（这是只猫）。
			    - **Local Crops（8个）**：小尺度的随机切片（只看局部细节，如 112x112）。目的是让模型学习**局部细节**（这是猫的胡须）。
			    - **训练逻辑**：模型被迫去猜测：这 8 个局部小块和那 2 个大块是不是属于同一张图？通过这种“以局部推全局”的过程，模型学到了极强的特征表达。
			- **参数解释**：
			    - **Side length（边长）**：就是切片的大小。Global 是 256 像素，Local 是 112 像素。
			    - **Effective sequence length（有效序列长度）**：指进入 Transformer 的 **Token 数量**。
			        - 计算方式：$\text{Token 数量} = (\text{边长} / \text{Patch Size})^2$。
			        - 因为 DINOv3 的 Patch Size 变了，所以调整边长（256/112）是为了保证算出来的 Token 数量和 DINOv2 一样多，这样显存占用和计算量就保持稳定。
			    - **在模型的哪里用到**：
			        - 在**数据预处理阶段**（DataLoader）进行切片。
			        - 在 **Transformer 的输入层**（Patch Embedding）将这些切片变成 Token 序列。
- 4. 本模型最大创新---Gram Anchoring
	- 问题的出现
		“Throughout our experiments, we have identified a relative independence between learning strong discriminative features and maintaining local consistency, as observed in the lack of correlation between global and dense performance. While combining the global DINO loss with the local iBOT loss has begun to address this issue, we observe that the balance is unstable, with global representation dominating as training progresses.” 
	- Gram Matrix
		- $G=X\cdot X^T$ ; dim=($d,d$)
		- [Gram 矩阵元素详解与几何意义](https://docs.google.com/document/d/1rjdu0pAiD7k2zz7k4oI5_aThhL9hUwRgeBPtVPF08Uw/edit?tab=t.0)
			- $$G_{i,j} = \frac{1}{N} \sum_{k=1}^{N} F_{k,i} \cdot F_{k,j}$$
				- 它取出了特征图中所有的 Token
				- 它关注的是第 $i$ 个维度和第 $j$ 个维度
				- 它把这 $N$ 个位置上，这两个维度的数值分别相乘并求平均（此处的N同$p$, an image is composed of $p$ patches/ $N$ tokens）
				- 物理意义：这其实就是在计算特征通道 $i$ 和通道 $j$ 之间的协方差（相关性）
				- 如果 $G_{i,j}$ 的值很高，说明在这张图中，i通道表示的特征经常和d通道表示的特征一起出现
	-  损失函数公式：$$L_{Gram}=\left|\left|X_S\cdot X_S^T-X_G\cdot X_G^T\right|\right|_F^2$$
		其中$X_S$和$X_T$矩阵是$p×d$ matrix of **$L_2$-normalized** local features of student and teacher.
		Computed only on global crops; start after 1M iterations
	- 初步发现：“Interestingly, we observe that the late application of LGram still manages to “repair” very degraded local features.” 
	- 在Refinement Step每10k interations 更新Gram Teacher与main EMA teacher一致。 
- 4.3 Leveraging Higher-Resolution Features in Gram Anchoring
	- We only compute Gram loss on the global crops
	- 确定采样区域，双路并行采样
		- 数据加载器在原始高分辨率大图上随机选定一个矩形区域 $R$
		- **学生路径 (Student Path)**：将区域 $R$ 裁剪出来并缩放到 **$224 \times 224$**。随后进行颜色抖动、高斯模糊等常规增强。
		- **老师路径 (Gram Teacher Path)**：将**同一个区域 $R$** 裁剪出来并缩放到 **$448 \times 448$**（甚至更高）。
	- 特征提取与下采样平滑
		- 学生路径输出低分辨率特征图 $X_S$
		- 老师路径输出高分辨率特征图，并利用**Bicubic插值**将高分辨率图转化为低分辨率
			*这一步通过插值平均了周围像素，使得得到的特征图 $F_{tea}^{smooth}$ ($X_T$)既保留了高分辨率带来的边缘细节，又消除了单像素的随机噪声。*
	- 计算Gram Loss
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
  $$L_{Gram}=\left|\left|X_S\cdot X_S^T-X_G\cdot X_G^T\right|\right|_F^2$$
 Initial train phase:$$L_{Pre} = L_{DINO} + L_{iBOT}+0.1*L_{DKoleo}$$
  Refinement Step:$$L_{Ref} = \omega_D L_{DINO} + L_{iBOT}+\omega_{DK}L_{DKoleo}+\omega_{Gram}L_{Gram}$$
- **代码核心逻辑 (GitHub Snippets)：**
```python
# 记录复现时发现的论文核心函数实现逻辑
def forward(self, x):
    # 例如：这里是论文提到的 Residual Connection
    identity = x
    out = self.conv_layers(x)
    return out + identity
````

---

## 5. 本科生专项：基础补课与疑问 (To-Learn)

- [ ] **基础概念补课 (用 Obsidian 双链链接)：**
    - [[Features]] & [[Dense Features]]:
	    - **Features** 是图像原始像素（RGB 数值）经过模型编码后产生的**高维向量表征**
	    - **Dense Features** 特指模型输出的、具有**高空间分辨率**的特征分布，它关注的是图像中的“每一个局部”甚至是“每一个像素”。
	    - **Global Feature（全局特征）**：整个图像最后被压缩成一个向量（通常是 `[CLS]` token）。它只告诉你“这张图里有一只鸟”，但它不关心鸟在哪。
		- **Dense Features（稠密特征）**：保留了 Patch 级别或像素级别的特征图（Feature Map）。它告诉你“坐标 (x,y) 处是鸟的喙”、“坐标 (z,w) 处是背景的树叶”。
	- [[Feature map]] 
		- 关键超参数 Patch Size，14×14 or 16×16，定义了“像素密度”d		
		- ![[Pasted image 20260127214653.png]]
		- ![[Pasted image 20260127214800.png]]
	- [[RoPE]] & [[RoPE-box Jittering]]
			Section 3.2 Update Model Architecture中讲的是模型如何理解图像中各个部分（Patch）的**位置关系**。
		- [【硬核】手撕RoPE旋转位置编码推导，嘎嘎简单，通俗易懂! from Bilibili]（https://www.bilibili.com/video/BV1FjrCBdESo/?share_source=copy_web&vd_source=0c211e12ad1b45ed8f807de27691bac9）
		- RoPE Jittering 公式与逻辑解析 （长）
			在 DINOv3 中，RoPE 的核心是将 Patch 的 2D 坐标 映射为 复数旋转。为了让你彻底理解 Jittering 是如何进入公式的，我们分三步拆解。
			### 1. 基础：2D RoPE 的数学表达
			对于一个位于坐标 $(x, y)$ 的 Patch，它的特征向量（维度为 $d$）会被拆分成很多对二维子向量。对于其中一对分量 $(z_1, z_2)$，RoPE 的转换公式为：$$\text{RoPE}(z, x, y) = \begin{pmatrix} \cos(x \cdot \omega_i) & -\sin(x \cdot \omega_i) \\ \sin(x \cdot \omega_i) & \cos(x \cdot \omega_i) \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \end{pmatrix}$$
			
			（注：实际实现中，$d$ 维向量的前一半通常处理 $x$ 坐标，后一半处理 $y$ 坐标，或者交替处理。）
			这里的核心是相位角 $\theta$：$$\theta_x = x \cdot \omega_i, \quad \theta_y = y \cdot \omega_i$$
			其中 $\omega_i$ 是一个预设的频率常数（控制旋转的快慢）。
			
			---
			
			1. Jittering 如何进入公式？
			
			在 DINOv3 的坐标框（Coordinate Box）逻辑下，原始坐标 $(x, y)$ 是归一化到 $[-1, 1]$ 的。Jittering 的逻辑就是在这个坐标上乘一个随机缩放因子 $s$。
			
			公式演变为：
			$$\theta_{jittered} = (x \cdot s) \cdot \omega_i$$
			其中 $s \in [0.5, 2]$ 是每个训练样本随机采样的缩放倍数。
			
			---
			
			1. 结合公式看 Jittering 的具体逻辑
			我们可以通过公式推导看到 Jittering 扮演的三个角色：
			#### A. 改变“空间频率” (Changing Frequency)
			从公式上看，$(x \cdot s) \cdot \omega_i$ 也可以写成 $x \cdot (s \cdot \omega_i)$。
			- 逻辑：Jittering 实际上是在训练过程中随机改变位置编码的频率。
			- 效果：如果 $s$ 很大，相位角随距离变化极快，模型感觉两个 Patch 离得很远（高频）；如果 $s$ 很小，相位角变化慢，模型感觉它们离得很近（低频）。这强迫模型学会识别不同“缩放倍数”下的空间模式。
			
			#### B. 保持相对关系的不变性 (Relative Property)
			当我们计算两个 Patch（坐标为 $pos_1, pos_2$）之间的注意力时，公式会变成：$$\text{Attn} \propto \cos((pos_1 \cdot s - pos_2 \cdot s) \cdot \omega_i) = \cos(s \cdot \Delta pos \cdot \omega_i)$$
			- 逻辑：Jittering 缩放了相对距离 $\Delta pos$。
			- 效果：模型在训练中看到的 $\Delta pos$ 是不断晃动的，这防止了模型去死记硬背“当角度差为 30° 时，对应的像素距离是多少”，而是让模型对比例更加敏感。
			
			#### C. 坐标框的重采样 (Box Re-sampling)
		
			在代码实现层级，Jittering 并不是直接改像素，而是改参考系
			1. 取一个 Patch。
			2. 计算它在原始 $[-1, 1]$ 框里的中心点 $(x, y)$。
			3. 执行 Jittering：将坐标映射变为 $x_{new} = x \cdot s, y_{new} = y \cdot s$。
			4. 将 $x_{new}, y_{new}$ 代入上述旋转矩阵。
			
		- **通俗解释：RoPE 与位置编码的改进**
			*通过随机改变 $s$，DINOv3 实际上是在告诉模型：“不要相信绝对的旋转角度，要相信相对的旋转趋势。” 这种训练方式让 7B 的大模型在面对超高分辨率图片或极小物体时，依然能通过 RoPE 准确感知到它们在归一化空间中的相对位置。*
		- **什么是 RoPE（旋转位置嵌入）**：
		    - **原理**：传统的 Transformer 给每个 Patch 加一个固定的数字（位置编码）**这样会污染原来的特征！！** RoPE 则不同，它通过“旋转”特征向量的维度来表示位置。
		    - **类比**：想象时钟的指针。绝对位置编码是告诉你现在是“3点”；而 RoPE 的旋转逻辑让你更容易算出“现在比刚才多了 15 分钟”。它擅长捕捉**相对位置**。
		- **什么是自定义变体（Custom Variant）**：
		    - **原理**：DINOv3 将图片定义为一个从 $[-1, 1]$ 的标准方框。无论图片多大，左上角都是 $(-1, -1)$，中心是 $(0, 0)$。
		    - **对象**：执行对象是 **Multi-head Attention** 中的 Query 和 Key。在算注意力得分之前，先根据 Patch 之间的相对坐标进行“旋转”变换。
		- **什么是 RoPE-box Jittering（位置框抖动）**：
		    - **原理**：在训练时，随机把那个 $[-1, 1]$ 的框缩放到 $[ -s, s ]$（比如变成 $[-0.5, 0.5]$ 或 $[-2, 2]$）。
		    - **作用**：让 DINOv3 无论面对高清图、缩略图还是长条图，都能保持极强的鲁棒性。
	- [[Downstream]]
		- **定义**：它不单指“分类”或“检测头”。它是一个统称，指模型在**预训练结束之后**，去完成那些**实际应用任务**（下游任务）时的表现。
		- **包含内容**：
		    1. **分类**：在特征上接一个简单的线性层。
		    2. **检测（Detection）**：接一个检测头（如 Faster R-CNN）。
		    3. **分割（Segmentation）**：接一个分割头。
		- **核心逻辑**：如果 DINOv3 的 "downstream performance" 好，说明这个“底座”提取的特征非常全能，稍微加个“头”就能在特定任务上拿高分。
	- [[Cosine Map]]
- [ ] **遇到的坑：**
- [ ] 准备问师兄的问题：
---

## 6. 关联阅读与总结 (Summary & Links)

- **上一代工作：** [[DINOv2]]
    
- **核心对比：** [DINOv2的weakness](https://docs.google.com/document/d/1CaADcs6hwx3n9SCqUfuVeMieiPUqMDzkomSHWTPnHe0/edit?tab=t.0)
    