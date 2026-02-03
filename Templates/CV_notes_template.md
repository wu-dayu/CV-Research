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
# 📄 [Paper Study] {{title}} {{date}}

## 1. 快速预览 (Quick Read - 10min)
- **年份/会刊：** {{YEAR}} / #CVPR #ICCV #ECCV #Arxiv
- **领域标签：** #CV/{{分类/检测/分割/底层}} 
- **核心痛点 (Motivation)：**
    - 
- **核心贡献 (Key Idea)：**
    - 
- **代码仓库：** [GitHub Link]
- **是否值得精读：** 🟢 必读 / 🟡 略读 / 🔴 仅作参考

---

## 2. 内容整理/核心架构与数据流 (Architecture & Data Flow)

- **内容整理**

- **模型结构图：**
  
- **张量变化 (Tensor Shapes)：**
    -

---

## 3. 深度技术检查清单 (Direction-Specific Checklist)

### 🟢 分类/骨干网络 (Classification/Backbone)
- [ ] **Basic Block:** 最小重复单元长什么样？(残差结构/Transformer Block/Ghost Block?)
- [ ] **特征提取:** 它是如何权衡局部特征（Conv）和全局特征（Attention）的？
- [ ] **降采样:** 图像分辨率是如何一步步缩小的？(Stride Conv / Pooling / Patch Merging?)
- [ ] **性能指标:** 参数量 (Params) 和计算量 (FLOPs) 处于什么量级？

### 🔵 目标检测 (Object Detection)
- [ ] **Neck & Head:** 它是如何融合多尺度特征的？(FPN, PAN, BiFPN 等)
- [ ] **检测范式:** 是预设固定框 (Anchor-based) 还是直接预测中心 (Anchor-free)？
- [ ] **损失函数:** 分类 Loss (如 Focal Loss) 和回归 Loss (如 GIoU/DIoU) 分别是什么？
- [ ] **后处理:** 是否需要 NMS 过滤重复框？还是 End-to-end (如 DETR/RT-DETR)?

### 🟣 图像分割 (Segmentation)
- [ ] **解耦方式:** 它是如何从低分辨率恢复到高分辨率的？(上采样/反卷积/Skip Connection?)
- [ ] **分类粒度:** 它是如何实现像素级 (Pixel-wise) 精准分类的？
- [ ] **边界处理:** 对物体的边缘和细节处是否有特殊的优化处理？

### 🟠 底层视觉/增强 (Low-level Vision)
- [ ] **数据来源:** 训练数据是人工合成的（加噪/降采样）还是真实采集的？
- [ ] **图像先验:** 利用了什么图像知识？(如平滑性、稀疏性、非局部自相似性等)

- **数据归纳偏差 (Inductive Bias)**：Transformer 相比 CNN 缺少平移不变性，这篇论文是如何补偿这一点的？（例如：Position Encoding）
    
- **计算复杂度**：该模型的瓶颈是在图像编码器（Encoder）还是提示解码器（Decoder）？这决定了它在实时部署时的表现。
    
- **损失函数函数形式**：为什么作者选择了这个特定的组合（如 Dice Loss + Focal Loss）？

---

## 4. 数学表达与代码复现 (Math & Code)
- **核心公式：**

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
    - 
- [ ] **遇到的坑：** 
    
- [ ] 准备问师兄的问题：
    
---

## 6. 关联阅读与总结 (Summary & Links)

- **上一代工作：** 
    
- **核心对比：** 
    

