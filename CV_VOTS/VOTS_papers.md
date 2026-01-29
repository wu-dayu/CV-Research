  大纲：
1. 基础视觉与多模态底座
	- ViT将Transformer引入视觉领域的开山之作，将图片切patch处理。
	- Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, 等. 《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》. arXiv:2010.11929. 预印本, arXiv, 2021年6月3日. [https://doi.org/10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929).
	
	- RepViT, 从 ViT 的角度重新审视轻量级 CNN
	- Wang, Ao, Hui Chen, Zijia Lin, Jungong Han和Guiguang Ding. 《RepViT: Revisiting Mobile CNN From ViT Perspective》. arXiv:2307.09283. 预印本, arXiv, 2024年3月14日. [https://doi.org/10.48550/arXiv.2307.09283](https://doi.org/10.48550/arXiv.2307.09283).
	
	- DINOv2; Serlf-supervised vision transformer. Meta推出的自监督学习模型，特征提取能力极强，无需标签即可分割物体。
	-  Oquab, Maxime, Timothée Darcet, Théo Moutakanni, 等. 《DINOv2: Learning Robust Visual Features without Supervision》. arXiv:2304.07193. 预印本, arXiv, 2024年2月2日. [https://doi.org/10.48550/arXiv.2304.07193](https://doi.org/10.48550/arXiv.2304.07193)
	
	
	- DINOv3
	- Siméoni, Oriane, Huy V. Vo, Maximilian Seitzer, 等. 《DINOv3》. arXiv:2508.10104. 预印本, arXiv, 2025年8月13日. [https://doi.org/10.48550/arXiv.2508.10104](https://doi.org/10.48550/arXiv.2508.10104).
	
2. 视觉任务专家：检测与分割（Detection&Segmentation）

	- DINO（检测）**DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** 当前最强的检测baseline之一，引入去噪训练（注：与meta的自监督DINO不同）
	- Zhang, Hao, Feng Li, Shilong Liu, 等. 《DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection》. arXiv:2203.03605. 预印本, arXiv, 2022年7月11日. [https://doi.org/10.48550/arXiv.2203.03605](https://doi.org/10.48550/arXiv.2203.03605).
	
	
	
	- SAM: Meta推出的分割基础模型，支持点、框、文本提示分割一切。
	- Kirillov, Alexander, Eric Mintun, Nikhila Ravi, 等. 《Segment Anything》. arXiv:2304.02643. 预印本, arXiv, 2023年4月5日. [https://doi.org/10.48550/arXiv.2304.02643](https://doi.org/10.48550/arXiv.2304.02643).
	
	- SAM2：在SAM基础上增加了视频分割和追踪能力，速度更快
	- Ravi, Nikhila, Valentin Gabeur, Yuan-Ting Hu, 等. 《SAM 2: Segment Anything in Images and Videos》. arXiv:2408.00714. 预印本, arXiv, 2024年10月28日. [https://doi.org/10.48550/arXiv.2408.00714](https://doi.org/10.48550/arXiv.2408.00714).
	
	- SAM3
	- Carion, Nicolas, Laura Gustafson, Yuan-Ting Hu, 等. 《SAM 3: Segment Anything with Concepts》. arXiv:2511.16719. 预印本, arXiv, 2025年11月20日. [https://doi.org/10.48550/arXiv.2511.16719](https://doi.org/10.48550/arXiv.2511.16719).


---
**Note Templates:**


1. **略读模板**
	# [Quick Read] {{Year}} - {{Title}}
	- **领域：** #CV/{{方向}}  **代码：** [Link/None]
	- **核心痛点 (What's the problem?):** - 前人方法的缺陷：
	- **核心贡献 (Key Idea):** - 它是怎么解决的？（一句话描述核心创新）
	- **主要结果 (Performance):** - 指标：(例如 mAP 提升了多少)
		- 是否值得精读？ [ ] Yes / [ ] No / [ ] 待定
	- **一句话总结：**	
	
2. **精读模板**
	# [Deep Dive] {{Title}}
	## 1. 核心架构图 (Architecture)
	技巧：在 Zotero 里截图，直接拖到这里。
	- [截图/流程图]
	- **输入输出尺寸变换 (Tensor Shape)：**
    - Input: (B, 3, H, W) -> ... -> Output: (...)
	## 2. 核心数学表达 (Methodology)
	- **Loss 函数：**
	- **关键算子/公式：** (记录你觉得精妙的公式)
	## 3. 实验细节 (Experiments)
	- **数据集：** #COCO / #ImageNet / #Custom
	- **消融实验 (Ablation Study)：** 论文里哪个模块最起作用？
	- **对比：** 它比 SOTA 强在哪里？
	## 4. 复现价值 (Implementation)
	- **难点：** (环境配置、显存需求、训练时长)
	- **代码关键行：** (贴一段你在 GitHub 里看到的实现逻辑)
	## 5. 个人思考与疑问
	- 为什么不直接用 X 而要用 Y？
	- 这里的参数 $\alpha$ 是调出来的还是有理论依据？
	6.## 📖 基础知识扫盲 (Concepts to Learn)
	- [ ] **术语1:** [[池化]] -> 状态：[已查阅/掌握]
	- [ ] **术语2:** [[批量归一化 (BN)]] -> 状态：[待深入理解]
	- [ ] **代码坑:** 为什么 PyTorch 里 `view` 和 `reshape` 不一样？
	
3. **细分方向的“深度精读”专有清单**
	你可以根据论文方向，将以下内容追加到上面的精读模板中：
	
	#### **A. 目标检测 (Object Detection) 专用**
	
	- [ ] **Backbone:** 用的什么特征提取器？(ResNet? Swin?)
	    
	- [ ] **Neck:** 怎么融合多尺度特征的？(FPN? PAN? BiFPN?)
	    
	- [ ] **Head:** 是 Anchor-based 还是 Anchor-free？
	    
	- [ ] **正负样本分配 (Label Assignment):** 怎么确定哪个框是前景？(ATSS? SimOTA?)
	    
	
	#### **B. 图像分割 (Segmentation) 专用**
	
	- [ ] **解耦方式:** 它是怎么恢复空间分辨率的？(上采样? 反卷积?)
	    
	- [ ] **感受野:** 怎么增大感受野的？(空洞卷积 Dilated Conv? 全局注意力?)
	    
	- [ ] **辅助损失:** 是否在中间层加了辅助的 Loss？
	    
	
	#### **C. 视觉 Transformer (ViT/Attention) 专用**
	
	- [ ] **Patch Embedding:** 图像是怎么切块的？
	    
	- [ ] **Positional Encoding:** 它是怎么记住像素位置信息的？(绝对位置? 相对位置? 旋转位置?)
	    
	- [ ] **Complexity:** 计算量是序列长度的平方级还是线性级？

