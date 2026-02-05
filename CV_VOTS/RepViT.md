
# 📄 [Paper Study] RepViT 2026-01-21

## 1. 快速预览 (Quick Read - 10min)
- **年份/会刊：** {{2024}} /  #CVPR #Arxiv
- **领域标签：** #CV/{{底层}} 
- **核心痛点 (Motivation)：**
    - CNN与ViT作为两个CV领域经典的backbone，当前没有有效的研究将ViT的特征吸收进入CNN网络结构中。
- **核心贡献 (Key Idea)：**
    - 论文通过**结构重参数化（Structural Re-parameterization）**，将 ViT 的先进宏观架构（如 Meta-Former 结构）注入到轻量级 CNN 中，在不增加推理成本的前提下显著提升性能。如图：
		    ![[Pasted image 20260124161430.png]]![[Pasted image 20260124161557.png]]
- **代码仓库：** [GitHub Link](https://github.com/THU-MIG/RepViT)
- **是否值得精读：** 🟢 必读 

---

## 2. 核心架构与数据流 (Architecture & Data Flow)

- **Overall Architecture: RepViT Blocks in each stage 1:1:7:1**
	-![[Pasted image 20260122164937.png]]
	
- **RepViT Block Design**
	![[Pasted image 20260122165304.png]]
- **Macro Design**
	![[Pasted image 20260122165458.png]]
- **张量变化 (Tensor Shapes - 重点)：**
    ![[7bcdfa32407c89e9bbff47ed5b7a661f.jpg]]![[153e2ddb0d5293b3604d68b558a5fa60.jpg]]
    ![[99d43afec438e9e4cdeaaf95b8731253 2.jpg]]

---

## 3. 深度技术检查清单 (Direction-Specific Checklist)

### 🟢 分类/骨干网络 (Classification/Backbone)
- [ ] **Basic Block:** 最小重复单元长什么样？(残差结构/Transformer Block/Ghost Block?)
- [ ] **特征提取:** 它是如何权衡局部特征（Conv）和全局特征（Attention）的？
- [ ] **降采样:** 图像分辨率是如何一步步缩小的？(Stride Conv / Pooling / Patch Merging?)
- [ ] **性能指标:** 参数量 (Params) 和计算量 (FLOPs) 处于什么量级？

---

## 4. 数学表达与代码复现 (Math & Code)
- **代码核心逻辑 (GitHub Snippets)：**
```
\\wsl$\Ubuntu\home\wudayu\CV_research\RepViT
```

---

## 5. 本科生专项：基础补课与疑问 (To-Learn)

- [ ] **基础概念补课 (用 Obsidian 双链链接)：**
    - [[Teacher Model]] & [[Distillation]] 
	    ![[Pasted image 20260122155405.png]]
	    ![[Pasted image 20260122155454.png]]
	    “Knowledge distillation (Hinton et al., 2014) aims at reproducing the output of a large model with a smaller model by minimizing some distance between both outputs for a set of given inputs.” (Oquab 等, 2024, p. 7)
    - [[Adam Optimizer]] & [[AdamW Optimizer]]
	    ![[Pasted image 20260122112643.png]]
	- [[深度可分离卷积]] i.e. [[Depthwise Separable Convolution]]
		卷积是对应位置元素相乘并求和
	  ![[00da3368ef918026e0163cd60e3946c6.jpg]]
	- [[FFN]] 与 [[Down Sampling]] 等维度变换的意义
		![[Pasted image 20260122202029.png]]
- [ ] **遇到的坑：** 
	- Python 3.10 + CUDA 12.8 + PyTorch Nightly 
---

## 6. 关联阅读与总结 (Summary & Links)

- **上一代工作：**
    
- **核心对比：** 
    

