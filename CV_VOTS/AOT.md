# 📄 [Paper Study] {{Associating Objects with Transformers for Video Object Segmentation}} {{2026-3-18}}

## Outline
- **年份/会刊：** {{2021}} / #Arxiv
- **领域标签：** #CV/{{分割}} 
- **核心痛点 (Motivation)：**
    - 当时的SOTA模型全部是single-object tracking
- **核心贡献 (Key Idea)：**
    - An identification mechanism to associte multiple targets into the same high-dimensional embedding space. Thus, we can simultaneously process multiple objects’ matching and segmentation decoding as efficiently as processing a single object. 将multiple targets映射至同一高维向量空间，能像处理单个obj一样同时跟踪多个obj
    - a Long Short-Term Transformer is designed for **constructing hierarchical matching and propagation**.每个LSTT块利用一个长期注意力与第一个帧的embedding进行匹配，并利用一个短期注意力与几个相邻帧的embedding进行匹配。
- 核心公式
	- ![[Pasted image 20260319142335.png]]
	- ![[Pasted image 20260319142355.png]]
- 架构图
	- ![[Pasted image 20260318193001.png]]
	- ![[Pasted image 20260318190755.png]]
	- ![[Pasted image 20260318214400.png]]
- 局限性：
	- **在 AOT-T（Tiny）、AOT-S（Small）和 AOT-B（Base）变体中**：为了保持平稳且高效的运行速度，模型只将**第一帧**作为长期记忆帧。
	- **在 AOT-L（Large）变体中**：采用了记忆读取策略，除了第一帧外，模型还会按照设定的间隔（每隔 $\delta$ 帧）将**历史预测帧**不断存储到长期记忆中，从而为当前帧提供更丰富的参考信息。We set δ to 2/5 for training/testing.
		- 在历史预测帧的保存中没有筛选，multi-obj可能存在遮挡问题，噪声引入；长视频处理时memory frames臃肿，Long term transformer易失效，易OOM
- **代码仓库：** [GitHub Link](https://github.com/yoxu515/aot-benchmark) 
