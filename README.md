# Stacked Ensemble of Metamodels for Expensive Global Optimization (SEMGO)
## Contributors
This work is devloped by [Ziliang Miao](https://github.com/ZiliangMiao), [Buwei He](https://github.com/Buwei-He) and [Hubocheng Tang](https://github.com/henyoujingshen). Thanks to [Jixiang Chen]() for his contributions to the paper discussion and revision. Thanks to the help of [Prof.Zhenkun Wang](https://scholar.google.com/citations?user=r9ezy2gAAAAJ&hl=zh-CN&oi=ao) (corresponding author) and the support of the School of System Design and Intelligent Manufacturing ([SDIM](https://sdim.sustech.edu.cn/)) at the Southern University of Science and Technology ([SUSTech](https://www.sustech.edu.cn/)). If you have any questions, please get in touch with Ziliang via ziliang.miao26@gmail.com.

## Introduction
This paper proposes a novel global optimization method, namely Stacked Ensemble of Metamodels for Expensive Global Optimization (SEMGO), which aims to improve the accuracy and robustness of the surrogate for expensive optimization problems. Since the existing metamodel ensemble methods leverage particular linear weighting strategies, they are likely to result in bias when facing various problems. SEMGO employs a learning-based second-layer model to combine the predictions of each metamodel in the first layer adaptively. The proposed SEMGO is compared with three state-of-the-art metamodel ensemble methods on seventeen widely used benchmark problems. The results show that SEMGO performs the best. It is also applied to solve a practical chip packaging design problem and get a better solution than the previous method.

## Proposed Method: SEMGO
<img src="Pictures/SEMGO Workflow.png" >
