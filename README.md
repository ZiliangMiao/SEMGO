# Stacked Ensemble of Metamodels for Expensive Global Optimization (SEMGO)
## Contributors
This work is devloped by [Ziliang Miao](https://github.com/ZiliangMiao), [Buwei He](https://github.com/Buwei-He), [Hubocheng Tang](https://github.com/henyoujingshen) and [Jixiang Chen](). Thanks to the help of [Prof.Zhenkun Wang](https://scholar.google.com/citations?user=r9ezy2gAAAAJ&hl=zh-CN&oi=ao) (corresponding author) and the support of the School of System Design and Intelligent Manufacturing ([SDIM](https://sdim.sustech.edu.cn/)) at the Southern University of Science and Technology ([SUSTech](https://www.sustech.edu.cn/)). If you have any questions, please get in touch with Ziliang via ziliang.miao26@gmail.com.

## Introduction
This paper proposes a novel global optimization method, namely Stacked Ensemble of Metamodels for Expensive Global Optimization (SEMGO), which aims to improve the accuracy and robustness of the surrogate for expensive optimization problems. Since the existing metamodel ensemble methods leverage particular linear weighting strategies, they are likely to result in bias when facing various problems. SEMGO employs a learning-based second-layer model to combine the predictions of each metamodel in the first layer adaptively. The proposed SEMGO is compared with three state-of-the-art metamodel ensemble methods on seventeen widely used benchmark problems. The results show that SEMGO performs the best. It is also applied to solve a practical chip packaging design problem and get a better solution than the previous method.
### Key Words
Ensemble Learning, Stacking, Ensemble of Metamodels(KRG, RBF, PRS), Expensive Global Optimization, Chip Packaging Optimization

## Contributions
* To the best of our knowledge, we are the first to introduce the ensemble learning method in expensive global optimization problems, which effectively avoids the potential bias in ensemble methods with the linear weighting strategy.
* We proposed an improved strategy for the training process of stacking, by dividing the k-fold training process into k iterations. Compared with the original implementation of stacking, it significantly reduces the additional model training cost incurred by k-fold training.
* We compared the proposed SEMGO with three state-of-the-art methods on seventeen well-known benchmark problems and applied SEMGO to a real-world chip packaging problem. The empirical result shows good robustness of SEMGO on various problems.

## Proposed Method: SEMGO
<!-- <div  align="center">    
</div> -->
<img src="Pictures/SEMGO Workflow.png" height="40%" width="40%" align=center/>
<img src="Pictures/Training Process.png" height="75%" width="75%" align=center/>
<img src="Pictures/Second-layer Prediction.png" height="75%" width="75%" align=center/>




