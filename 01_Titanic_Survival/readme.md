# 🚢泰坦尼克号生还者预测 (Titanic Survival Prediction)

本项目是基于 Kaggle 经典竞赛 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) 的统计学习实战练习。目标是利用乘客的个人信息（如年龄、性别、舱位等级等）来预测其在泰坦尼克号事故中是否生还。

## 🏆 Kaggle 提交成绩 (Scores)

记录模型迭代过程中的得分变化：

| 提交次数 | 模型/策略 | Public Score | 状态 |
| :---: | :--- | :---: | :---: |
| 1 | Random Forest模型 | 0.77033 | 历史记录 |
| 2 | SVM模型 | **0.77751** | 最佳提交 🎉 |

## 📁 目录结构

* **数据集 (Data)**
    * `train.csv`: 训练集，包含特征及标签 $y$（生还与否）。
    * `test.csv`: 测试集，用于评估模型性能并生成最终提交结果。
    * `gender_submission.csv`: 官方提供的预测示例。
* **模型实现 (Models)**
    * `RandomForestClassifier.py`: 使用随机森林（Random Forest）集成算法实现的分类模型。
    * `SVM.py`: 使用支持向量机（Support Vector Machine）实现的分类模型。
* **预测结果 (Submissions)**
    * `submission.csv`: 随机森林模型生成的预测结果。
    * `svm_submission.csv`: SVM 模型生成的预测结果。
* **其他**
    * `.gitignore`: 忽略非必要文件（如虚拟环境、缓存等）。

## 🛠 算法原理

本项目主要采用了以下两种统计学习方法：

### 1. 支持向量机 (SVM)
支持向量机通过寻找一个最优超平面，使得不同类别的样本点到该超平面的几何间隔最大。对于线性不可分数据，通过核函数将特征映射到高维空间。
其核心优化问题为：
$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
$$
满足约束条件：
$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i=1, 2, \dots, N
$$

### 2. 随机森林 (Random Forest)
随机森林是一种基于决策树的 Bagging 集成学习方法。它通过对样本和特征进行随机采样，构建多棵决策树，并利用投票法（Voting）决定最终分类结果。
设第 $k$ 棵树的预测结果为 $h_k(\mathbf{x})$，则最终预测结果 $H(\mathbf{x})$ 为：
$$
H(\mathbf{x}) = \text{argmax}_Y \sum_{k=1}^K I(h_k(\mathbf{x}) = Y)
$$

## 🚀 使用说明

### 1. 安装依赖
确保已安装 Python 以及相关数据科学库：
```bash
pip install pandas numpy scikit-learn
