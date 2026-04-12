🛳️ Titanic Survival Prediction (泰坦尼克号生还者预测)
本项目是基于 Kaggle 经典竞赛 Titanic - Machine Learning from Disaster 的机器学习实战项目。项目使用了统计学习方法中的经典分类算法（随机森林与支持向量机）来预测泰坦尼克号乘客的生还情况。

📁 文件结构
当前仓库包含以下文件和目录：

数据文件
train.csv: 训练集数据，包含乘客特征及是否生还的标签（Survived）。

test.csv: 测试集数据，包含乘客特征，需预测其生还情况。

gender_submission.csv: Kaggle 提供的官方示例提交文件（假设所有女性都生还）。

源代码
RandomForestClassifier.py: 基于 随机森林 (Random Forest) 算法构建预测模型的 Python 脚本。包含数据预处理、模型训练及结果预测。

SVM.py: 基于 支持向量机 (Support Vector Machine) 算法构建预测模型的 Python 脚本。

预测结果
submission.csv: 由 RandomForestClassifier.py 运行生成的最终预测结果文件，可直接提交至 Kaggle。

svm_submission.csv: 由 SVM.py 运行生成的最终预测结果文件，可直接提交至 Kaggle。

配置文件
.gitignore: Git 忽略配置文件，用于忽略不需要提交至版本库的本地环境文件。

🚀 快速开始
1. 环境依赖
运行本项目代码需要 Python 环境，并安装以下主要第三方库：

pandas (数据分析与处理)

scikit-learn (机器学习算法库)

您可以运行以下命令安装所需依赖：

Bash
pip install pandas scikit-learn
2. 运行代码
克隆本项目到本地后，确保您的终端或命令行工具处于当前项目目录下。

运行随机森林模型：

Bash
python RandomForestClassifier.py
运行结束后，会在当前目录下生成 submission.csv 文件。

运行 SVM 模型：

Bash
python SVM.py
运行结束后，会在当前目录下生成 svm_submission.csv 文件。

📊 模型介绍
随机森林 (Random Forest): 一种集成学习方法，通过构建多个决策树并进行投票来决定最终分类结果，对特征缺失和非线性关系有较好的鲁棒性。

支持向量机 (SVM): 一种强大的分类算法，旨在寻找一个最优超平面将不同类别的数据点进行划分，适用于高维空间数据。

🔗 相关链接
Kaggle: Titanic - Machine Learning from Disaster
