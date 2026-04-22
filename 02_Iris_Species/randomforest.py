# =========================================
# 鸢尾花数据集分类实验（单文件版）
# =========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_csv("IRIS.csv")

# 查看前5行
print("数据前5行：")
print(data.head())

# 查看列名
print("\n列名：")
print(data.columns)

# 假设最后一列是标签列 Species
target_col = "Species"

# 如果有 Id 列，需要删掉
if "Id" in data.columns:
    data = data.drop(columns=["Id"])

# 构造特征和标签
X = data.drop(columns=[target_col])
y = data[target_col]

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 建立模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("\n准确率：", accuracy_score(y_test, y_pred))
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()