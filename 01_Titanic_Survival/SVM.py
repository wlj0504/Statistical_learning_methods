import pandas as pd##pandas是一个库，可以读取 CSV、Excel、SQL 数据库等格式的数据
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # ★ SVM 必备利器：标准化工具

# ==========================================
# 1. 加载数据
# ==========================================
print("正在加载数据...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# ==========================================
# 2. 数据预处理
# ==========================================
def clean_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df


train_data = clean_data(train_data)
test_data = clean_data(test_data)

# ==========================================
# 3. 特征选择
# ==========================================
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

# ==========================================
# 4. 本地验证集划分
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# ★ 新增步骤：数据标准化 (Scaling)
# ==========================================
# 将数据转换为均值为0，方差为1的标准正态分布
scaler = StandardScaler()

# 注意：只能用训练集来 fit（计算均值和方差），然后转换训练集和验证集
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==========================================
# 5. 训练 SVM 模型
# ==========================================
# kernel='rbf' 即使用径向基（高斯）核函数，对应书中 7.3 节的非线性支持向量机
# C 是惩罚参数，控制对误分类样本的容忍度
model = SVC(kernel='rbf', C=1.0, random_state=42)

# 在标准化后的训练集上训练
model.fit(X_train_scaled, y_train)

# 预测并评估
val_predictions = model.predict(X_val_scaled)
print(f"SVM 本地验证集准确率 (Accuracy): {accuracy_score(y_val, val_predictions):.4f}")

# ==========================================
# 6. 生成最终预测
# ==========================================
# 用全部数据重新训练时，也需要对全量数据进行标准化
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

# 提取测试集特征，并使用之前的 scaler 进行转换
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

# 生成提交文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('svm_submission.csv', index=False)
print("预测完成，已生成 svm_submission.csv 文件！")