import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 加载数据
# ==========================================
print("正在加载数据...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# ==========================================
# 2. 数据预处理 (Data Preprocessing)
# ==========================================
def clean_data(df):
    """
    清洗数据并填充缺失值的函数，确保训练集和测试集处理方式一致
    """
    # 填充缺失值
    df['Age'] = df['Age'].fillna(df['Age'].median())     # 年龄缺失值用中位数填充
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())   # 票价缺失值用中位数填充
    df['Embarked'] = df['Embarked'].fillna('S')           # 登船港口缺失值用最常见的'S'填充

    # 将文本类别的特征转换为机器学习模型能看懂的数值
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df

train_data = clean_data(train_data)
test_data = clean_data(test_data)

# ==========================================
# 3. 特征选择 (Feature Selection)
# ==========================================
# 我们选择那些直觉上对生存率有较大影响的特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train_data[features]
y = train_data['Survived']

# ==========================================
# 4. 本地验证集划分 (验证模型表现)
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. 训练模型 (Model Training)
# ==========================================
# 初始化随机森林分类器
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# 在切分出的训练集上训练，并在验证集上查看准确率
model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
print(f"本地验证集准确率 (Accuracy): {accuracy_score(y_val, val_predictions):.4f}")

# ==========================================
# 6. 生成最终预测与提交文件
# ==========================================
# 为了达到最好的效果，我们用**全部**的训练数据重新训练一次模型
model.fit(X, y)

# 提取测试集的特征并进行预测
X_test = test_data[features]
predictions = model.predict(X_test)

# 按照 Kaggle 要求的格式创建提交数据框
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})

# 保存为 CSV 文件
submission.to_csv('submission.csv', index=False)
print("预测完成，已生成 submission.csv 文件，你可以将其提交到 Kaggle！")