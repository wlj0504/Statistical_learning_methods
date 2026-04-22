import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# 1. 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
PassengerId = test_df['PassengerId']

# 为了方便进行特征工程，将训练集和测试集合并处理
all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# 2. 特征工程 (Feature Engineering)
# 2.1 提取称呼 (Title)
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# 将稀有称呼替换为 'Rare'，并归一化拼写
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

# 2.2 填充缺失的 Age (根据 Title 分组的中位数填充)
all_data['Age'] = all_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

# 2.3 构建家庭规模特征 (FamilySize)
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
# 根据人数划分为离散变量
all_data['IsAlone'] = 1
all_data.loc[all_data['FamilySize'] > 1, 'IsAlone'] = 0

# 2.4 填充 Embarked 和 Fare 的极少量缺失值
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)

# 2.5 票价 (Fare) 分箱处理 (减轻异常值影响)
all_data['FareBin'] = pd.qcut(all_data['Fare'], 4, labels=[0, 1, 2, 3])
all_data['FareBin'] = all_data['FareBin'].astype(int)

# 2.6 年龄 (Age) 分箱处理
all_data['AgeBin'] = pd.cut(all_data['Age'], 5, labels=[0, 1, 2, 3, 4])
all_data['AgeBin'] = all_data['AgeBin'].astype(int)

# 3. 数据编码与特征选择
# 类别特征编码
label_cols = ['Sex', 'Embarked', 'Title']
for col in label_cols:
    all_data[col] = LabelEncoder().fit_transform(all_data[col])

# 删除不需要的列 (降低维度)
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch']
all_data.drop(drop_cols, axis=1, inplace=True)

# 分离训练集和测试集
train_X = all_data[all_data['Survived'].notnull()].drop(['Survived'], axis=1)
train_y = all_data[all_data['Survived'].notnull()]['Survived']
test_X = all_data[all_data['Survived'].isnull()].drop(['Survived'], axis=1)

# 4. 构建与训练 XGBoost 模型
# 这里的超参数是经过初步调优的，防止过拟合
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 使用交叉验证评估一下本地得分
cv_scores = cross_val_score(xgb_model, train_X, train_y, cv=5, scoring='accuracy')
print(f"本地 5 折交叉验证平均准确率: {cv_scores.mean():.4f}")

# 5. 训练并生成预测结果
xgb_model.fit(train_X, train_y)
predictions = xgb_model.predict(test_X).astype(int)

# 保存提交文件
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
output.to_csv('xgboost_submission.csv', index=False)
print("预测完成，已生成 xgboost_submission.csv 文件！")