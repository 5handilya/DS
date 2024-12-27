import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

########################################################
# LOAD & PREPROCESS

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
def preprocess(data):
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = data[features]
    return X

X_train = preprocess(train_data)
X_test = preprocess(test_data)
y_train = train_data['Survived']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_tensor = torch.FloatTensor(X_train_scaled)
Y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_scaled)

########################################################
# MODEL & TRAINING

class MLP(nn.Module): 
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential( #3+2 hidden layers: 64-64-32
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid() 
            )

    def forward(self, x):
        return self.layers(x)

model = MLP()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00008)
n_epochs = 25000
batch_size = 32

for epoch in range(n_epochs):
    model.train()
    #batched training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_Y = Y_train_tensor[i:i+batch_size]    

        #forward pass
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_Y)

        #backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #logging
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_predictions = (train_outputs.squeeze() > 0.5).float()
            train_accuracy = accuracy_score(y_train, train_predictions.numpy())
            print(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy:.4f}")

################################################################
# TESTING
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs.squeeze() > 0.5).float()

# output kaggle submission
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_preds.numpy().astype(int)
})
submission.to_csv('submission4.csv', index=False)