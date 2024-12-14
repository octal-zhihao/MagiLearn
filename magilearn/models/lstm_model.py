import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel:
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
        self.model = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                             num_layers=self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X, y, n_epochs=100):
        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs, (hn, cn) = self.model(X)
            outputs = self.fc(outputs[:, -1, :])  # 获取最后一个时间步的输出
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs, (hn, cn) = self.model(X)
            outputs = self.fc(outputs[:, -1, :])
            _, predicted = torch.max(outputs, dim=1)
        return predicted.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
