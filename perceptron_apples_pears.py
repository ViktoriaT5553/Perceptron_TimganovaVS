from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def Loss(y_pred, y):
    '''
    Считаем среднеквадратичную ошибку
    '''
    y_pred = y_pred.reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    return 0.5 * np.mean((y_pred - y) ** 2)

class Perceptron:
    def __init__(self, w=None, b=0, penalty='l2', alpha=0.001, learning_rate=0.05):
        """
        :param: w -- вектор весов
        :param: b -- смещение
        :param: penalty -- вид регуляризации ('l1', 'l2' или None)
        :param: alpha -- коэффициент регуляризации
        :param: learning_rate -- скорость обучения
        """
        # Пока что мы не знаем размер матрицы X, а значит не знаем, сколько будет весов
        self.w = w
        self.b = b
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        
    def activate(self, x):
        return np.array(x > 0, dtype=np.int64)
        
    def forward_pass(self, X):
        """
        Эта функция рассчитывает ответ перцептрона при предъявлении набора объектов
        :param: X -- матрица объектов размера (n, m), каждая строка - отдельный объект
        :return: вектор размера (n, 1) из нулей и единиц с ответами перцептрона 
        """
        n = X.shape[0]
        y_pred = np.zeros((n, 1))  # y_pred == y_predicted - предсказанные классы
        y_pred = self.activate(X @ self.w + self.b)
        return y_pred.reshape(-1, 1)
    
    def backward_pass(self, X, y, y_pred):
        """
        Обновляет значения весов перцептрона в соответствии с этим объектом
        :param: X -- матрица объектов размера (n, m)
                y -- вектор правильных ответов размера (n, 1)
        В этом методе ничего возвращать не нужно, только правильно поменять веса
        с помощью градиентного спуска.
        """
        n = len(y)
        y = np.array(y).reshape(-1, 1)
        if self.penalty == 'l2':
            grad_w = (X.T @ (y_pred - y) / n) + self.alpha * self.w
        elif self.penalty == 'l1':
            grad_w = (X.T @ (y_pred - y) / n) + self.alpha * np.sign(self.w)
        else:
            grad_w = X.T @ (y_pred - y) / n
        
        # Убедитесь, что формы совпадают
        assert self.w.shape == grad_w.shape, f"Формы не совпадают: self.w.shape={self.w.shape}, grad_w.shape={grad_w.shape}"
        
        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * np.mean(y_pred - y)

    def fit(self, X, y, num_epochs=500):
        """
        Спускаемся в минимум
        :param: X -- матрица объектов размера (n, m)
                y -- вектор правильных ответов размера (n, 1)
                num_epochs -- количество итераций обучения
        :return: losses -- вектор значений функции потерь
        """
        self.w = np.zeros((X.shape[1], 1))  # столбец (m, 1)
        self.b = 0  # смещение
        losses = []  # значения функции потерь на различных итерациях обновления весов
        
        for i in range(num_epochs):
            # предсказания с текущими весами
            y_pred = self.forward_pass(X)
            # считаем функцию потерь с текущими весами
            losses.append(Loss(y_pred, y))
            # обновляем веса в соответсвие с тем, где ошиблись раньше
            self.backward_pass(X, y, y_pred)

        return losses

# Чтение данных
data = pd.read_csv("C:\\Users\\TEMP.LAPTOP-EM0D1PRH\\Desktop\\Magistracy\\Artificial Intelligence\\apples_pears.csv")

# Извлечение признаков и целевого вектора
X = data.iloc[:,:2].values  # матрица объекты-признаки
y = data['target'].values.reshape((-1, 1))  # классы (столбец из нулей и единиц)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Создание модели с регуляризацией
perceptron = Perceptron(penalty='l2', alpha=0.01, learning_rate=0.03)

# Обучение модели
losses = perceptron.fit(X_scaled, y, num_epochs=700)

# Визуализация функции потерь
plt.figure(figsize=(10, 8))
plt.plot(losses)
plt.title('Функция потерь', fontsize=15)
plt.xlabel('номер итерации', fontsize=14)
plt.ylabel('$Loss(\hat{y}, y)$', fontsize=14)
plt.show()

# Прогнозирование
predictions = perceptron.forward_pass(X_scaled)

# Визуализация результатов
plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=predictions.ravel(), cmap='spring')
plt.title('Классификация яблок и груш', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()

