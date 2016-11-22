from numpy import genfromtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel, scatter, title, xticks, bar

data = genfromtxt('yar_realty.txt', delimiter=',')

districts = {1:"Zavolga", 2:"Bragino", 3:"Center", 4:"Pyaterka", 5:"Frunze", 6:"Perekop"}

def data_by_district(num):
    return array([row for row in data if row[8] == num])

def get_cost(district, property_size, room_number):
    dis_data = data_by_district(district)
    X = dis_data[:, 1:3]
    y = dis_data[:,5]
    m = y.size
    y.shape = (m, 1)
    x, mean_r, std_r = feature_normalize(X)
    it = ones(shape=(m, 3))
    it[:, 1:3] = x
    iterations = 400
    alpha = 0.01
    theta = zeros(shape=(3, 1))
    theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
    price = array([1.0,   ((room_number - mean_r[0]) / std_r[0]), ((property_size - mean_r[1]) / std_r[1])]).dot(theta)
    return int(price/10000)/100.0

def feature_normalize(X):
    mean_r = []
    std_r = []
    X_norm = X
    n_c = X.shape[1]

    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r

def compute_cost(X, y, theta):
    m = y.size
    predictions = X.dot(theta)
    sqErrors = (predictions - y)
    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):
        predictions = X.dot(theta)
        theta_size = theta.size

        for it in range(theta_size):
            temp = X[:, it]
            temp.shape = (m, 1)
            errors_x1 = (predictions - y) * temp
            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()
        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

prices = []
districts_names = []
districts_range = range(6)

for i in districts_range:
    prices.append(get_cost(i+1, 40, 2))
    districts_names.append(districts[i+1])

bar(districts_range, prices, align="center")
title('Predicted price of a 40 sq. meter, 2 room apartment')
xticks(districts_range, districts_names)
xlabel('District')
ylabel('Price')
show()




