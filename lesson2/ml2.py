from numpy import loadtxt, ones, zeros, array
from pylab import scatter, title,xlabel, ylabel, show, plot

def compute_cost(x, y, theta):
    m = y.size
    predictions = x.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J

def gradient_descent(x, y, theta, alpha, num_iters):
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):
        predictions = x.dot(theta).flatten()

        errors_x1 = (predictions - y) * x[:,0]
        errors_x2 = (predictions - y) * x[:,1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i,0] = compute_cost(x, y, theta)

    return theta, J_history

data = loadtxt('ex1data1.txt', delimiter=',')
#print data[:,0]

scatter(data[:,0], data[:,1], marker='o',c='b')
title('Profits distribution')
xlabel('Population of city in 10,000s')
ylabel('Profit in $10,000s')
#show()

x = data[:,0]
y = data[:,1]

m = y.size
it = ones(shape=(m,2))
it[:,1] = x

theta = zeros(shape=(2,1))

iterations = 1500
alpha = 0.01

print 'compute cost:', compute_cost(it, y, theta)

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print 'theta:', theta
print 'j_history:', J_history

predict1 = array([1,3.5]).dot(theta).flatten()
print 'For population = 35.000 we predict a profit of', predict1 * 10000
predict2 = array([1,7.0]).dot(theta).flatten()
print 'For population = 70.000 we predict a profit of', predict2 * 10000

result = it.dot(theta).flatten()
plot(data[:,0],result)
show()



