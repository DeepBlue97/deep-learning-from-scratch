import numpy as np
from gradient_2d_my import numerical_gradient
from gradient_2d_my import function_2


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        print('grad:    ', grad)
        x -= lr * grad
        print('   x:    ', x)
    return x


init_x = np.array([-3.0, 4.0])
x = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=1000)
print(x)

