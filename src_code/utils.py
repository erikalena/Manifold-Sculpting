import numpy as np



def generate_swiss_roll(n):
    """
    Function to generate swiss roll dataset
    input:
        n: number of samples

    output:
        data: swiss roll dataset

    """
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    for i in range(n):
        t = 8*i/n +2

        x[i]= t*np.sin(t)
        z[i]= t*np.cos(t)
        y[i]= np.random.uniform(-1, 1, 1)*6

    data = np.column_stack((x, y, z))

    return data