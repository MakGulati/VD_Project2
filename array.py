import numpy as np
'''
test = np.random.randint(100,size=(6, 2))
print(test)

new = test.reshape(-1,3,2)

print(new)'''
a=np.array([[1,2,3],[1,2,4],[1,2,3]])
x=[tuple(row) for row in a]
uniques=np.unique(x)
print(uniques)