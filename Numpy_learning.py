import numpy as np
# arr1 = np.array([1, 2, 3])
# print(arr1) # 输出[1 2 3]，这是一个整型数组
# arr2 = np.array([1.0, 2, 3])
# print(arr2) # 输出[1. 2. 3.]，这是一个浮点型数组
# arr3 = [1, 2, 3]
# print(arr3) # 输出[1, 2, 3]，这是一个整型列表
# arr4 = [1.0, 2, 3]
# print(arr4) # 输出[1.0, 2, 3]，这是一个浮点型列表
# print(type(arr1))
# print(type(arr2))
# print(type(arr3))
# print(type(arr4))
# #numpy数组打印时元素之间用空格隔开，而列表打印时元素之间用逗号隔开

# arr1 = np.array([1, 2, 3])
# print(arr1)
# print(type(arr1))
# arr2 = arr1.astype(float) # 将整型数组转化为浮点型数组
# print(arr2)
# print(type(arr2))

# arr1 = np.ones(3) # 生成一个全1的数组
# print(arr1)
# arr2 = np.ones((1, 3)) # 生成一个1行3列的全1的矩阵
# print(arr2)
# arr3 = np.ones((1,1,3))
# print(arr3)
# print(arr1.shape)
# print(arr2.shape)
# print(arr3.shape)

# arr1 =np.arange(10)
# print(arr1)
# arr2 = arr1.reshape(-1,5) # 将数组reshape为5行，每行5列，-1表示由其他维度自动推断出剩余维度
# print(arr2)
# arr1 = np.zeros(3)
# print(arr1)
# arr2 = np.zeros((2, 3))
# print(arr2)
# arr3 = np.zeros((2,2,3))
# print(arr3)

# arr1 = np.random.rand(3,3)+60 # 生成一个3行3列的随机数组，范围在0到1之间，并加上60
# print(arr1)
# arr2 = np.random.randint(10,20,(2,3)) # 生成一个2行3列的随机整数数组，范围在10到20之间
# print(arr2)
# arr3 = np.random.normal(0,1,(9,10))
# print(arr3)
# arr1 = np.arange(1,10)
# print(arr1)
# print(arr1[0])
# print(arr1[2])
# print(arr1[-1])
# arr1[3]=100
# print(arr1)
# arr2 = np.array([[1,2,3],[4,5,6]])
# print(arr2)
# print(arr2[0,2])
# print(arr2[1,-2])
# arr1 = np.arange(1,100,10)
# print(arr1)
# print(arr1[[0,2]])
# arr1 = np.arange(0,100,4).reshape(5,5)
# print(arr1)
# print(arr1[[0,1],[3,3]])
#print(arr1[[0,1],[3,3]])打印的是第0行第3列和第1行第3列的值

# arr1 = np.arange(1,10) # 生成一个从1到9的数组
# print(arr1)
# print(arr1[1:3]) # 取数组的第2到第3个元素，正常情况下，左闭右开，即[1,3)
# print(arr1[1:])
# print(arr1[:2])
# print(arr1[2:-2])
# print(arr1[2:])
# print(arr1[:-2])
# print(arr1[::2])
# print(arr1[::3])
# print(arr1[1:-1:3])

# arr2 = np.arange(0,100,4).reshape(5,5) # 生成一个从0到99，步长为4的数组，并将其重塑为5行5列的矩阵
# print(arr2)
# print(arr2[1:3,1:-1]) # 取第2行到第3行，第2列到倒数第2列的元素。注意：矩阵的1：3是不包括第三行的
# print(arr2[::3,::2]) # 隔行隔列取元素

# arr2 = np.arange(0,100,4).reshape(5,5) # 生成一个从0到99，步长为4的数组，并将其重塑为5行5列的矩阵
# print(arr2)
# print(arr2[2,:]) # 取第3行的所有元素
# print(arr2[1:3,:]) # 取第2行到第3行的所有元素
# print(arr2[:,2]) # 取第3列的所有元素

# arr3 = np.arange(0,100,4).reshape(5,5) # 生成一个从0到99，步长为4的数组，并将其重塑为5行5列的矩阵
# print(arr3)
# cut = arr3[:3]  # cut是一个切片，取出了矩阵的前3行
# print(cut)
# cut[0,0] = 100  # 如果尝试修改切片的元素，但由于切片是引用(绑定）到原始矩阵，因此修改会影响到原始矩阵
# print(arr3)
# copy = arr3[:3].copy()  # copy不同于cut，它是一个副本，不会影响原始矩阵
# copy[0,0] = 100
# print(copy)
# print(arr3)

arr1 = np.arange(0,20,2).reshape(2,5)
print(arr1)
arr2 = arr1.T # 矩阵转置
print(arr2)
arr3 = np.fliplr(arr1) # 矩阵水平翻转
print(arr3)
arr4 = np.flipud(arr1) # 矩阵垂直翻转
print(arr4)