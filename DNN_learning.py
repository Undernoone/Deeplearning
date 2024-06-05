import torch
ts1 = torch.randn(3,4) # 生成一个3x4的随机张量
print(ts1)
ts2 = ts1.to('cuda:0')
print(ts2)
class DNN(torch.nn.Module):
model = Dnn().to('cuda:0')