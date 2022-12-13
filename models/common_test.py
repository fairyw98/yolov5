a=[1,2,3]
b=[4,5,6]
_conv2d=[]
for i in range(len(a)):
    for j in range(len(b)):
        _conv2d.append((i,j))

# print(_conv2d)
EnCoder = list([
    (_coder_channels,_en_stride)
        for _coder_channels,_en_stride in _conv2d])

print(EnCoder)
