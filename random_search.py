# 给定搜索空间 partition_id, quantization_bit, coder_channels, en_stride
# 生成候选
# 将候选参数输入目标函数，得到评价结果
# 记录评价结果

import argparse
import csv
import os
import random
import yaml

# partition_id = [0,1,3,5,7] # 5
partition_id = [1] # 5
# quant = [255,15,3,1]
# quantization_bit = [1,2,4,8] # 4
quantization_bit = [-1]
# coder_channels = [0.03125,0.0625,0.09375,0.1250,0.15625, 0.1875, 0.21875, 0.2500, 0.5000, 1.0] # 10
coder_channels = [1,2,4,8,16,32,64] # 7
# en_stride = [0.03125,0.0625,0.09375,0.1250,0.15625, 0.1875, 0.21875, 0.2500, 0.5000, 1.0] # 10
en_stride = [1,2,3,5,6,7,9] # 7
# de_stride = [1] # 7

def generate_scheme(partition_id=partition_id,
                    quantization_bit=quantization_bit,
                    coder_channels=coder_channels,
                    en_stride = en_stride
                    ):

    p = random.choice(partition_id)
    q = random.choice(quantization_bit)
    c = random.choice(coder_channels)
    e = random.choice(en_stride)
    # d = random.choice(de_stride)

    scheme = (p,q,c,e,)
    return scheme

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        content = yaml.safe_load(file)
    return content

def change_yaml(content,scheme):
    tmp = content['backbone'][scheme[0]]
    tmp[2] = 'wfz_Conv_compression'
    param = []

    param.extend([tmp[3][-3]])
    param.extend(scheme[1:])
    param.extend(tmp[3][-2:])

    tmp[3] = param

    return content

def write_yaml(yaml_path,content):
    with open(yaml_path,'w') as file:
        yaml.dump(content,file)
    # return content
    return

def main(args):
    # yaml_path = "/home/wangfz/wksp/yolov5-wfz/models/yolov5s-wfz.yaml"
    with open('database.txt','w',newline='') as f:
        f.write('database\n')

    schemes = []
    scheme_num = args.scheme_num
    for i in range(scheme_num):
        scheme = generate_scheme()
        with open('database.txt','a+',newline='') as f:
            f.write(str(scheme))
            f.write('\n')
        print(scheme)
        # schemes.append(scheme)
        # 改写配置文件 yolov5s-wfz.yaml
        # 先复制一份 yolov5-wfz-copy.yaml
        os.system(f'cp /home/wangfz/yolov5/models/yolov5s.yaml  /home/wangfz/yolov5/models/yolov5s-test.yaml')    
        yaml_path = "/home/wangfz/yolov5/models/yolov5s-test.yaml"
        content = read_yaml(yaml_path)
        scheme = generate_scheme()
        content = change_yaml(content,scheme)
        write_yaml(yaml_path,content)

        os.system(f'python train.py')


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme_num', type=int, default=500)
    opt = parser.parse_args()
    main(opt)
    # os.system(f'cp /home/wangfz/wksp/yolov5-wfz/models/yolov5s.yaml  /home/wangfz/wksp/yolov5-wfz/models/yolov5s-test.yaml')    
    # yaml_path = "/home/wangfz/wksp/yolov5-wfz/models/yolov5s-test.yaml"
    # content = read_yaml(yaml_path)
    # scheme = generate_scheme()
    # content = change_yaml(content,scheme)
    # write_yaml(yaml_path,content)