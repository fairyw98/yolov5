# 给定搜索空间 partition_id, quantization_bit, coder_channels, en_stride
# 生成候选
# 将候选参数输入目标函数，得到评价结果
# 记录评价结果

import argparse
import csv
import os
import random
import yaml
import pandas as pd


# import utils.config as config
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval,rand,anneal

space = [
    hp.choice('partition_id',[0,1,3,5,7]), # 3
    hp.choice('quant_bits',[-1,1,2,4,8]), # 6
    hp.choice('coder_channels',[1,2,4,8,16,32,64]), # 7
    hp.choice('en_stide',[1,2,3,5,6,7,9]), # 7
]

def trick1(database_path,scheme):
    df1 = pd.read_csv(database_path).copy()
    df2 = df1.loc[(df1['partition_id'] == scheme[0]) &
                (df1['quant_bits'] == scheme[1] ) &
                (df1['coder_channels'] == scheme[2] ) &
                (df1['en_stride'] == scheme[3] )
                ,:]
    
    # print(df2)
    if df2.empty:
        return -1
    else:
        df2 = df2.reset_index(drop=True)
        # return float(df2['min_loss'][0])
        return (float(df2['map50'][0]),float(df2['loss'][0]))


# define an objective function
def objective(args):
    scheme = args
    config_yaml_path = "config.yaml"
    results = (0,0,0,0,100)
    content = read_yaml(config_yaml_path)
    content = change_yaml_train(content,results)
    write_yaml(config_yaml_path,content)

    config = content
    database_path = config['database_csv']

    trick_res = trick1(database_path,scheme)
    if trick_res!=-1:
        # print(trick_res)
        if os.path.exists(database_path) is True: 
            with open(database_path,'a+',newline='') as f:
                writer = csv.writer(f)
                row_list = [scheme[0],scheme[1],scheme[2],scheme[3],trick_res[0],trick_res[1]]
                writer.writerow(row_list)
        return trick_res[-1]
    # return 0

    os.system(f'cp ./models/yolov5s.yaml ./models/yolov5s-test.yaml')    
    yaml_path = "./models/yolov5s-test.yaml"
    content = read_yaml(yaml_path)
    content = change_yaml(content,scheme)
    write_yaml(yaml_path,content)

    os.system(f'python train.py \
                                --cfg {yaml_path} \
                                --epochs 500')

    config_yaml_path = "config.yaml"
    content = read_yaml(config_yaml_path)
    # content = change_yaml(content,scheme)
    # write_yaml(yaml_path,content)

    config = content

    if os.path.exists(database_path) is True: 
        with open(database_path,'a+',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scheme[0],scheme[1],scheme[2],scheme[3],config['results'][2],config['results'][4]])

    # return -config.best_acc
    return config['results'][4]

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


def change_yaml_train(content,res):
    # res = (1,2,3,4,5)
    tmp = content['results']
    # print(tmp)
    # print(res)
    tmp[0] = float(res[0])
    tmp[1] = float(res[1])
    tmp[2] = float(res[2])
    tmp[3] = float(res[3])
    tmp[4] = float(res[4])

    # content['results'] = tmp

    return content

def write_yaml(yaml_path,content):
    with open(yaml_path,'w') as file:
        yaml.dump(content,file)
    # return content
    return

def main(args):
    yaml_path = "config.yaml"
    content = read_yaml(yaml_path)
    # content = change_yaml(content,scheme)
    # write_yaml(yaml_path,content)

    config = content

    database_path = config['database_csv']

    alg_methods = {}
    alg_methods['random'] = rand.suggest
    alg_methods['anneal'] = anneal.suggest
    alg_methods['tpe'] = tpe.suggest

    alg = alg_methods[args.algo]
    max_evals = args.max_evals

    with open(database_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['partition_id','quant_bits','coder_channels','en_stride','map50','loss'])

    # minimize the objective over the space
    best = fmin(objective, 
                space, 
                algo=alg, 
                max_evals=max_evals)

    print(best)
    print(space_eval(space, best))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='random')
    parser.add_argument('--max_evals', type=int, default=500)
    opt = parser.parse_args()
    main(opt)
    