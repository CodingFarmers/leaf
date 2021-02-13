# 环境配置
```
conda create -n torch17 python==3.7
source activate torch17
pip install -r requirements.txt
```
验证环境配置
```
python test_enviroments.py
```

# 数据集配置
```
cd leaf/
ln -s {源数据集路径} cassava
```

# 训练
``
python train.py --model swsl_resnext101_32x8d --batch-size 8 --gpu 0,1 
``