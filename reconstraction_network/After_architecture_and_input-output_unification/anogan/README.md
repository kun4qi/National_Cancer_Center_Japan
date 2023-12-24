# Anogan

## 実行環境
cuda11.6, cudnn8.4, python3.10.10, gcc12.2.0(`module load gcc/12.2.0 python/3.10/3.10.10 cuda/11.6 cudnn/8.4`)


## データ作成
1. オリジナルのデータセットから```separate_slice.py```で画像をnpyで保存します。


## 学習
```train_anogan.py``` 
潜在空間からのサンプリングは良くできていましたが、再構成がどうしてもできなくて原因がわからないまま終わっています...



そのときの学習条件は```configs/config.json```に保存されています。




[Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921)
