# VQGAN

## 実行環境
cuda11.6, cudnn8.4, python3.10.10, gcc12.2.0(`module load gcc/12.2.0 python/3.10/3.10.10 cuda/11.6 cudnn/8.4`)


## データ作成
オリジナルのデータセットから```separate_slice.py```で画像をnpyで保存します。

## 学習
```train_vqgan_lightning.py``` 




そのときの学習条件は```config_train_vqgan.json```に保存されていますが、画像の範囲とモデルの出力を0-1に統一したところ再構成が全くできなくなってしまい、
recon lossのみで学習させれば再構成はできましたが、潜在空間に関する情報は学習させていない状況です。そのため、recon lossのみの状態から潜在空間に関するlossを足して調整しないといけないです。



[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
