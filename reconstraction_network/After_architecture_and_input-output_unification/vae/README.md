# VAE

## 実行環境
cuda11.6, cudnn8.4, python3.10.10, gcc12.2.0(`module load gcc/12.2.0 python/3.10/3.10.10 cuda/11.6 cudnn/8.4`)


## データ作成
オリジナルのデータセットから```separate_slice.py```で画像をnpyで保存します。


## 学習
```train_vae_lightning.py``` 


そのときの学習条件は```config_train_vae.json```に保存されています。




[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
