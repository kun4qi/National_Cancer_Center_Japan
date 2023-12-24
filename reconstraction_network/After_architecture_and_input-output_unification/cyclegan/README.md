# cyclegan

## 実行環境
cuda11.6, cudnn8.4, python3.10.10, gcc12.2.0(`module load gcc/12.2.0 python/3.10/3.10.10 cuda/11.6 cudnn/8.4`)


## データ作成
1. オリジナルのデータセットから```separate_slice.py```で画像をnpyで保存します。

2. 異常→正常で背景画像に過学習してしまって、ドメイン変換ができていなかったので、画像内のピクセル値の合計でフィルタリング ```save_filtered_image.py```

## 学習
```train_cyclegan_ssim.py``` 
identity lossをl2 lossではなく、ssim lossにすることでgan lossとの学習の方向性を緩和


そのときの学習条件は```config_train_cyclegan.json```に保存されています。



## 注意
train_cyclegan.pyでは、Aが正常画像、Bが異常画像としています。


[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
