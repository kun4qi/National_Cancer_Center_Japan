# 注意
vae_introvaeのリポジトリに実行環境について記載あり

## データ作成
1. オリジナルのデータセットから```separate_slice.py```で画像をnpyで保存します。(VAE、introVAEのレポジトリと同じコード)

1. そのあと、```separate_filldata.py```で画像内の白い領域の面積に応じてデータをフィルタリングしました。(全データでやるとサンプリングがうまくいかなかったので、行いました。また、その時の閾値は60000にしました。```python3 separate_filldata.py --th 60000```のように実行しました。そうすると、fill_data_60000フォルダが作成されます。

## 訓練
1. introVAEのときは画像のサイズを256でやっていたのですが、それだとAnoganはうまくいかなかったので、Resizeして画像のサイズは64で行いました。configで指定しています。

## やってないこと
ランダムサンプリングのみやってて、潜在空間の探索はやってないです。ランダムサンプリングのみやってて、潜在空間の探索はやってないです。 

[Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921)
