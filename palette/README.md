# palette
## メモ
timestepsを500, 750, 1000で学習をさせましたが、timestepsが大きいほど良い再構成が得られたわけではなく、それぞれのtimestepsで挙動が違ったため、データによって最適なtimestepsを確認する必要がありそうです。

## データ
https://github.com/nightrome/cocostuff



訓練：train2017.zip
テスト：val2017.zip
マスク：stuffthingmaps_trainval2017.zip

## 論文
[Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)
