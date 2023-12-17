# 実行環境
cuda11.6, cudnn8.4, python3.10.10, gcc12.2.0(`module load gcc/12.2.0 python/3.10/3.10.10 cuda/11.6 cudnn/8.4`)


ライブラリはrequirements.txtでinstallできますが、pytorchだけできなかったので、
`pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`
こちらで別にインストールしました。

注:この前まではpython3.7でやっていて、abciの変更で3.7が使えなくなってしまい、vaeとintrovaeが動くかは確認しました。
ほかのコードもおそらく動くと思いますが、エラーがでたらすみません。

# 学習

separate_slice.pyでオリジナルのデータセットからnpyとして保存した後に、trainingを行ってます。


 VAEはddpでやってなくて、introVAEはddpでやってます。 
 
 
 VAEはt-sneやってないです。


[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

[IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis](https://arxiv.org/abs/1807.06358)
