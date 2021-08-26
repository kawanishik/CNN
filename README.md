# 深層学習の練習

1. CNNを使用して、人表情認識を行う
1. データセットは[kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)のものを使用

## ファイルの説明
1. training.py
   - python training.py
   - 上記のコマンドを使用することで、データを学習
1. test.py
   - python test.py
   - training.pyを使用して、学習したデータを使用して、表情を予測する
1. cnn_colab.ipynb
   - Google Colab上で動かすことができるプログラム
   - ローカル上でも動かすことができ、学習とテストを一つのファイルで完結させている


## 参考
1. [データセット](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
1. CNNを組むために参考にした[サイト](https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c)