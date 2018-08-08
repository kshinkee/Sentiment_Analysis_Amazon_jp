# 乱数シードの指定
import os
os.environ['PYTHONHASHSEED'] = '0'
import random
random.seed(0)
import numpy as np
np.random.seed(42)
import pandas as pd

import JapaneseTokeniser as jt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback


##############################################################################

"""
ハイパーパラメータ設定

max_features: ワードの種類数上限
             (出現頻度がmax_features位以下の単語は、一律の低頻度単語として扱う)
maxlen:       ワード長
             文章の長さ(ワード単位)がmaxlen以上の場合は、後方を切り捨て
embed_size:   単語分散表現のベクトルサイズ
rnn_size:     RNNセルのベクトルサイズ
batch_size:   バッチサイズ
epochs:       学習ループの回数
"""
max_features = 30000
maxlen = 32
embed_size = 50
rnn_size = 80
batch_size = 32
epochs = 1
##############################################################################

# Read CSV files here.
base_file_path = os.path.dirname(__file__)

file_ml = './data/input/test.csv'
df_ml = pd.read_csv(file_ml, encoding='shift-jisx0213')
# file_cd = './data/input/test_cd.csv'
# df_cd = pd.read_csv(file_cd, encoding='shift-jisx0213')
# file_rain = './data/input/test_rain.csv'
# df_rain = pd.read_csv(file_rain, encoding='shift-jisx0213')
# file_novel = './data/input/test_novel.csv'
# df_novel = pd.read_csv(file_novel, encoding='cp932')

# col_names_ml = ['c{0:02d}'.format(i) for i in range(14)]
# file_eng_ml = './data/input/train_eng_ml.csv'
# df_ml = pd.read_csv(file_eng_ml, sep=',', names=col_names_ml, encoding='cp932')
# file_eng_sample = './data/input/train_eng_sample.csv'
# df_sample = pd.read_csv(file_eng_sample, sep=',', names=col_names_ml, encoding='cp932')
# file_eng_wear = './data/input/train_eng_wear.csv'
# df_wear = pd.read_csv(file_eng_wear, sep=',', names=col_names_ml, encoding='cp932')
# file_eng_grave = './data/input/train_eng_grave.csv'
# df_grave = pd.read_csv(file_eng_grave, encoding='UTF_8')
# Combine data frames above
# df_total = pd.concat([df_ml, df_cd, df_rain, df_novel])
df_total = df_ml.copy(deep=True)


# Shuffle rows of the total data frame
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
df_total = df_total.sample(frac=1).reset_index(drop=True)

##############################################################################

# Extract columns to be used and map the stars {1, 2 / 3, 4, 5} to {0: bad / 1: good}
df_tokenised = df_total[['％星', 'レビュー内容']]
print(df_tokenised.head(20))
df_tokenised = df_tokenised.dropna(how='any', axis=0)
# df_tokenised = df_tokenised.drop(['レビュー内容'] is None, axis=0)
print("Modified data frame")
print(df_tokenised.head(20))
# 星の数に応じて評価を良い／悪いに分類。
df_tokenised.iloc[:, [0]] = np.where(df_tokenised.iloc[:, [0]] >= 3, 1, 0)
df_tokenised.iloc[:, [1]] = df_tokenised.iloc[:, [1]].fillna('fillna').values
train, test = train_test_split(df_tokenised, test_size=0.2, random_state=233)

##############################################################################

# X_train: テキスト部訓練データ、Y_train: 星の部分の訓練データ
X_train = train['レビュー内容']
y_train = train['％星']
X_test = test['レビュー内容']
y_test = test['％星']

# Tokenise Japanese sentences using Janome library.
jtokeniser = jt.JapaneseTokeniser()
X_train_list = jt.JapaneseTokeniser.analyse_japanese2(jtokeniser, X_train)
X_test_list = jt.JapaneseTokeniser.analyse_japanese2(jtokeniser, X_test)

tokenizer = text.Tokenizer(num_words=max_features)
list_train = []
list_test = []
for item in X_train_list:
   tokenizer.fit_on_texts(item)
   item = sequence.pad_sequences(tokenizer.texts_to_sequences(item), maxlen=maxlen)
   list_train.extend(item.tolist())
for item in X_test_list:
   tokenizer.fit_on_texts(item)
   item = sequence.pad_sequences(tokenizer.texts_to_sequences(item), maxlen=maxlen)
   list_test.extend(item.tolist())

# X_train_list = tokenizer.texts_to_sequences(X_train_list)
# X_test_list = tokenizer.texts_to_sequences(X_test_list)
#
# # TODO: ValueError: invalid literal for int() with base 10
# X_train_list = sequence.pad_sequences(X_train_list, maxlen=maxlen)
# X_test_list = sequence.pad_sequences(X_test_list, maxlen=maxlen)


##############################################################################
# トレーニング中にcallbackで回る
# 学習経過確認用のクラス
class RocAucEvaluation(Callback):
   def __init__(self, validation_data=(), interval=1):
       super(Callback, self).__init__()
       self.interval = interval
       self.X_val, self.y_val = validation_data

   def on_epoch_end(self, epoch, logs={}):
       if epoch % self.interval == 0:
           y_pred = self.model.predict(self.X_val, verbose=0)
           scores = [roc_auc_score(self.y_val[:, i], y_pred[:, i])
                     for i in range(1)]
           print("\n ROC-AUC - epoch: %d - score: %.6f \n" \
                 % (epoch+1, scores[0]))

##############################################################################

"""
モデル構造本体
"""
inp = Input(shape=(maxlen, ))

x = Embedding(max_features, embed_size)(inp)  # ワードを Embedding
x = SpatialDropout1D(0.2)(x)  # ドロップアウト. Spatial? が何かは忘れた

# Bidirectional RNN (Cell: GRU)
x = Bidirectional(GRU(rnn_size, return_sequences=True))(x)
# 各セルの全出力を結合（平均 & 最大）
# (このやり方は若干特殊で、出力最後尾を線形結合するのがスタンダードな気がします)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)

# 上記の 平均 & 最大 を結合
# (6ラベルが予測対象なので、最終出力のサイズは(6, ) )
conc = concatenate([avg_pool, max_pool])
outp = Dense(1, activation="sigmoid")(conc)

model = Model(inputs=inp, outputs=outp)
# lossは クロスエントロピー
# (一般的には、categorical_crossentropy だけど
#  今回は対象が 0 or 1 なので、binary_...)
# 最適化手法は Adam
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# モデル表示
# 各層における Output Shape の確認、見直しに使用
model.summary()
##############################################################################


RocAuc = RocAucEvaluation(validation_data=(list_test, y_test), interval=1)

print(len(list_train), len(list_test), len(y_train), len(y_test))

# モデルフィッティング
# epoch数指定で、早期打ち切りなし
hist = model.fit(np.ndarray(list_train), y_train, batch_size=batch_size,
                epochs=epochs, validation_data=(list_test, y_test),
                callbacks=[RocAuc], verbose=1)

# testデータで予測
# (kaggle なので正解データはないが、本来はここで正解データとの一致度(auc や正答率等)を確認)
y_pred = model.predict(list_test, batch_size=1024)
