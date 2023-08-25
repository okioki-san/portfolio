import tensorflow as tf
import numpy as np
from PIL import Image
from googletrans import Translator

# 画像の読み込み
image = Image.open("food.jpg")

# 画像の前処理
image = image.resize((299, 299))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# 学習済みモデルのロード
model = tf.keras.applications.InceptionV3()

# 画像に対する予測の実行
predictions = model.predict(image_array)

# 予測結果の出力
predicted_class = tf.keras.applications.inception_v3.decode_predictions(predictions)[0][0]

text_en = predicted_class[1]
confidence = int(predicted_class[2] * 100)

# 予測結果を日本語訳
text_ja = Translator().translate(text_en, dest='ja').text

print("予測結果: " + text_ja)
print("信頼度: " + str(confidence) + "%")