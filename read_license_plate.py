import os
import sys
import cv2
import numpy as np
from keras.models import load_model
from easyocr import Reader

path = sys.argv[1]

if path is None or path == '':
    raise ValueError('Usage: python prediction.py <path_to_image>')

try:
    image = cv2.imread(path)
except Exception:
    raise ValueError('Invalid path provided')

resized_image = cv2.resize(image, (224, 224))
scaled_image = resized_image / 255

# model to use
model = load_model('./models/vgg16/model_500.h5')

y = model.predict(np.array([scaled_image]))
xmin, ymin, xmax, ymax = y[0][0], y[0][1], y[0][2], y[0][3]
xmin = int(xmin * image.shape[1])
ymin = int(ymin * image.shape[0])
xmax = int(xmax * image.shape[1])
ymax = int(ymax * image.shape[0])

modifier = 0
a, b, c, d = int(ymin - image.shape[0] * modifier), int(ymax + image.shape[0] * modifier), int(xmin - image.shape[1] * modifier), int(xmax + image.shape[1] * modifier)
sliced_image = image[a:b, c:d]

reader = Reader(['en'])
result = reader.readtext(sliced_image)

os.makedirs('output', exist_ok=True)

cv2.imwrite('./output/sliced_result.png', sliced_image)
cv2.imwrite('./output/rectangle_result.png', cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2))


if len(result) == 0:
    with open('./output/result.txt', 'w') as f:
        f.write('No text found')
else:
    with open('./output/result.txt', 'w') as f:
        for i in result:
            f.write(f'{i[1]}\n')
