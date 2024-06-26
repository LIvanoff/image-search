## Установка

```shell
git clone https://github.com/LIvanoff/image-search
```

Перейти в папку с проектом и установить зависимости
```shell
cd image-search
pip install -r requirements.txt
```

## Запуск 

Создать объект класса ModelLauncher, указав одну из задач: `tagging`, `image_text_enc` или `text_enc`.
```python
from src.models.inference import ModelLauncher

# модель для тэгирования
model = ModelLauncher('tagging')

# модель для кодирования изображения
model = ModelLauncher('image_enc')

# модель для кодирования текста
model = ModelLauncher('text_enc')

# модель для распознавания текста
easyocr = ModelLauncher('ocr')  
```
#### Вызвать функцию тэггинга
```python
tags = model.tagging(file)
# tags = ['peson', 'car']
```
#### Вызвать функцию поиска похожих изображений
```python
model.find_images(file)
```
#### Вызвать функцию векторизации картинки
```python
vector = model.vectorize(file)
# vector = [0.3258 -0.19153 -0.031129 0.16856 -0.32208 ... -0.9297]
# vector.shape = (512,)
# type(vector) = <class 'numpy.ndarray'>
```
#### Вызвать функцию векторизации текста
```python
text = 'Москва, 1980 г.'
model.vectorize(text)
# vector = [0.3258 -0.19153 -0.031129 0.16856 -0.32208 ... -0.9297]
# vector.shape = (384,)
# type(vector) = <class 'numpy.ndarray'>
```
