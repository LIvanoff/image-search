## Установка

```shell
git clone https://github.com/LIvanoff/image-search
```

Перейти в папку с проектом и установить зависимости
```shell
cd image-search
pip3 install requirements.txt
```

## Запуск 

Создать объект класса ModelLauncher, указав одну из задач: 'tagging', 'image_text_enc' или 'text_enc'.
```python
model = ModelLauncher('tagging')
```
#### Вызвать функцию тэггинга
```python
model.tagging(file)
```
#### Вызвать функцию поиска похожих изображений
```python
model.find_images(file)
```
#### Вызвать функцию векторизации картинки
```python
model.vectorize(file)
```
#### Вызвать функцию векторизации текста
```python
text = 'Москва, 1980 г.'
model.vectorize(text)
```
