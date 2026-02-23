# Violence Detection System (PyTorch)

Данный проект представляет собой систему классификации видео для обнаружения актов насилия в режиме реального времени. Модель построена на базе архитектуры **R(2+1)D**, которая анализирует видео как последовательность кадров, учитывая и изображение, и динамику движения.

## Результаты

* **Целевая точность:** 85%+
* **Итоговая точность на валидации:** ~99.6% (на Real Life Violence Situations Dataset)
* **Формат модели:** `.safetensors`

---

## Технологический стек

* **Framework:** PyTorch
* **Architecture:** R(2+1)D-18 (Fine-tuning)
* **Video Processing:** Decord, OpenCV
* **Data Cleaning:** Cleanvision (удаление дублей и битых файлов)
* **Model Storage:** Safetensors

---

## Ограничения и особенности (Limitations)

В ходе тестирования были выявлены специфические смещения (biases), связанные с особенностями исходного датасета:

* **Зависимость от освещения:** Модель может ошибочно классифицировать темные сцены как «Violence», а очень светлые — как «Non-Violence». Это связано с тем, что в обучающей выборке большинство сцен насилия зафиксировано камерами наблюдения в условиях плохой освещенности.
* **Рекомендация:** Для повышения надежности в реальных условиях рекомендуется дополнительная очистка данных по яркости или использование аугментаций (Random Brightness/Contrast) для выравнивания выборки.

---

## Структура проекта

* `Violance_detection.ipynb` — ноутбук с полным циклом: от загрузки и очистки данных до обучения.
* `model_acc_XX.safetensors` — веса модели.
* `requirements.txt` — зависимости проекта.

---

## Инструкция по запуску

### 1. Установка зависимостей

```bash
pip install torch torchvision decord safetensors opencv-python tqdm cleanvision numpy

```

### 2. Подготовка данных

Использовался датасет [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset).
В коде реализован препроцессинг:

* Удаление дубликатов через хеширование.
* Фильтрация видео короче 16 кадров.

### 3. Инференс

Функция `predict_violence` в ноутбуке позволяет запустить анализ на любом `.mp4` файле.

---

## Детали реализации

1. **Preprocessing:** Видео нарезается на 16 равномерных кадров, ресайзится до 224x224 и нормализуется.
2. **Fine-tuning:** Использована предобученная `r2plus1d_18`. Финальный слой переписан под 2 класса.
3. **Optimization:** AdamW (lr=1e-4), Mixed Precision (FP16) для ускорения обучения на GPU.

---

## Тестирование на реальных данных

Для проверки качества вне датасета использовались фрагменты из следующих видео:

1. [https://www.youtube.com/watch?v=tQp2EgxGmeo](https://www.youtube.com/watch?v=tQp2EgxGmeo)
2. [https://youtu.be/6949uuxkG6Y](https://www.google.com/search?q=https://youtu.be/6949uuxkG6Y)
3. [https://youtu.be/6AR0Uj-prHQ](https://www.google.com/search?q=https://youtu.be/6AR0Uj-prHQ)
4. [https://youtu.be/Pex0-KUbnxA](https://www.google.com/search?q=https://youtu.be/Pex0-KUbnxA)
5. [https://youtu.be/E1ulV2cGrwA](https://www.google.com/search?q=https://youtu.be/E1ulV2cGrwA)

Средняя точность на данных примерах подтверждает работоспособность модели, несмотря на выявленные особенности с освещением.
