# Violence Detection System (PyTorch)

Данный проект представляет собой систему классификации видео для обнаружения актов насилия в режиме реального времени. Модель обучена на базе архитектуры **R(2+1)D**, которая эффективно анализирует как пространственные (кадры), так и временные (движение) признаки видео.

## Результаты

* **Целевая точность:** 85%+
* **Итоговая точность на валидации:** ~99% (на Real Life Violence Situations Dataset)
* **Формат модели:** `.safetensors` (безопасный формат весов)

---

## Технологический стек

* **Framework:** PyTorch
* **Architecture:** R(2+1)D-18 (Pre-trained)
* **Video Processing:** Decord, OpenCV
* **Data Cleaning:** Cleanvision
* **Model Storage:** Safetensors

---

## Структура проекта

* `Violance_detection.ipynb` — основной ноутбук с процессом загрузки данных, очистки, обучения и тестирования.
* `model_acc_XX.safetensors` — финальные веса обученной модели.
* `requirements.txt` — список необходимых библиотек для запуска.

---

## Инструкция по запуску

### 1. Среда окружения

Рекомендуется использовать Python 3.10+ и наличие GPU (CUDA) для инференса.

```bash
pip install torch torchvision decord safetensors opencv-python tqdm cleanvision

```

### 2. Подготовка данных

Для обучения использовался датасет [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset).
В ноутбуке реализована автоматическая очистка данных:

* Удаление дубликатов.
* Удаление поврежденных видео или файлов короче 16 кадров.

### 3. Инференс (Проверка своих видео)

В ноутбуке представлена функция `predict_violence`, которая позволяет прогнать любое видео через модель:

```python
result, confidence = predict_violence("path_to_your_video.mp4", model, transform)
print(f"Prediction: {result} ({confidence*100:.2f}%)")

```

---

## Детали реализации

1. **Preprocessing:** Видео нарезается на 16 равномерно распределенных кадров. Кадры нормализуются под стандарты ImageNet и ресайзятся до .
2. **Fine-tuning:** Использовалась предобученная модель `r2plus1d_18`. Финальный полносвязный слой был заменен для бинарной классификации (Violence / Non-Violence).
3. **Optimization:** Использовался оптимизатор `AdamW` с `learning_rate=1e-4` и автоматическое смешанное обучение (`Mixed Precision / GradScaler`) для ускорения процесса на GPU.

---

## Тестирование на реальных данных

Модель была протестирована на 5 сторонних видео из интернета, не входящих в обучающую выборку. Результаты подтвердили высокую обобщающую способность (точность > 90% на реальных примерах).

---

### Подсказка для тебя:

Не забудь создать файл `requirements.txt` в корне репозитория и добавить туда:

```text
torch
torchvision
decord
safetensors
opencv-python
tqdm
cleanvision
numpy

```
## Для тестов использовались кадры из этих видео
https://www.youtube.com/watch?v=tQp2EgxGmeo

https://youtu.be/6949uuxkG6Y?si=-34jBhlBDIQzcGkH

https://youtu.be/6AR0Uj-prHQ?si=a67KotX1OdNhqc2U

https://youtu.be/Pex0-KUbnxA?si=SL8-FgnJxnUP8MHX

https://youtu.be/E1ulV2cGrwA?si=f7P0kk4xO_Jez100
