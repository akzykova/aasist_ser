## Emotion-Aware AASIST: детекция синтезированной речи в условяих эмоционально окрашенного голоса

P.S. Название репозитория отражает концепцию попытки объединения модели детекции синтезированной речи (AASIST) и модели по распознаванию эмоций (Speech Emotion Recognition)

Данный репозиторий содержит фреймворк для обучения и оценки систем аудио-антиспуфинга с учётом эмоциональной окраски речи. Он объединяет 6 моделей и созданный эмоциональный датасет:

* **AASIST**
* **AASIST\_Concat** — конкатенация эмбеддингов с feature-картами;
* **AASIST\_FiLM** — Feature-wise Linear Modulation (FiLM) для масштабирования и смещения активаций;
* **AASIST\_GFiLM** — модифицированная схема FiLM с дополнительным gating-механизмом,
* **AMSDF**
* **AASIST\_WAV2VEC**

### Начало работы

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/akzykova/aasist_ser.git
   cd aasist_ser
   ```
2. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

   Рекомендуемое окружение для GPU-тренировки:

   * GPU: NVIDIA Tesla V100 (не менее 16 ГБ памяти)

### Подготовка данных

1. **ASVspoof 2019 (Logical Access)**

   ```bash
   python download_dataset.py
   ```

   или вручную:

   * Скачать и распаковать `LA.zip` с сайта [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
   * Указать путь к данным в конфигурации.

2. **Эмоциональный TTS-датасет**
   Вручную:

   * Скачать и распаковать `Dataset.zip` с сайта [Kaggle] (https://www.kaggle.com/datasets/annazykovamyzina/dataset-of-synthesized-emotional-speech)
   * Указать путь к данным в конфигурации

   ToDo:
   Добавить на гугл-диск

### Обучение моделей (Train)

Обучение моделей производится на основе обучающей части ASVspoof 2019

* **Оригинальный AASIST**:

  ```bash
  python main.py --config config/AASIST.conf
  ```

* **Расширенные модели с эмоциями**:

  ```bash
  python main.py --config config/AASIST_Concat.conf
  python main.py --config config/AASIST_FiLM.conf
  python main.py --config config/AASIST_GFiLM.conf
  ```

* **AMSDF**

   ```bash
   python main.py --config config/AMSDF.conf
   ``` 

* **WAV2VEC**

   ```bash
   python main.py --config config/WAV2VEC.conf
   ``` 

### Оценка моделей (Evaluation)

Для оценки предобученных моделей предлагается воспользоваться весами. Для загрузки весов указывайте модель:

   ```bash
   python download_weights.py --"AMSDF"
   ``` 

Для оценки предобученных моделей используйте флаг `--eval`:

```bash
python main.py --eval --config config/AASIST_Concat.conf
```

Вывод будет содержать EER для eval-части ASVspoof2019 и для всего Datset of Emotional Synthesized Speech, для каждой из эмоций (нейтральная, грусть, радость, гнев, удивление).

### Запуск моделей (Inference)
Также вы можете протестировать любую из 6 моделей на любых аудиозаписях в формате flac.

```bash
python inference.py --config config/AASIST_Concat.conf --test_dir {test_dir}
```
Вывод будет содержать txt - файл со значениями spoofing_scores для каждой из аудиозаписи.

