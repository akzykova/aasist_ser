## Emotion-Aware AASIST: детекция синтезированной речи в условяих эмоционально окрашенного голоса

P.S. Название репозитория отражает концепцию попытки объединения модели детекции синтезированной речи (AASIST) и модели по распознаванию эмоций (Speech Emotion Recognition)

Данный репозиторий содержит фреймворк для обучения и оценки систем аудио-антиспуфинга с учётом эмоциональной окраски речи. Он объединяет созданный эмоциональный датасет (Datset of Synthesized Emotional Speech, DSES) и 6 моделей по детекции синтезированной речи:

* **AASIST**
* **AASIST\_Concat**
* **AASIST\_FiLM**
* **AASIST\_GFiLM**
* **AMSDF**
* **AASIST\_WAV2VEC**

Генерация эмоционального датасета реализована в папке `DSES` и представлена в виде трёх Jupyter-ноутбуков, каждый из которых соответствует одной из моделей TTS: Zonos, EmoSpeech и Cosyvoice. Ноутбуки содержат процесс синтеза речи с различными эмоциями на основе заданных фраз.

### Клонирование репозитория и установка зависимостей

   ```bash
   git clone https://github.com/akzykova/aasist_ser.git
   cd aasist_ser
   pip install -r requirements.txt
   ```

   Для работы с моделями **AMSDF** и **AASIST\_WAV2VEC** предлагается следующая инструкция:

   ```bash
   conda create -n SSL python=3.8 numpy=1.23.5
   conda activate SSL
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
   --------------install fairseq--------------
   git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
   cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
   pip install --editable ./
   --------------install requirement--------------
   git clone https://github.com/akzykova/aasist_ser.git
   cd aasist_ser
   pip install -r requirement.txt
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

2. **Dataset of Synthesized Emotional Speech**
 
   Вручную:

   * Скачать и распаковать `Dataset.zip` с сайта https://www.kaggle.com/datasets/annazykovamyzina/dataset-of-synthesized-emotional-speech
   * Указать путь к данным в конфигурации

   ToDo:
   Добавить на гугл-диск

### Обучение моделей (Train)

Обучение моделей производится на основе обучающей части ASVspoof 2019

* **AASIST**:

  ```bash
  python main.py --config config/AASIST.conf
  ```

* **Модели с добавлением эмоциональных эмбэддингов**:

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

Для оценки предобученных моделей предлагается воспользоваться весами. Для загрузки весов укажите необходимую модель:

   ```bash
   python download_weights.py --model "AMSDF"
   ``` 

Перед оценкой модели убедитесь, что загруженные веса находятся в `./models/weights`. При запуске используйте флаг `--eval`:

```bash
python main.py --eval --config config/AASIST_Concat.conf
```

Вывод будет содержать EER для eval-части ASVspoof2019 и для всего Datset of Synthesized Emotional Speech, для каждой из эмоций (нейтральная, грусть, радость, гнев, удивление) по отдельности.

### Запуск моделей (Inference)
Также после загрузки весов вы можете протестировать любую из 6 моделей на любых аудиозаписях в формате flac.

```bash
python inference.py --config config/AASIST_Concat.conf --test_dir {test_dir}
```
Вывод будет содержать txt - файл со значениями spoofing_scores для каждой из аудиозаписи.

