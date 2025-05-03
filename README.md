## Emotion-Aware AASIST: Антиспуфинг с учётом эмоций

Данный репозиторий содержит фреймворк для обучения и оценки систем аудио-антиспуфинга с учётом эмоциональной окраски речи. Исходная архитектура AASIST расширена за счёт интеграции эмоциональных эмбеддингов через несколько схем:

* **AASIST\_Concat** — конкатенация эмбеддингов с feature-картами;
* **AASIST\_FiLM** — Feature-wise Linear Modulation (FiLM) для масштабирования и смещения активаций;
* **AASIST\_GFiLM** — модифицированная схема FiLM с дополнительным gating-механизмом.

### Начало работы

1. Клонируйте репозиторий:

   ```bash
   git clone <URL-репозитория>
   cd <папка с проектом>
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
   Репликация аудио с различными эмоциями на основе ESD через Cosyvoice, Zonos и Emospeech.

  ToDo

### Конфигурации и обучение

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

### Оценка моделей

Для оценки предобученных моделей используйте флаг `--eval`:

```bash
python main.py --eval --config config/AASIST_Concat.conf
```

Вывод будет содержать EER и min t-DCF.

### Разработка собственных моделей

1. Определите класс модели `Model` в новом файле Python.
2. Создайте конфигурацию в `config/` (пример: `config/YourModel.conf`).
3. Запустите обучение:

   ```bash
   python main.py --config config/YourModel.conf
   ```
