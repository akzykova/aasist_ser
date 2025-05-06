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
   python download_dataset.py --dataset LA
   ```

   или вручную:

   * Скачать и распаковать `LA.zip` с сайта [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
   * Указать путь к данным в конфигурации.

2. **Dataset of Synthesized Emotional Speech**
 
   ```bash
   python download_dataset.py --dataset emotional
   ```

   или вручную:

   * Скачать и распаковать `Dataset.zip` с сайта https://www.kaggle.com/datasets/annazykovamyzina/dataset-of-synthesized-emotional-speech
   * Указать путь к данным в конфигурации
 
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
   python main.py --config config/AASIST_WAV2VEC.conf
   ``` 

### Оценка моделей (Evaluation)

Для оценки предобученных моделей необходимо сначала загрузить соответствующие веса. Используйте следующую команду, указав название необходимой модели:

   ```bash
   python download_weights.py --model "AASIST"
   ``` 

Убедитесь, что скачанные веса находятся в директории ./models/weights.

После этого запустите оценку модели с помощью флага --eval и нужного конфигурационного файла:

```bash
python main.py --eval --config config/AASIST.conf
```

Вывод будет содержать значения EER (Equal Error Rate) для eval-части ASVspoof2019 и для всего Datset of Synthesized Emotional Speech, для каждой из эмоций (нейтральная, грусть, радость, гнев, удивление) по отдельности.

### Запуск моделей (Inference)
После загрузки весов вы можете протестировать любую из 6 моделей на произвольных аудиофайлах в формате `.flac`

```bash
python inference.py --config config/AASIST_Concat.conf --test_dir {test_dir}
```
Результатом работы будет `.txt`-файл, содержащий spoofing scores для каждой аудиозаписи из указанной директории.

### Благодарность

Данный проект опирается на разработки и идеи из следующих открытых источников:

[AASIST[1]](https://github.com/clovaai/aasist)
[SSLAS[2]](https://github.com/TakHemlata/SSL_Anti-spoofing)
[SER[3]](https://github.com/Chien-Hung/Speech-Emotion-Recognition)
[Wav2vec[4]](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr)
[AMSDF[5]](https://github.com/ItzJuny/AMSDF)

```
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}}
```

```
@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```

```
@article{chen20183,
  title={3-D convolutional recurrent neural networks with attention model for speech emotion recognition},
  author={Chen, Mingyi and He, Xuanji and Yang, Jing and Zhang, Han},
  journal={IEEE Signal Processing Letters},
  volume={25},
  number={10},
  pages={1440--1444},
  year={2018},
  publisher={IEEE}
}
```
```
@article{babu2021xlsr,
      title={XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale}, 
      author={Arun Babu and Changhan Wang and Andros Tjandra and Kushal Lakhotia and Qiantong Xu and Naman Goyal and Kritika Singh and Patrick von Platen and Yatharth Saraf and Juan Pino and Alexei Baevski and Alexis Conneau and Michael Auli},
      year={2021},
      volume={abs/2111.09296},
      journal={arXiv},
}
```
```
@ARTICLE{wu2024audio,
  author={Wu, Junyan and Yin, Qilin and Sheng, Ziqi and Lu, Wei and Huang, Jiwu and Li, Bin},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Audio Multi-view Spoofing Detection Framework Based on Audio-Text-Emotion Correlations}, 
  year={2024},
  volume={19},
  number={},
  pages={7133-7146},
  doi={10.1109/TIFS.2024.3431888}}
```