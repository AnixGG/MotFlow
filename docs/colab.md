# Запуск в Google Colab

Выберите среду выполнения с GPU и выполните ячейку.

```bash
%%bash
git clone --branch workspace --single-branch https://github.com/AnixGG/MotFlow.git
cd MotFlow
bash run.sh
```

По умолчанию скрипт сравнивает sparseOptFlow 8× и RAFT-large 128 на всех 45 клипах SportsMOT val с подтверждённым `YOLO11m best`. Веса проверяются по SHA-256. При первом запуске Colab скачает val с кадрами, **6,59 ГБ**, и установит PyTorch с CUDA 12.8 и зависимости в отдельное окружение. Следующие вызовы в той же сессии используют локальный кеш.

Готовый архив `.tar.gz` появляется в `/content/motflow-results/`. Он содержит прогоны, конфиги с SHA-256, MOT-треки, замеры времени, официальные метрики TrackEval, парное сравнение, список версий пакетов и лог. Его можно скачать через файловую панель Colab. Для записи в уже подключённый Google Drive задайте `MOT_RESULTS_DIR=/content/drive/MyDrive/motflow-results` перед запуском.

## Варианты запуска

```bash
# Один клип для проверки окружения и прогонов
MOT_SEQUENCE=v_00HRwkvvjtQ_c001 bash run.sh

# Полное сравнение на YOLO11s
MOT_CHECKPOINT=yolo11s_best bash run.sh

# Полный прогон SIFT на проверенном YOLO11m
MOT_METHOD=sift2 bash run.sh

# Уже распакованный val и локальный YOLO11n best
MOT_VAL=/content/dataset/val MOT_CHECKPOINT=yolo11n_best \
  MOT_WEIGHTS=/content/weights/yolo11n_best.pt bash run.sh
```

Для `yolo11n_best` скрипт принимает файл с SHA-256 из манифеста. Опубликованная ссылка YOLO11n возвращает `last.pt`, который можно проверить отдельным запуском с `MOT_CHECKPOINT=yolo11n_last`. Для сверки со старым архивом запустите оба варианта на одном `MOT_SEQUENCE` с `MOT_METHOD=none` и передайте новые треки в скрипт сравнения. [Манифест чекпойнтов](../configs/verified_checkpoints.yaml) · [Сравнение с архивом](../scripts/compare_prediction_fingerprints.py)
