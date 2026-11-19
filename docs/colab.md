# Запуск в Google Colab

## Определить, какой YOLO11n использовался в старом прогоне

Выберите среду выполнения с GPU. Через панель «Файлы» загрузите два файла в `/content/`.

| Файл на ноутбуке | Путь в Colab |
|---|---|
| `archive_res/MOT_weights/yolo11n_sportsmot_best.pt` | `/content/yolo11n_sportsmot_best.pt` |
| `archive_res/traces/experiments805nano/experiments/yolo11n/none/tracks/v_00HRwkvvjtQ_c001.txt` | `/content/v_00HRwkvvjtQ_c001.txt` |

Запустите одну ячейку.

```bash
%%bash --no-raise-error
set -euo pipefail
cd /content
if [[ ! -d MotFlow/.git ]]; then
  git clone --branch workspace --single-branch https://github.com/AnixGG/MotFlow.git
fi
cd MotFlow
git pull --ff-only origin workspace
bash run.sh
```

Используйте URL в этой ячейке как обычный текст. При сбое Colab покажет вывод команды. Строки ошибки перед завершением ячейки укажут, какой файл или зависимость не удалось загрузить. Скрипт устанавливает совместимые версии NumPy и pandas в среде Colab с Python 3.13.

Скрипт скачивает `last.pt`, сверяет хеши обоих чекпойнтов и трека, воспроизводит `gmc=none` для обоих весов на одном клипе с теми же настройками и сравнивает новые боксы и confidence со старым треком. При первом запуске Colab скачает SportsMOT val с кадрами, **6,59 ГБ**, и установит PyTorch с CUDA 12.8 в отдельное окружение. Следующие вызовы в той же сессии используют кеш. [Манифест чекпойнтов](../configs/verified_checkpoints.yaml)

Готовый архив `.tar.gz` появится в `/content/motflow-results/`. Внутри `verdict.txt` содержит `best`, `last` или `inconclusive`. Значение `inconclusive` означает, что совпадение оказалось недостаточно сильным для идентификации. Пара боксов засчитывается при IoU от 0,5 и разнице confidence до 0,001. Для вывода нужны не менее 100 кадров старого трека, F1 таких совпадений от 0,90 и разница от 0,10 между вариантами. `fingerprints.csv` содержит числа для проверки вывода. Метрики TrackEval на том же клипе лежат в `best/none/trackeval/`, `last/none/trackeval/` и `archive/trackeval/`.

## Полный прогон

Команда `bash run.sh benchmark` сравнивает sparseOptFlow 8× и RAFT-large 128 на 45 клипах с проверенным `YOLO11m best`. Для повторного прогона n с `best` используйте последний пример ниже. Архив содержит прогоны, конфиги с SHA-256, треки, замеры времени, метрики TrackEval, парное сравнение, список версий пакетов и лог. Для записи в подключённый Google Drive задайте `MOT_RESULTS_DIR=/content/drive/MyDrive/motflow-results`.

Путь к `colab.log` выводится в начале запуска. Лог показывает номер метода и клипа, обработанные кадры и примерное время до конца текущего трекинга. У TrackEval отдельный счётчик и своя оценка времени. При необходимости откройте лог текущего запуска через панель «Файлы».

```bash
# Один клип для проверки окружения и прогонов
MOT_SEQUENCE=v_00HRwkvvjtQ_c001 bash run.sh benchmark

# Полное сравнение на YOLO11s
MOT_CHECKPOINT=yolo11s_best bash run.sh benchmark

# Полный прогон SIFT на проверенном YOLO11m
MOT_METHOD=sift2 bash run.sh benchmark

# Полное сравнение на YOLO11n best
MOT_CHECKPOINT=yolo11n_best MOT_WEIGHTS=/content/yolo11n_sportsmot_best.pt bash run.sh benchmark
```

## Пилот RAFT ONNX

На одной T4 сравниваются sparse 8×, RAFT-large 128 с PyTorch и RAFT-large 128 с ONNX Runtime CUDA. YOLO11s best и его встроенные ReID-признаки используются во всех трёх режимах. В ONNX экспортируется одно обновление потока в FP32. Перед трекингом скрипт проверяет разницу медианных сдвигов на реальных парах кадров и сохраняет профиль распределения операторов по устройствам.

```bash
git pull --ff-only origin workspace
MOT_SEQUENCE=v_00HRwkvvjtQ_c001 bash run.sh onnx
```

В архиве результатов смотрите `raft_export/parity.json` и `benchmark/summary.csv`. Сводка содержит HOTA, IDF1, среднюю задержку, p95 и долю кадров дольше 40 мс. После успешной проверки на одном клипе команда `bash run.sh onnx` повторяет сравнение на всех 45 клипах. Первый замер скорости считается пилотным.
