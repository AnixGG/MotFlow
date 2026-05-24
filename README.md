# RAFT-GMC для multi-object tracking на спортивных видео

Проект посвящен исследованию multi-object tracking на спортивных видео SportsMOT. В основе пайплайна используются детектор YOLO11, трекер BoT-SORT и разные методы глобальной компенсации движения камеры (GMC).

Основная цель работы — проверить, может ли нейросетевой optical flow на базе RAFT использоваться как модуль GMC в BoT-SORT и дать полезный компромисс между качеством трекинга и задержкой обработки кадра на доступной GPU.

## Что сравнивается

В экспериментах сравниваются варианты BoT-SORT с разными методами компенсации движения камеры:

- `none` — без компенсации движения камеры;
- `sparseOptFlow` — стандартный sparse optical flow из Ultralytics BoT-SORT;
- `orb`, `sift`, `ecc` — стандартные методы GMC из Ultralytics;
- `raft` — реализованный в проекте RAFT-GMC.

Для оценки используются метрики multi-object tracking, включая MOTA, IDF1 и HOTA, а также измерение задержки обработки кадра.

## Структура проекта

```text
src/
  pipeline.py                  общий pipeline запуска экспериментов
  tracking/
    botsort_runner.py          запуск BoT-SORT по последовательностям SportsMOT
    ultralytics_runtime.py     подключение Ultralytics YOLO и patch GMC для BoT-SORT
  gmc/
    raft_gmc.py                реализация RAFT-GMC
  raft/
    raft_wrapper.py            wrapper над torchvision RAFT
  evaluation/
    metrics.py                 расчет MOT-метрик и HOTA
    visualization.py           вспомогательная визуализация результатов
  utils/                       конфиги, ввод-вывод, логирование, timing

configs/
  run_baseline.yaml            пример запуска baseline-эксперимента
  run_experiment.yaml          пример запуска RAFT-GMC
  botsort_*.yaml               конфиги BoT-SORT для разных GMC

scripts/
  run_baseline.py              запуск baseline pipeline
  run_experiment.py            запуск experimental pipeline

notebooks/
  run.ipynb                    ноутбук для запуска экспериментов
  train-yolo-sportsmot.ipynb   ноутбук дообучения YOLO11 на SportsMOT

results/
  pretrained_models/           результаты обучения YOLO11
  experiments/                 результаты tracking-экспериментов

docs/
  dataset.md                   описание SportsMOT и структуры данных
```

## Установка

Основной файл зависимостей:

```bash
pip install -r requirements.txt
```

Для окружения с CUDA 12.8 приведен пример установки:

```bash
bash install.sh
```

`install.sh` устанавливает PyTorch/torchvision под CUDA 12.8, затем зависимости из `requirements.txt`. В других окружениях, например в Colab или на другой версии CUDA, версию PyTorch может потребоваться выбрать отдельно.

## Данные

Эксперименты рассчитаны на структуру SportsMOT:

```text
dataset/
  train/
    <sequence_name>/
      img1/
      gt/gt.txt
  val/
    <sequence_name>/
      img1/
      gt/gt.txt
```

Путь к данным задается в YAML-конфигах через поле `data`. Для оценки используются ground truth файлы `gt/gt.txt`, а результаты трекинга сохраняются в MOT-формате.

Подробнее структура SportsMOT и формат разметки описаны в `docs/dataset.md`.

## Запуск

Перед запуском нужно проверить в конфиге пути:

- `data` — путь к split датасета, например `dataset/val`;
- `model` — путь к весам YOLO11;
- `outdir` — директория для сохранения результатов;
- `device` — устройство, например `cuda` или `cpu`.

Запуск baseline:

```bash
python3 scripts/run_baseline.py --config configs/run_baseline.yaml
```

Запуск RAFT-GMC эксперимента:

```bash
python3 scripts/run_experiment.py --config configs/run_experiment.yaml
```

Также для интерактивного запуска можно использовать ноутбук:

```text
notebooks/run.ipynb
```

## Конфигурация

Основные параметры запуска задаются в `configs/run_baseline.yaml` и `configs/run_experiment.yaml`.

Пример общих параметров:

```yaml
data: dataset/val
outdir: results/experiments/yolo11n/raft_small128
device: cuda
warmup_frames: 5

model: path/to/yolo11.pt
conf: 0.25
iou: 0.7
imgsz: 640
classes: [0]

tracker_config: configs/botsort_baseline.yaml
gmc: raft
```

Для стандартных методов GMC используется параметр `gmc_downscale`:

```yaml
gmc: sparseOptFlow
gmc_downscale: 2
```

Для RAFT-GMC используется секция `raft_gmc`:

```yaml
gmc: raft

raft_gmc:
  model: small
  device: cuda
  mixed_precision: false
  scale_gmc: 1
  image_size: 128
  sample_step: 8
  ransac_reproj_threshold: 3.0
```

Параметр `image_size` задает размер входа RAFT-GMC. Значение должно быть не меньше 128 и делиться на 8.

## RAFT-GMC

RAFT-GMC реализован в `src/gmc/raft_gmc.py` и подключается к BoT-SORT через стандартный интерфейс GMC. Для каждого нового кадра модуль:

1. Подготавливает текущий кадр и уменьшает его до заданного `image_size`.
2. На первом кадре сохраняет состояние и возвращает единичную матрицу.
3. Для последующих кадров оценивает optical flow между предыдущим и текущим кадром с помощью RAFT.
4. Отбрасывает некорректные значения flow, содержащие `NaN` или `Inf`.
5. Агрегирует flow в глобальное смещение камеры через медиану.
6. Возвращает матрицу сдвига формата `2 x 3`, совместимую с BoT-SORT:

```text
[1 0 dx]
[0 1 dy]
```

В RAFT-wrapper используется `torchvision.models.optical_flow`:

- `raft_small` или `raft_large`;
- предобученные веса torchvision;
- без дообучения RAFT;
- `num_flow_updates = 1`;
- `mask_predictor = None`;
- опциональный `mixed_precision` для CUDA.

## Оценка и выходные файлы

Для каждой последовательности pipeline сохраняет:

```text
<outdir>/
  config.yaml          сохраненная конфигурация запуска
  env_info.txt         информация об окружении
  run.log              лог выполнения
  metrics.csv          итоговые метрики качества
  timing.csv           задержка обработки кадров
  tracks/
    <sequence>.txt     треки в MOT-формате
```

Если в конфиге включена визуализация, дополнительно создается директория `vis/`.

`metrics.csv` содержит агрегированные метрики трекинга. `timing.csv` содержит задержку по последовательностям и строку `OVERALL`, включая среднее, стандартное отклонение, p50 и p95.

## Протокол измерения времени

Измерение задержки выполняется внутри tracking pipeline:

- batch size равен 1;
- измеряется обработка кадра через `model.track(...)`;
- сохранение видео и визуализация не входят в latency, если `visualization.enabled: false`;
- перед и после измеряемого участка вызывается `torch.cuda.synchronize()` для CUDA;
- перед основным прогоном последовательности выполняется warmup на `warmup_frames` кадрах;
- после warmup состояние BoT-SORT сбрасывается;
- результаты сохраняются отдельно по последовательностям и агрегируются в строке `OVERALL`.

## Обучение YOLO11

Дообучение детекторов YOLO11 на SportsMOT вынесено в ноутбук:

```text
notebooks/train-yolo-sportsmot.ipynb
```

Сохраненные результаты обучения лежат в:

```text
results/pretrained_models/
```

Для каждой версии модели доступны артефакты Ultralytics:

- `args.yaml` — параметры обучения;
- `results.csv` — значения метрик по эпохам;
- `results.png` — графики обучения;
- `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`;
- `confusion_matrix.png`;
- `val_batch*_pred.jpg` и `val_batch*_labels.jpg`.

## Результаты экспериментов

Готовые результаты tracking-экспериментов находятся в:

```text
results/experiments/
```

Они сгруппированы по модели детектора и методу GMC, например:

```text
results/experiments/yolo11n/raft_small128/
results/experiments/yolo11n/sparse_downscale4/
results/experiments/yolo11s/raft_small128/
results/experiments/yolo11m/sparse_downscale2/
```

Внутри каждой директории находятся `metrics.csv` и `timing.csv`, которые используются для сравнения качества и скорости.

## Воспроизводимость

Для воспроизводимого запуска важно фиксировать:

- версию `ultralytics`;
- версию PyTorch и CUDA;
- используемые веса YOLO11;
- список последовательностей SportsMOT;
- параметры `imgsz`, `conf`, `iou`;
- метод GMC и его параметры;
- количество warmup кадров;
- устройство запуска.

Эти данные частично сохраняются автоматически в `config.yaml` и `env_info.txt` внутри директории результата.
