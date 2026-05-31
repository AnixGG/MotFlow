# Веса моделей

Веса обученных YOLO11-моделей не хранятся в git из-за размера файлов. Для воспроизведения экспериментов их нужно скачать отдельно и указать путь к нужному `.pt` файлу в YAML-конфиге.

## Ссылки на Google Drive

- YOLO11n SportsMOT: `https://drive.google.com/file/d/1RbC3D9emgMee1VnjczDspzn8spKwXOp9/view?usp=sharing`
- YOLO11s SportsMOT: `https://drive.google.com/file/d/1Kjo-ei1KV7nVa_jmnVzB8YWVu9tp0BqW/view?usp=sharing`
- YOLO11m SportsMOT: `https://drive.google.com/file/d/1mHuZBesRudoCe8dBUs9fcs2tKNbAhJBc/view?usp=sharing`


## Куда положить файлы

Рекомендуемая локальная структура:

```text
models/
  yolo11n_sportsmot.pt
  yolo11s_sportsmot.pt
  yolo11m_sportsmot.pt
```

Папка `models/` игнорируется git, поэтому веса останутся только локально.

## Как указать веса в конфиге

Для локального запуска можно указать относительный путь:

```yaml
model: models/yolo11n_sportsmot.pt
```

Для Google Colab можно указать путь к файлу на Google Drive:

```yaml
model: /content/drive/MyDrive/MOT_weights/yolo11n_sportsmot.pt
```

В `notebooks/run.ipynb` путь к весам задается переменной:

```python
YOLO_MODEL_PATH = Path("/content/drive/MyDrive/MOT_weights/yolo11n_sportsmot.pt")
```

При запуске ноутбук подставляет этот путь в runtime-копии `configs/run_baseline.yaml` и `configs/run_experiment.yaml`.

## Связанные результаты обучения

Даже без `.pt` файлов в репозитории остаются артефакты обучения:

```text
results/pretrained_models/
```

Там находятся `args.yaml`, `results.csv`, `results.png`, PR/F1-кривые, confusion matrix и примеры validation batches для обученных моделей.
