# SportsMOT dataset

В проекте используется датасет [SportsMOT](https://arxiv.org/abs/2304.05170) — набор спортивных видеопоследовательностей для задачи multi-object tracking. Датасет собран для трекинга игроков на площадке/поле и содержит сцены из трех видов спорта: basketball, volleyball и football.

По описанию авторов SportsMOT содержит 240 видеопоследовательностей, более 150 тысяч кадров и более 1.6 миллиона bounding boxes. Официальный split состоит из 45 train, 45 validation и 150 test sequences. Разбиение сделано так, чтобы последовательности из одной игры не попадали в разные split.

Датасет не хранится в этом репозитории. Официальная статья ссылается на страницу SportsMOT проекта DeeperAction, а она отправляет к странице соревнования CodaLab и официальному GitHub-репозиторию SportsMOT. Также существует страница SportsMOT на Hugging Face, где указан полный размер датасета около 36.5 GB. В экспериментах автора для удобства использовалась приватная копия датасета на Kaggle, но это не является обязательной частью проекта: pipeline работает с любой локальной/смонтированной копией SportsMOT, если она приведена к ожидаемой структуре.

## Использование в этом проекте

В этом проекте SportsMOT используется в двух местах:

- train split — для дообучения YOLO11-детекторов;
- validation split — для оценки трекинга и сравнения GMC-методов.

Тестовый split SportsMOT не используется в основном экспериментальном pipeline, потому что для него обычно нет открытой ground truth разметки в том же виде, что и для train/val.

## Ожидаемая структура

Код ожидает структуру, близкую к MOTChallenge:

```text
dataset/
  train/
    <sequence_name>/
      img1/
        000001.jpg
        000002.jpg
        ...
      gt/
        gt.txt
      seqinfo.ini
  val/
    <sequence_name>/
      img1/
        000001.jpg
        000002.jpg
        ...
      gt/
        gt.txt
      seqinfo.ini
```

В YAML-конфигах поле `data` должно указывать на директорию split, внутри которой лежат папки sequences. Например:

```yaml
data: dataset/val
```

Тогда для sequence `v_00HRwkvvjtQ_c001` pipeline будет искать:

```text
dataset/val/v_00HRwkvvjtQ_c001/img1/
dataset/val/v_00HRwkvvjtQ_c001/gt/gt.txt
```

Если после распаковки датасет лежит глубже, например:

```text
dataset/sportsmot_publish_val/dataset/val/
```

то можно либо указать этот путь в конфиге:

```yaml
data: dataset/sportsmot_publish_val/dataset/val
```

либо перенести/создать ссылку на `dataset/val`.

## Формат ground truth

Файл `gt/gt.txt` использует MOTChallenge-подобный формат:

```text
frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
```

Пример строки:

```text
1, 0, 215, 289, 31, 117, 1, 1, 1
```

Поля:

- `frame` — номер кадра, начиная с 1;
- `id` — идентификатор объекта внутри sequence;
- `bb_left`, `bb_top` — координаты левого верхнего угла bounding box;
- `bb_width`, `bb_height` — ширина и высота bounding box;
- `conf` — флаг валидности разметки;
- `class` — класс объекта;
- `visibility` — видимость объекта.

В коде оценки используются только валидные player-разметки:

```text
conf == 1
class == 1
```

Это соответствует функции `load_gt_file(...)` в `src/evaluation/metrics.py`.

## Формат результатов трекинга

Pipeline сохраняет предсказанные треки в MOT-формате:

```text
frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
```

Для каждой sequence создается отдельный файл:

```text
<outdir>/tracks/<sequence_name>.txt
```

Эти файлы затем используются для расчета MOTA, IDF1 и HOTA.

## Использование для обучения YOLO11

Для обучения детектора разметка SportsMOT преобразуется в YOLO-формат. В рамках проекта используется один целевой класс — игрок. После преобразования каждая строка YOLO-разметки имеет вид:

```text
class_id x_center y_center width height
```

Координаты нормализуются относительно размера изображения. Важно контролировать, что:

- `x_center`, `y_center`, `width`, `height` лежат в допустимом диапазоне;
- `width` и `height` положительные;
- train и validation split не смешиваются.

Ноутбук для дообучения YOLO11:

```text
notebooks/train-yolo-sportsmot.ipynb
```

## Sequence metadata

Каждая последовательность содержит `seqinfo.ini`. В нем указаны параметры видео, например:

```ini
[Sequence]
name=v_2j7kLB-vEEk_c001
imDir=img1
frameRate=25
seqLength=826
imWidth=1280
imHeight=720
imExt=.jpg
```

Эта информация полезна для проверки частоты кадров, разрешения и длины sequence.

## Почему SportsMOT важен для этой работы

SportsMOT сложнее для трекинга, чем многие pedestrian MOT benchmarks, из-за особенностей спортивных сцен:

- быстрые и нерегулярные перемещения игроков;
- частые пересечения и частичные окклюзии;
- похожий внешний вид игроков одной команды;
- движение камеры, следящей за игрой.

Именно поэтому в проекте отдельно исследуется влияние GMC: компенсация движения камеры может менять качество ассоциации треков и задержку обработки кадра.

## Источники

- статья SportsMOT: https://arxiv.org/abs/2304.05170
- страница SportsMOT проекта DeeperAction: https://deeperaction.github.io/datasets/sportsmot.html
- официальный GitHub SportsMOT: https://github.com/MCG-NJU/SportsMOT
- CodaLab competition: https://codalab.lisn.upsaclay.fr/competitions/12424
- страница SportsMOT на Hugging Face: https://huggingface.co/datasets/MCG-NJU/SportsMOT
