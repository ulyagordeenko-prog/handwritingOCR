# Распознавание рукописного текста

Настольное приложение: фото страницы → текст. Каждая строка показывается
картинкой, под ней — распознанный текст в редактируемом поле, покрашенный
по уверенности модели. Работает офлайн, всё считается на вашем компьютере.

## Установка

Нужен Python 3.11+. Видеокарта NVIDIA желательна, но не обязательна —
без неё работает медленнее.

```bash
git clone <адрес репозитория>
cd ocr_project
```

Установить менеджер пакетов uv, если его ещё нет:

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Поставить зависимости:

```bash
uv sync
```

## Языковая модель (необязательно, но заметно улучшает результат)

Приложение работает и без неё, но с ней выбирает более осмысленный вариант
из нескольких прочтений строки. Скачивание — 529 МБ, один раз:

```bash
uv run python -c "
import torch, os, shutil
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

d = snapshot_download('ai-forever/rugpt3small_based_on_gpt2',
                      allow_patterns=['*.json','*.txt','pytorch_model.bin'])
sd = torch.load(os.path.join(d, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
sd = {k: v.contiguous() for k, v in sd.items()}
sd.pop('lm_head.weight', None)
os.makedirs('models/rugpt3small', exist_ok=True)
save_file(sd, 'models/rugpt3small/model.safetensors')
for f in os.listdir(d):
    if f != 'pytorch_model.bin':
        shutil.copy(os.path.join(d, f), 'models/rugpt3small/')
"
```

Модель распознавания (TrOCR-ru, 1.3 ГБ) скачается сама при первом запуске.

## Запуск

```bash
uv run python run_app.py
```

Нажмите «Открыть фото» и выберите изображение. Для пробы в репозитории лежит
`test_page.jpg`. Первый запуск дольше — скачивается базовая модель.

## Как читать результат

Цвет фона под текстом означает уверенность модели:

- белый — уверена, скорее всего верно
- жёлтый — сомневается, стоит взглянуть
- красный — не уверена, почти наверняка нужна правка

Текст редактируется прямо в окне; «Сохранить текст» выгружает `.txt`
уже с вашими правками.

## Что внутри

| Файл | Роль |
|---|---|
| `src/ocr_project/segmentation.py` | режет фото на строки, выравнивает наклон, делит развороты |
| `src/ocr_project/recognizer.py` | читает строки: TrOCR-ru + дообученный адаптер |
| `src/ocr_project/rescorer.py` | выбирает вариант, больше похожий на русский язык |
| `src/ocr_project/app.py` | интерфейс |
| `checkpoints/trocr-lora-v5/` | наш дообученный адаптер (12 МБ) |

## Точность

Ошибка на символ (CER) на 60 строках, не участвовавших в обучении:

| | CER |
|---|---|
| TrOCR-ru без дообучения | 21.9% |
| с адаптером из этого репозитория | **6.9%** |

## Ограничения

- Текст, просвечивающий с обратной стороны листа, распознаётся как отдельные
  строки. Помечается низкой уверенностью, но не отсеивается.
- Слитный курсив на выцветших бланках (подписи в штампах) читается плохо.
- Одна страница обрабатывается около минуты на слабой видеокарте.
