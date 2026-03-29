# ADDA Neural Preconditioner

AI-прекондиционеры для ускорения итерационного решателя BiCGStab в программе [ADDA](https://github.com/adda-team/adda) (метод дискретных диполей, DDA).

Проект основан на идеях из [neural-incomplete-factorization](https://github.com/paulhausner/neural-incomplete-factorization) (Paul Hausner) — GNN-прекондиционеры для систем линейных уравнений. Оттуда взяты базовые блоки (GraphNet, MLP), утилиты обучения и общий подход "нейросеть предсказывает приближённую обратную матрицу". Однако все финальные архитектуры (ConvSAI, K², Spectral) разработаны с нуля для специфики DDA.

Поиск гиперпараметров проводился с помощью [autoresearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — автоматизированный framework для запуска экспериментов. За 36 экспериментов (18×5мин + 6×30мин + 12×60мин) найдена K² архитектура (squared kernel).

## Две модели

### [K² v3](models/k2v3/) — Свёрточный прекондиционер
- **13.7M параметров**, стенсильное ядро с r_cut=7, K² удваивает до r_cut=14
- Один файл весов для всех задач (универсальный)
- До **54x** ускорения итераций в ADDA
- [Подробный гайд](docs/k2v3_guide.md)

### [Spectral](models/spectral/) — Спектральный прекондиционер
- **228K параметров** (в 60 раз меньше), поточечная MLP в частотной области
- Видит матрицу задачи D_hat → адаптируется к конкретной физике
- До **48x** ускорения, побеждает K² v3 в 29/31 тестах
- Нужен переэкспорт для каждой задачи
- [Подробный гайд](docs/spectral_guide.md) | [PDF](docs/spectral_guide.pdf)

## Результаты в ADDA

| Конфигурация | Без прекондиционера | Spectral (228K) | K² v3 (13.7M) |
|---|:---:|:---:|:---:|
| Сфера g33 m=3.0 | 12 676 итер | **290 (44x)** | 419 (30x) |
| Сфера g33 m=3.5 | 20 001 | **413 (48x)** | 626 (32x) |
| Сфера g48 m=3.0 | 20 000 | **448 (45x)** | FAIL |
| Куб g33 m=3.0 | 12 676 | **290 (44x)** | 418 (30x) |
| Куб g48 m=3.0 | 20 000 | **448 (45x)** | 1044 (19x) |

## Быстрый старт

### Требования
- Python 3.10+, PyTorch 2.0+, CUDA
- FFTW3 (для сборки ADDA)

### Установка
```bash
pip install torch numpy matplotlib
cd adda/src
make seq FFTW3_INC_PATH=$HOME/.local/include FFTW3_LIB_PATH=$HOME/.local/lib
```

### K² v3
```bash
# Экспорт (один раз, подходит для любого grid/m)
python apps/export_universal_precond.py \
    --checkpoint models/k2v3/checkpoints/best_model.pt \
    --squared_kernel --shape sphere --grid 33 \
    --m_re 3.0 --kd 0.4189 --output /tmp/precond.precond

# Запуск ADDA
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda -grid 33 -m 3.0 0.0 -shape sphere \
    -eps 5 -dpl 15 -precond /tmp/precond.precond
```

### Spectral
```bash
# Экспорт (нужен для каждой задачи)Поиск гиперпараметров проводился с помощью autoresearch (Andrej Karpathy) — автоматизированный framework для запуска экспериментов. За 36 экспериментов (18×5мин + 6×30мин + 12×60мин) найдена K² архитектура (squared kernel), давшая +5% к медианному ускорению.
python apps/export_spectral_precond.py \
    --checkpoint models/spectral/checkpoints/best_model.pt \
    --shape sphere --grid 33 --m_re 3.0 --kd 0.4189 \
    --output /tmp/precond.precond

# Запуск ADDA (обязательно -grid, НЕ -size!)
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda -grid 33 -m 3.0 0.0 -shape sphere \
    -eps 5 -dpl 15 -precond /tmp/precond.precond
```

### Обучение с нуля
```bash
# K² v3 (~8 часов на RTX 3090 Ti)
python train_v7/train.py --name my_k2v3 --device 0 --save \
    --loss adversarial --squared_kernel \
    --r_cut 7 --hidden_size 512 --num_layers 4 \
    --grid_min 8 --grid_max 64 --num_steps 60000 --lr 5e-4

# Spectral (~9 часов на RTX 3090 Ti)
python train_v7/train.py --name my_spectral --device 0 --save \
    --loss adversarial --spectral --squared_kernel \
    --freq_hidden 256 --freq_layers 5 \
    --grid_min 8 --grid_max 64 --num_steps 60000 --lr 1e-3
```

## Структура проекта

```
models/
  k2v3/                         — K² v3: чекпоинт + README
  spectral/                     — Spectral: чекпоинт + README
neural_precond/
  model.py                      — все архитектуры (ConvSAI_Universal, ConvSAI_Spectral, ...)
  loss.py                       — функции потерь (adversarial probe, probe, bicgstab)
core/
  fft_matvec.py                 — FFT матрично-векторное произведение (Python аналог ADDA MatVec)
  models.py                     — базовые блоки (GraphNet, MLP) из neural-incomplete-factorization
  utils.py                      — утилиты
  logger.py                     — логгер обучения
train_v7/
  train.py                      — скрипт обучения (K² v3 и Spectral)
apps/
  export_universal_precond.py   — экспорт K² v3 в .precond
  export_spectral_precond.py    — экспорт Spectral в .precond
  export_sai_precond.py         — экспорт SAI (базовый формат)
  adda_matrix.py                — генерация диполей, LDR поляризуемость
krylov/
  bicgstab.py                   — Python BiCGStab (для валидации при обучении)
adda_src_modified/
  precond.c, precond.h          — загрузка и применение прекондиционера в ADDA
  calculator.c                  — вызов DumpDhat() после InitDmatrix()
  param.c                       — CLI аргументы -precond, -dump_dhat
  vars.c, vars.h                — переменные precond_filename, dump_dhat_filename
  ADDAmain.c                    — инициализация прекондиционера
docs/
  spectral_guide.md             — подробный гайд по Spectral (со словарём)
  spectral_guide.pdf            — PDF версия
  k2v3_guide.md                 — подробный гайд по K² v3
```

## Благодарности

- [neural-incomplete-factorization](https://github.com/paulhausner/neural-incomplete-factorization) — базовые GNN блоки и общий подход к нейросетевым прекондиционерам
- [autoresearch](https://github.com/karpathy/autoresearch) — автоматизированный поиск гиперпараметров, через который найдена K² архитектура
- [ADDA](https://github.com/adda-team/adda) — программа дискретных диполей
