# ADDA Neural Preconditioner

AI-прекондиционеры для ускорения итерационного решателя BiCGStab в программе [ADDA](https://github.com/adda-team/adda) (метод дискретных диполей, DDA).

## Две модели

### [K² v3](models/k2v3/) — Свёрточный прекондиционер (13.7M параметров)
Предсказывает свёрточное ядро из физических параметров (m, kd, grid) и формы частицы. K² удваивает эффективный радиус через `M_hat = K_hat @ K_hat` в частотной области. Один файл весов для всех задач.

### [Spectral](models/spectral/) — Спектральный прекондиционер (228K параметров)
Для каждой частотной точки предсказывает прекондиционер из матрицы задачи D_hat. Нет ограничения по дальности. Побеждает K² v3 на больших grid. Нужен переэкспорт для каждой задачи.

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

### Использование K² v3
```bash
# Экспорт
python apps/export_universal_precond.py \
    --checkpoint models/k2v3/checkpoints/best_model.pt \
    --squared_kernel --shape sphere --grid 33 \
    --m_re 3.0 --kd 0.4189 --output /tmp/precond.precond

# ADDA
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda -grid 33 -m 3.0 0.0 -shape sphere \
    -eps 5 -dpl 15 -precond /tmp/precond.precond
```

### Использование Spectral
```bash
# Экспорт (нужен для каждой задачи)
python apps/export_spectral_precond.py \
    --checkpoint models/spectral/checkpoints/best_model.pt \
    --shape sphere --grid 33 --m_re 3.0 --kd 0.4189 \
    --output /tmp/precond.precond

# ADDA (обязательно -grid, НЕ -size!)
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda -grid 33 -m 3.0 0.0 -shape sphere \
    -eps 5 -dpl 15 -precond /tmp/precond.precond
```

### Обучение с нуля
```bash
# K² v3
python train_v7/train.py --name my_k2v3 --device 0 --save \
    --loss adversarial --squared_kernel \
    --r_cut 7 --hidden_size 512 --num_layers 4 \
    --grid_min 8 --grid_max 64 --num_steps 60000 --lr 5e-4

# Spectral
python train_v7/train.py --name my_spectral --device 0 --save \
    --loss adversarial --spectral --squared_kernel \
    --freq_hidden 256 --freq_layers 5 \
    --grid_min 8 --grid_max 64 --num_steps 60000 --lr 1e-3
```

## Структура проекта

```
models/
  k2v3/                          — K² v3 модель + README
  spectral/                      — Spectral модель + README
neural_precond/
  model.py                       — все архитектуры
  loss.py                        — функции потерь
neuralif/
  fft_matvec.py                  — Python FFT матрично-векторное произведение
train_v7/
  train.py                       — скрипт обучения
apps/
  export_universal_precond.py    — экспорт K² v3 в .precond
  export_spectral_precond.py     — экспорт Spectral в .precond
  adda_matrix.py                 — генерация диполей, LDR поляризуемость
krylov/
  bicgstab.py                    — Python BiCGStab
adda/src/
  precond.c, precond.h           — C код прекондиционера в ADDA
docs/
  spectral_guide.md              — подробный гайд по Spectral (со словарём терминов)
  spectral_guide.pdf             — PDF версия гайда
```

## Документация

- [Spectral Guide (MD)](docs/spectral_guide.md) — подробный гайд с словарём терминов
- [Spectral Guide (PDF)](docs/spectral_guide.pdf) — PDF версия
- [K² v3 README](models/k2v3/README.md)
- [Spectral README](models/spectral/README.md)
