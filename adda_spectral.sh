#!/bin/bash
# adda_spectral.sh — запуск ADDA с автоматическим экспортом Spectral прекондиционера
#
# Использование (вместо adda):
#   ./adda_spectral.sh -grid 33 -m 3.0 0.0 -shape sphere -dpl 15 -eps 5
#
# Скрипт:
#   1. Парсит аргументы ADDA (grid, m, shape, dpl)
#   2. Вычисляет kd = 2*pi/dpl
#   3. Экспортирует прекондиционер через Python
#   4. Запускает ADDA с -precond
#   5. Удаляет временный файл

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADDA_BIN="${SCRIPT_DIR}/adda/src/seq/adda"
EXPORT_SCRIPT="${SCRIPT_DIR}/apps/export_spectral_precond.py"
CHECKPOINT="${SCRIPT_DIR}/models/spectral/checkpoints/best_model.pt"

# Defaults
GRID=""
M_RE=""
M_IM="0.0"
SHAPE="sphere"
DPL="15"
AY="1.0"
AZ="1.0"

# Parse ADDA arguments to extract what we need for export
ADDA_ARGS=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        -grid)
            i=$((i+1)); GRID="${!i}"
            ADDA_ARGS+=("$arg" "$GRID")
            ;;
        -m)
            i=$((i+1)); M_RE="${!i}"
            i=$((i+1)); M_IM="${!i}"
            ADDA_ARGS+=("-m" "$M_RE" "$M_IM")
            ;;
        -shape)
            i=$((i+1)); SHAPE="${!i}"
            ADDA_ARGS+=("$arg" "$SHAPE")
            # Check for shape sub-arguments (ellipsoid has 2 extra args)
            if [ "$SHAPE" = "ellipsoid" ]; then
                i=$((i+1)); AY="${!i}"
                i=$((i+1)); AZ="${!i}"
                ADDA_ARGS+=("$AY" "$AZ")
            fi
            ;;
        -dpl)
            i=$((i+1)); DPL="${!i}"
            ADDA_ARGS+=("$arg" "$DPL")
            ;;
        -precond)
            echo "ERROR: -precond не нужен, adda_spectral.sh создаёт его автоматически" >&2
            exit 1
            ;;
        *)
            ADDA_ARGS+=("$arg")
            ;;
    esac
    i=$((i+1))
done

# Validate required args
if [ -z "$GRID" ] || [ -z "$M_RE" ]; then
    echo "Использование: $0 -grid N -m M_RE M_IM -shape SHAPE -dpl DPL [другие аргументы ADDA]"
    echo ""
    echo "Обязательные: -grid, -m"
    echo "По умолчанию: -shape sphere, -dpl 15"
    echo ""
    echo "Примеры:"
    echo "  $0 -grid 33 -m 3.0 0.0 -shape sphere -dpl 15 -eps 5"
    echo "  $0 -grid 24 -m 2.5 0.0 -shape ellipsoid 1.0 2.0 -dpl 15 -eps 5"
    echo "  $0 -grid 48 -m 3.5 0.0 -shape cube -dpl 15 -eps 5"
    exit 1
fi

# Compute kd = 2*pi/dpl
KD=$(python3 -c "import math; print(f'{2*math.pi/float(\"$DPL\"):.10f}')")

# Temp file for preconditioner
PRECOND_FILE=$(mktemp /tmp/spectral_precond_XXXXXX.precond)
trap "rm -f '$PRECOND_FILE'" EXIT

# Export
echo "=== Spectral Neural Preconditioner ==="
echo "  grid=$GRID  m=${M_RE}+${M_IM}i  shape=$SHAPE  dpl=$DPL  kd=$KD"

EXPORT_ARGS=(
    python3 "$EXPORT_SCRIPT"
    --checkpoint "$CHECKPOINT"
    --shape "$SHAPE"
    --grid "$GRID"
    --m_re "$M_RE" --m_im "$M_IM"
    --kd "$KD"
    --output "$PRECOND_FILE"
)

# Add ellipsoid aspect ratios
if [ "$SHAPE" = "ellipsoid" ]; then
    EXPORT_ARGS+=(--ay "$AY" --az "$AZ")
fi

echo "  Экспорт прекондиционера..."
"${EXPORT_ARGS[@]}" 2>&1 | grep -v "^$"

# Run ADDA
echo "  Запуск ADDA..."
echo "==="

export LD_LIBRARY_PATH="${HOME}/.local/lib:${LD_LIBRARY_PATH}"
"$ADDA_BIN" "${ADDA_ARGS[@]}" -iter bicgstab -precond "$PRECOND_FILE"
