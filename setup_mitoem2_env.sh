#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_mitoem2_env.sh [SCRATCH_DIR]
#
# Default SCRATCH_DIR is /tempory/luiznery. The script keeps conda envs,
# package caches, pip cache, and temporary files under SCRATCH_DIR.

SCRATCH_DIR="${1:-/tempory/luiznery}"
ENV_NAME="mitoem2"
ENV_PREFIX="$SCRATCH_DIR/conda_envs/$ENV_NAME"
MINIFORGE_DIR="$SCRATCH_DIR/miniforge3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASHRC="$HOME/.bashrc"

mkdir -p \
    "$SCRATCH_DIR/conda_pkgs" \
    "$SCRATCH_DIR/conda_envs" \
    "$SCRATCH_DIR/pip_cache" \
    "$SCRATCH_DIR/tmp"

export CONDA_PKGS_DIRS="$SCRATCH_DIR/conda_pkgs"
export CONDA_ENVS_PATH="$SCRATCH_DIR/conda_envs"
export PIP_CACHE_DIR="$SCRATCH_DIR/pip_cache"
export TMPDIR="$SCRATCH_DIR/tmp"

install_miniforge() {
    if [ -x "$MINIFORGE_DIR/bin/conda" ]; then
        "$MINIFORGE_DIR/bin/conda" install -p "$MINIFORGE_DIR" -c conda-forge gh -y
        return
    fi

    local installer="$SCRATCH_DIR/Miniforge3-Linux-x86_64.sh"
    local url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

    if command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$installer"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$installer" "$url"
    else
        echo "Need curl or wget to download Miniforge." >&2
        exit 1
    fi

    bash "$installer" -b -p "$MINIFORGE_DIR"
    "$MINIFORGE_DIR/bin/conda" install -p "$MINIFORGE_DIR" -c conda-forge gh -y
}

ensure_bashrc_block() {
    local marker_start="# >>> mitoem2 scratch setup >>>"
    local marker_end="# <<< mitoem2 scratch setup <<<"
    local tmp
    tmp="$(mktemp)"

    if [ -f "$BASHRC" ]; then
        awk -v start="$marker_start" -v end="$marker_end" '
            $0 == start {skip=1; next}
            $0 == end {skip=0; next}
            !skip {print}
        ' "$BASHRC" > "$tmp"
    else
        : > "$tmp"
    fi

    cat >> "$tmp" <<EOF

$marker_start
export SCRATCH_DIR="$SCRATCH_DIR"
export CONDA_PKGS_DIRS="\$SCRATCH_DIR/conda_pkgs"
export CONDA_ENVS_PATH="\$SCRATCH_DIR/conda_envs"
export PIP_CACHE_DIR="\$SCRATCH_DIR/pip_cache"
export TMPDIR="\$SCRATCH_DIR/tmp"

if [ -f "\$SCRATCH_DIR/miniforge3/etc/profile.d/conda.sh" ]; then
    . "\$SCRATCH_DIR/miniforge3/etc/profile.d/conda.sh"
fi

__mitoem2_fix_python() {
    if [ "\${CONDA_PREFIX:-}" = "\$SCRATCH_DIR/conda_envs/mitoem2" ]; then
        export PATH="\$CONDA_PREFIX/bin:\${PATH//\$CONDA_PREFIX\\/bin:/}"
        alias python="\$CONDA_PREFIX/bin/python"
        alias python3="\$CONDA_PREFIX/bin/python3"
    else
        case "\$(alias python 2>/dev/null)" in
            *"\$SCRATCH_DIR/conda_envs/mitoem2/bin/python"*) unalias python ;;
        esac
        case "\$(alias python3 2>/dev/null)" in
            *"\$SCRATCH_DIR/conda_envs/mitoem2/bin/python3"*) unalias python3 ;;
        esac
    fi
    hash -r 2>/dev/null || true
}

if declare -F conda >/dev/null; then
    if ! declare -F __conda_original >/dev/null; then
        eval "\$(declare -f conda | sed '1s/^conda /__conda_original /')"
    fi
    conda() {
        __conda_original "\$@"
        local status=\$?
        __mitoem2_fix_python
        return "\$status"
    }
fi

__mitoem2_fix_python
$marker_end
EOF

    mv "$tmp" "$BASHRC"
}

patch_repo_files() {
    cd "$REPO_DIR"

    if [ -f environment.yml ] && [ ! -f environment.full-export.yml ]; then
        cp environment.yml environment.full-export.yml
    fi

    cat > environment.yml <<'EOF'
name: mitoem2
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - git
  - gh
  - setuptools
  - wheel
  - affogato=0.4.2
  - pip:
      - -r requirements.txt
EOF

    if [ -f requirements.txt ]; then
        sed -i '/^affogato @ git+https:\/\/github.com\/constantinpape\/affogato.git@0.4.2$/d' requirements.txt
        sed -i 's|^torch @ https://download-r2.pytorch.org/whl/cu126/torch-2\.6\.0%2Bcu126-cp39-cp39-manylinux_2_28_x86_64\.whl.*|torch==2.6.0+cu126|' requirements.txt
        sed -i 's|^torchvision @ https://download-r2.pytorch.org/whl/cu126/torchvision-0\.21\.0%2Bcu126-cp39-cp39-linux_x86_64\.whl.*|torchvision==0.21.0+cu126|' requirements.txt
        grep -qxF -- '--extra-index-url https://download.pytorch.org/whl/cu126' requirements.txt || {
            tmp_req="$(mktemp)"
            {
                sed -n '1,/^[^#]/p' requirements.txt | sed '$d'
                echo '--extra-index-url https://download.pytorch.org/whl/cu126'
                sed -n '/^[^#]/,$p' requirements.txt
            } > "$tmp_req"
            mv "$tmp_req" requirements.txt
        }
    fi

    if [ -f setup.py ]; then
        "$MINIFORGE_DIR/bin/python" "$REPO_DIR/scripts_patch_setup.py" 2>/dev/null || "$MINIFORGE_DIR/bin/python" - <<'PY'
from pathlib import Path

path = Path("setup.py")
text = path.read_text()
old = 'requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]'
new = '''requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]'''
if old in text:
    text = text.replace(old, new)
path.write_text(text)
PY
    fi

    if [ -f pyproject.toml ] && ! grep -q '^\[project\.scripts\]' pyproject.toml; then
        "$MINIFORGE_DIR/bin/python" - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
block = '''[project.scripts]
mitoem2-train = "mitoem2.scripts.train:main"
mitoem2-evaluate = "mitoem2.scripts.evaluate:main"
mitoem2-inference = "mitoem2.scripts.inference:main"

'''
marker = "[tool.setuptools]"
if marker in text:
    text = text.replace(marker, block + marker, 1)
else:
    text = text.rstrip() + "\n\n" + block
path.write_text(text)
PY
    fi
}

create_activation_hook() {
    mkdir -p "$ENV_PREFIX/etc/conda/activate.d"
    cat > "$ENV_PREFIX/etc/conda/activate.d/ensure-mitoem2-python.sh" <<'EOF'
#!/usr/bin/env bash
export PATH="$CONDA_PREFIX/bin:${PATH//$CONDA_PREFIX\/bin:/}"
alias python="$CONDA_PREFIX/bin/python"
alias python3="$CONDA_PREFIX/bin/python3"
hash -r 2>/dev/null || true
EOF
}

create_or_update_env() {
    cd "$REPO_DIR"
    local conda_bin="$MINIFORGE_DIR/bin/conda"
    local mamba_bin="$MINIFORGE_DIR/bin/mamba"

    # shellcheck disable=SC1091
    source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
    conda config --set auto_activate_base false >/dev/null

    if [ -x "$mamba_bin" ]; then
        if [ -d "$ENV_PREFIX/conda-meta" ]; then
            "$mamba_bin" env update -p "$ENV_PREFIX" -f environment.yml
        else
            "$mamba_bin" env create -p "$ENV_PREFIX" -f environment.yml
        fi
    else
        if [ -d "$ENV_PREFIX/conda-meta" ]; then
            "$conda_bin" env update -p "$ENV_PREFIX" -f environment.yml --solver=classic
        else
            "$conda_bin" env create -p "$ENV_PREFIX" -f environment.yml --solver=classic
        fi
    fi

    create_activation_hook
    "$ENV_PREFIX/bin/python" -m pip install -e "$REPO_DIR"
}

verify_env() {
    "$ENV_PREFIX/bin/python" - <<'PY'
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
}

install_miniforge
ensure_bashrc_block
patch_repo_files
create_or_update_env
verify_env

cat <<EOF

Done.

Open a new terminal, or run:
  source "$BASHRC"
  conda activate "$ENV_PREFIX"
  python --version

Expected Python: 3.11.x
EOF
