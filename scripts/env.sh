# Shared defaults for the shell entry points in scripts/.  Every value can be
# overridden from the environment before a script is started.
#   APE_ROOT      repository root (default: the parent of this directory)
#   APE_PY        Python interpreter of the pinned environment (default: python)
#   APE_TMP       scratch directory for logs, pid files and run records (default: $APE_ROOT/tmp)
#   APE_ACCIDENT  ACCIDENT dataset root with metadata-real.csv and real_videos/ (default: $APE_ROOT/external/ACCIDENT_2026)
#   HF_HOME       Hugging Face cache (default: ~/.cache/huggingface)
#   TORCH_HOME    torchvision weight cache (default: ~/.cache/torch)
_ape_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${APE_ROOT:=$(cd "${_ape_env_dir}/.." && pwd)}"
: "${APE_PY:=python}"
: "${APE_TMP:=${APE_ROOT}/tmp}"
: "${APE_ACCIDENT:=${APE_ROOT}/external/ACCIDENT_2026}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"
: "${TORCH_HOME:=${HOME}/.cache/torch}"
export APE_ROOT APE_PY APE_TMP APE_ACCIDENT HF_HOME TORCH_HOME
unset _ape_env_dir
