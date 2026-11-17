#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fenics_python="${FENICS_PYTHON:-python}"
steps="${STEPS:-100}"
repeats="${REPEATS:-3}"
output_prefix="${OUTPUT_PREFIX:-cylinder3d_100_8r}"
raw_prefix="${output_prefix}_raw_$$"
output_dir="${repo_root}/test_cases/efficiency/output"
mpi_ranks="${MPI_RANKS:-8}"

if [[ -n "${MPIEXEC:-}" ]]; then
    mpi_exec="${MPIEXEC}"
else
    python_prefix="$("${fenics_python}" -c 'import sys; print(sys.prefix)')"
    mpi_exec="${python_prefix}/bin/mpiexec"
    if [[ ! -x "${mpi_exec}" ]]; then
        mpi_exec="mpiexec"
    fi
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [[ "$(uname -s)" == "Darwin" ]]; then
    # The MPICH sockets provider starts two busy progress threads per rank on
    # macOS. The tcp provider keeps this benchmark at one CPU per rank.
    export FI_PROVIDER="${FI_PROVIDER:-tcp}"
    mpi_binding_args=()
    export RPCS_MPI_BINDING="os-managed"
else
    mpi_version="$("${mpi_exec}" --version 2>&1 || true)"
    if grep -Eqi 'Open MPI|OpenRTE' <<<"${mpi_version}"; then
        mpi_binding_args=(--bind-to core --map-by core)
        export RPCS_MPI_BINDING="openmpi-core"
    elif grep -Eqi 'HYDRA|MPICH' <<<"${mpi_version}"; then
        mpi_binding_args=(-bind-to core)
        export RPCS_MPI_BINDING="mpich-core"
    else
        mpi_binding_args=()
        export RPCS_MPI_BINDING="launcher-default"
    fi
fi

for repeat in $(seq 1 "${repeats}"); do
    case $((repeat % 2)) in
        1) methods=("monolithic-SUPG/PSPG" "RPCS-VMS") ;;
        0) methods=("RPCS-VMS" "monolithic-SUPG/PSPG") ;;
    esac
    for method in "${methods[@]}"; do
        case "${method}" in
            "monolithic-SUPG/PSPG") slug="monolithic" ;;
            "RPCS-VMS") slug="rpcs_vms" ;;
        esac
        run_id="run${repeat}"
        common_args=(
            "${repo_root}/test_cases/efficiency/run.py"
            --steps "${steps}"
            --methods "${method}"
            --run-id "${run_id}"
            --expected-ranks "${mpi_ranks}"
            --output-dir "${output_dir}"
            --output-prefix "${raw_prefix}_${slug}_${run_id}"
        )
        "${mpi_exec}" "${mpi_binding_args[@]}" -n "${mpi_ranks}" \
            "${fenics_python}" "${common_args[@]}" "$@"
    done
done

"${fenics_python}" "${repo_root}/test_cases/efficiency/postprocess.py" \
    --input-dir "${output_dir}" \
    --input-prefix "${raw_prefix}" \
    --output-prefix "${output_prefix}"

# Keep per-method measurements and residual audits for independent review.
