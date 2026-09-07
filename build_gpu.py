from __future__ import annotations

"""Build CUDA RIS DLL with installed x64 MSVC + nvcc."""

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parent
CUDA_ROOT = Path(os.environ.get(
    "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"))
VCVARS_CANDIDATES = [
    Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
]


def main() -> int:
    nvcc = CUDA_ROOT / "bin" / "nvcc.exe"
    vcvars = next((p for p in VCVARS_CANDIDATES if p.exists()), None)
    if not nvcc.exists():
        raise SystemExit(f"nvcc not found: {nvcc}")
    if vcvars is None:
        raise SystemExit("MSVC vcvars64.bat not found")
    dll = ROOT / "qldpc_gpu.dll"
    source = ROOT / "qldpc_gpu.cu"
    cmd = (
        f'call "{vcvars}" && "{nvcc}" -O3 -arch=sm_86 --shared '
        f'-Xcompiler=/MD -Xcompiler=/EHsc -Xptxas=-dlcm=ca '
        f'-o "{dll}" "{source}"'
    )
    subprocess.run(cmd, cwd=ROOT, check=True, shell=True)
    print(dll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
