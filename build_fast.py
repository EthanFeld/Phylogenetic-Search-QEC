from __future__ import annotations

"""Build qldpc_fast.cpp with installed x64 MSVC."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
VCVARS_CANDIDATES = [
    Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
]


def main():
    vcvars = next((p for p in VCVARS_CANDIDATES if p.exists()), None)
    if vcvars is None:
        raise SystemExit("MSVC vcvars64.bat not found")
    dll = ROOT / "qldpc_fast.dll"
    source = ROOT / "qldpc_fast.cpp"
    cmd = f'call "{vcvars}" && cl /nologo /O2 /EHsc /LD /Fe:"{dll}" "{source}"'
    subprocess.run(cmd, cwd=ROOT, check=True, shell=True)
    print(dll)


if __name__ == "__main__":
    main()
