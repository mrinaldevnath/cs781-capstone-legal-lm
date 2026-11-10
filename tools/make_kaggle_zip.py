"""Make the flat code folder (and zip) that a phase's Kaggle notebooks expect.

On Kaggle every notebook copies ONE folder of code into /kaggle/working and runs the scripts from there, so the shared
files in common/ must sit next to the phase files. This script copies common/*.py, common/configs/ and the phase's own
files into one folder and zips it. Upload the zip as a Kaggle dataset and attach it to the phase's notebooks.

    python tools/make_kaggle_zip.py phase4_judgment_prediction      ->  dist/phase4_judgment_prediction_code.zip
    python tools/make_kaggle_zip.py all                             ->  one zip per phase
"""
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(phase):
    src = os.path.join(ROOT, phase)
    assert os.path.isdir(src), f"no folder {phase}"
    out = os.path.join(ROOT, "dist", phase + "_code")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in glob.glob(os.path.join(ROOT, "common", "*.py")):
        shutil.copy(f, out)
    shutil.copytree(os.path.join(ROOT, "common", "configs"), os.path.join(out, "configs"))
    for f in glob.glob(os.path.join(src, "*.py")) + glob.glob(os.path.join(src, "*.json")):
        shutil.copy(f, out)                                   # phase files win over common files with the same name
    z = shutil.make_archive(out, "zip", out)
    print(f"{phase}: {len(os.listdir(out))} files -> {os.path.relpath(z, ROOT)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    phases = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "phase*"))) if which == "all" else [which]
    for p in phases:
        build(p)
