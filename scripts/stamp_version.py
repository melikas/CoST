"""Write CODE_VERSION (git commit and dirty flag) before uploading a copy without .git.

    python scripts/stamp_version.py

scripts/run_experiment.py records it in every manifest and refuses scientific runs whose
code is uncommitted.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*command):
    return subprocess.run(['git', *command], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


commit = git('rev-parse', 'HEAD')
dirty = bool(git('status', '--porcelain', '--untracked-files=no'))
(ROOT / 'CODE_VERSION').write_text(json.dumps(dict(git_commit=commit, git_dirty=dirty)) + '\n')
print(f'CODE_VERSION: {commit} ({"DIRTY: commit first" if dirty else "clean"})')
