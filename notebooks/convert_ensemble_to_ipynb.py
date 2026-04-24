"""Convert the ensemble training .py script into a proper .ipynb notebook."""
import json
import re

INPUT = r"c:\EDI_FINAL _V2\notebooks\ensemble_odir_training.py"
OUTPUT = r"c:\EDI_FINAL _V2\notebooks\Ensemble_ODIR5K_Training.ipynb"

with open(INPUT, 'r', encoding='utf-8') as f:
    content = f.read()

# Split on '# %%' cell markers
raw_cells = re.split(r'^# %%', content, flags=re.MULTILINE)

cells = []
for raw in raw_cells:
    raw = raw.strip()
    if not raw:
        continue

    if raw.startswith(' [markdown]'):
        text = raw[len(' [markdown]'):].strip()
        lines = []
        for line in text.split('\n'):
            line = line.rstrip()
            if line.startswith('# '):
                lines.append(line[2:])
            elif line == '#':
                lines.append('')
            else:
                lines.append(line)
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [l + '\n' for l in lines]
        })
    else:
        lines = raw.split('\n')
        source_lines = []
        for line in lines:
            source_lines.append(line + '\n')
        if source_lines and source_lines[-1] == '\n':
            source_lines = source_lines[:-1]
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip('\n')

        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": source_lines,
            "outputs": [],
            "execution_count": None
        })

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python"
        },
        "accelerator": "GPU",
        "gpuClass": "standard"
    },
    "cells": cells
}

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook saved to: {OUTPUT}")
print(f"   Total cells: {len(cells)}")
md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
code_count = sum(1 for c in cells if c['cell_type'] == 'code')
print(f"   Markdown: {md_count} | Code: {code_count}")
