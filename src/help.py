import matplotlib.pyplot as plt
import os
import sys

def save_figure(outputDir, fileName, plt):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'results', outputDir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, fileName)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Uloženo do: {file_path}")
