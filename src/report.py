"""Rebuild figures from measured JSON outputs, without retraining."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import NAMES


def create_figures(output: Path):
    metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    histories = json.loads((output / 'history.json').read_text(encoding='utf-8'))
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold',
                         'savefig.dpi': 160})
    names = list(metrics)
    colors = ['#a4adb8' if not name.startswith('LSTM') else '#2878a8' for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, key, title in zip(axes, ['accuracy', 'stay_accuracy', 'switch_accuracy'],
                              ['All next clicks', 'Same-category clicks', 'Category switches']):
        values = [100 * (metrics[name][key] or 0) for name in names]
        bars = ax.bar(names, values, color=colors, width=.65)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.set_ylabel('Accuracy (%)')
        ax.tick_params(axis='x', labelrotation=30)
        ax.yaxis.grid(True, alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Can sequence history help at category switches?', fontsize=15, fontweight='bold')
    fig.savefig(figures / 'model_comparison.png')
    plt.close(fig)

    target = 'LSTM-10' if 'LSTM-10' in metrics else next(n for n in reversed(names) if n.startswith('LSTM'))
    matrix = np.asarray(metrics[target]['confusion_matrix'])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.8, 5.5), layout='constrained')
    heat = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{matrix[i, j]:,}\n{normalized[i, j]:.1%}', ha='center', va='center',
                    color='white' if normalized[i, j] > .55 else '#203040')
    ax.set(xticks=range(4), yticks=range(4), xticklabels=NAMES, yticklabels=NAMES,
           xlabel='Predicted category', ylabel='Actual category', title=f'{target} | Test confusion matrix')
    fig.colorbar(heat, ax=ax, label='Fraction within actual category', shrink=.85)
    fig.savefig(figures / 'confusion_matrix.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, len(histories), figsize=(4.4 * len(histories), 3.8),
                              squeeze=False, layout='constrained')
    for ax, (name, history) in zip(axes[0], histories.items()):
        epochs = [row['epoch'] for row in history]
        for key, label, color in [('train_loss', 'Train', '#2878a8'),
                                  ('validation_loss', 'Validation', '#d76544')]:
            ax.plot(epochs, [row[key] for row in history], 'o-', label=label, color=color, markersize=3)
        best = min(history, key=lambda row: row['validation_loss'])['epoch']
        ax.axvline(best, color='#999999', linestyle=':', label=f'Selected epoch {best}')
        ax.set(title=name, xlabel='Epoch', ylabel='Cross entropy')
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.savefig(figures / 'learning_curves.png')
    plt.close(fig)

    lstms = sorted((n for n in metrics if n.startswith('LSTM-')), key=lambda n: int(n.split('-')[1]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    for ax, key, title, color in zip(axes, ['accuracy', 'switch_accuracy'],
                                    ['All next clicks', 'Category switches'], ['#2878a8', '#d76544']):
        lengths = [int(n.split('-')[1]) for n in lstms]
        values = [100 * (metrics[n][key] or 0) for n in lstms]
        ax.plot(lengths, values, 'o-', color=color, linewidth=2)
        for x, y in zip(lengths, values):
            ax.annotate(f'{y:.2f}%', (x, y), xytext=(0, 8), textcoords='offset points', ha='center')
        ax.set(title=title, xlabel='Maximum observed history (clicks)', ylabel='Accuracy (%)',
               xticks=lengths, ylim=(0, 105))
        ax.grid(alpha=.15)
    fig.suptitle('Same target clicks, different amounts of history', fontweight='bold')
    fig.savefig(figures / 'history_length.png')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    create_figures(parser.parse_args().output)
