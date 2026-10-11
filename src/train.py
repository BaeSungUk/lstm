"""Run reproducible CPU experiments: python -m src.train."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import CATEGORY, SESSION, Examples, clean_data, download_data, file_sha256, make_examples, split_sessions
from .metrics import evaluate
from .model import CategoryLSTM

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def loader(examples: Examples, shuffle=False, batch_size=64) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(examples.x), torch.from_numpy(examples.lengths),
                            torch.from_numpy(examples.y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def loss_on(model, batches, criterion) -> float:
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for x, lengths, y in batches:
            total += criterion(model(x, lengths), y).item() * len(y)
            count += len(y)
    return total / count


def fit_model(train: Examples, validation: Examples, checkpoint: Path, epochs=20):
    if epochs < 1:
        raise ValueError('Epochs must be positive.')
    model = CategoryLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.CrossEntropyLoss()
    train_batches, val_batches = loader(train, shuffle=True), loader(validation)
    best_loss, best_epoch, stale, history = float('inf'), 0, 0, []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for x, lengths, y in train_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, lengths), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val_loss = loss_on(model, val_batches, criterion)
        history.append({'epoch': epoch, 'train_loss': total / len(train.y), 'validation_loss': val_loss})
        print(f'length={train.x.shape[1]:2d} epoch={epoch:2d} '
              f'train={history[-1]["train_loss"]:.4f} validation={val_loss:.4f}', flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    return model, history, {'best_epoch': best_epoch, 'epochs_run': len(history),
                            'best_validation_loss': best_loss,
                            'seconds': time.perf_counter() - start,
                            'parameters': sum(p.numel() for p in model.parameters())}


def predict(model, examples: Examples) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, lengths, _ in loader(examples, batch_size=512):
            predictions.append(model(x, lengths).argmax(dim=1).numpy())
    return np.concatenate(predictions)


def markov_predictions(train: Examples, test: Examples):
    counts = np.zeros((4, 4), dtype=np.int64)
    np.add.at(counts, (train.last, train.y), 1)
    choices = counts.argmax(axis=1)
    # For an unseen previous category, fall back to the TRAIN target majority.
    choices[counts.sum(axis=1) == 0] = np.bincount(train.y, minlength=4).argmax()
    return choices[test.last], counts


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run_experiments(output: Path, lengths=(1, 5, 10), epochs=20, seed=42, threads=2):
    if epochs < 1 or threads < 1 or not lengths or any(n < 1 for n in lengths):
        raise ValueError('Epochs, threads and all history lengths must be positive.')
    if len(set(lengths)) != len(lengths):
        raise ValueError('History lengths must be unique.')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    seed_everything(seed)
    raw_path = download_data(ROOT / 'data')
    frame, audit = clean_data(pd.read_csv(raw_path, sep=';'))
    parts, split_audit = split_sessions(frame)
    baseline = {name: make_examples(part, 1) for name, part in parts.items()}
    for name in parts:
        split_audit[name]['examples'] = len(baseline[name].y)
        split_audit[name]['category_target_counts'] = np.bincount(baseline[name].y, minlength=4).tolist()
    groups = [set(p[SESSION]) for p in parts.values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    save_json(output / 'data_audit.json', {'cleaning': audit, 'split': split_audit,
                                         'csv_sha256': file_sha256(raw_path)})
    # Manifest makes the exact session split independently inspectable.
    pd.concat([p.groupby(SESSION)['date'].agg(['min', 'max']).assign(split=name)
               for name, p in parts.items()]).to_csv(output / 'split_manifest.csv', index_label='session_id')
    environment = {package: importlib.metadata.version(package)
                   for package in ['torch', 'numpy', 'pandas', 'scikit-learn', 'matplotlib']}
    environment.update(python=platform.python_version(), platform=platform.platform(),
                       device='cpu', threads=threads)
    config = {'seed': seed, 'lengths': list(lengths), 'epochs': epochs, 'patience': 4,
              'batch_size': 64, 'learning_rate': .001, 'input_size': 4, 'hidden_size': 32,
              'num_layers': 1, 'environment': environment}
    save_json(output / 'config.json', config)
    test = baseline['test']
    predictions = pd.DataFrame({'session_id': test.session_ids, 'target_order': test.orders,
                                'last': test.last, 'target': test.y, 'is_switch': test.y != test.last})
    predictions['Repeat-last'] = test.last
    predictions['Markov'] , counts = markov_predictions(baseline['train'], test)
    pd.DataFrame(counts, index=range(1, 5), columns=range(1, 5)).to_csv(output / 'transition_counts.csv')
    metrics = {name: evaluate(test.y, predictions[name], test.last) for name in ['Repeat-last', 'Markov']}
    histories, training = {}, {}
    for length in lengths:
        seed_everything(seed)
        sets = {name: make_examples(part, length) for name, part in parts.items()}
        for name, examples in sets.items():
            reference = baseline[name]
            assert np.array_equal(examples.y, reference.y)
            assert np.array_equal(examples.session_ids, reference.session_ids)
            assert np.array_equal(examples.orders, reference.orders)
        label = f'LSTM-{length}'
        checkpoint = output / 'checkpoints' / f'lstm_{length}.pt'
        model, history, info = fit_model(sets['train'], sets['validation'], checkpoint, epochs)
        pred = predict(model, sets['test'])
        metrics[label] = evaluate(test.y, pred, test.last)
        histories[label], training[label] = history, info
        predictions[label] = pred
        # Save after each completed run; interrupted experiments retain evidence.
        save_json(output / 'metrics.json', metrics)
        save_json(output / 'history.json', histories)
        save_json(output / 'training.json', training)
        predictions.to_csv(output / 'predictions.csv', index=False)
        print(f'{label}: accuracy={metrics[label]["accuracy"]:.4f} '
              f'switch={metrics[label]["switch_accuracy"]}', flush=True)
    table = pd.DataFrame([{ 'model': name, **{k: v for k, v in result.items()
                                              if k not in ['confusion_matrix', 'class_support']}}
                           for name, result in metrics.items()])
    table.to_csv(output / 'metrics.csv', index=False)
    from .report import create_figures
    create_figures(output)
    print(table.to_string(index=False), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lengths', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    try:
        run_experiments(args.output, args.lengths, args.epochs, args.seed, args.threads)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f'Experiment failed: {exc}\n')


if __name__ == '__main__':
    main()
