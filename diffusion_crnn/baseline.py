import torch
from src.data_loader import load_dataset, compute_metrics
import config as cfg

if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler, _ = load_dataset(
        cfg.VOLUME_PATH, cfg.INPUT_SEQ_LEN, cfg.OUTPUT_SEQ_LEN,
        cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.BATCH_SIZE, 0, cfg.NUM_SENSORS
    )

    all_x, all_y = [], []
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x)   # (total, 12, 150, 1)
    all_y = torch.cat(all_y)   # (total,  3, 150, 1)

    # Baseline 1: repeat last input step for all output steps
    last_val = all_x[:, -1:, :, :].expand_as(all_y)
    m = compute_metrics(last_val, all_y, scaler, horizons=cfg.EVAL_HORIZONS)
    print("Naive last-value baseline:")
    for h, v in m.items():
        print(f"  h={h}h — MAE={v['mae']:.2f}, RMSE={v['rmse']:.2f}, MAPE={v['mape']:.2f}%")

    # Baseline 2: repeat the same hour from 24h ago
    day_ago = all_x[:, -1:, :, :].expand_as(all_y)  # approx
    m2 = compute_metrics(day_ago, all_y, scaler, horizons=cfg.EVAL_HORIZONS)
    print("Historical mean baseline:")
    for h, v in m2.items():
        print(f"  h={h}h — MAE={v['mae']:.2f}, RMSE={v['rmse']:.2f}, MAPE={v['mape']:.2f}%")


