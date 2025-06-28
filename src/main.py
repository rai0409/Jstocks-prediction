import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ========================
# データ読み込みと分割
# ========================
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # TickerとDateでソート
    df = df.sort_values(['Ticker', 'Date'])

    # 対数収益率に変更
    df["target"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))  # 対数収益率

    # NaN, inf を含む行を削除（shiftの影響やゼロ割りなど）
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["target"])

    # データ分割
    train_df = df[df['Date'] < '2023-01-01']
    val_df = df[df['Date'] >= '2023-01-01']

    print(f"{file_path} の Trainデータ件数: {len(train_df)}")
    print(f"{file_path} の Validationデータ件数: {len(val_df)}")

    return train_df, val_df

# --- TimeSeriesDataSet の作成 ---
def create_datasets(train_df, val_df):
    known_reals = ["weekday", "SMA_20", "SMA_100", "SMA_200", "RSI", "MACD"]
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",  # ここを 'target' に変更
        group_ids=["Ticker"],
        time_varying_unknown_reals=["Adj Close"],
        time_varying_known_reals=known_reals,
        categorical_encoders={"Ticker": NaNLabelEncoder(add_nan=True)},
        target_normalizer=GroupNormalizer(groups=["Ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        max_encoder_length=60,
        min_encoder_length=60,
        max_prediction_length=30,
        min_prediction_length=30,
        allow_missing_timesteps=True
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    return training, validation

# --- モデル学習 ---
def train_model(training, validation, group_name):
    train_loader = training.to_dataloader(train=True, batch_size=256, num_workers=16)
    val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=16)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=5e-4,  # 推奨された learning_rate に変更
        hidden_size=128,
        attention_head_size=8,
        dropout=0.2,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=50,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="model-split-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-3,
        mode="min"
    )

    trainer = Trainer(
        max_epochs=50,  # 最大エポック数は50
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_progress_bar=True
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(f"✅ Training start: {group_name}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"✅ Training done: {group_name}")

    best_model_path = checkpoint_callback.best_model_path
    print(f"✅ Best model saved at: {best_model_path}")

    return best_model_path

# --- モデルのロードと予測 ---
def predict(best_model_path, val_loader):
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    predictions = best_model.predict(val_loader, mode="raw", return_x=True)
    return predictions

# --- 評価指標の計算 ---
def evaluate_predictions(predictions):
    preds = predictions.output.prediction[:, :, 0]
    actuals = predictions.x["decoder_target"]

    preds_np = preds.detach().cpu().numpy()
    acts_np = actuals.detach().cpu().numpy()

    mae = mean_absolute_error(acts_np.flatten(), preds_np.flatten())
    rmse = np.sqrt(mean_squared_error(acts_np.flatten(), preds_np.flatten()))

    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ RMSE: {rmse:.2f}")

    return preds_np, acts_np, mae, rmse

# --- 予測結果の可視化 ---
def plot_predictions(acts_np, preds_np, group_name):
    plt.figure(figsize=(12, 6))
    plt.plot(acts_np.flatten(), label='Actual', color='blue', alpha=0.7)
    plt.plot(preds_np.flatten(), label='Prediction', color='red', alpha=0.7)
    plt.title(f"Actual vs Prediction - {group_name.replace('_', ' ')}")
    plt.xlabel("Time Steps")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # スペースを自動調整してレイアウトをきれいに
    plt.savefig(f"actual_vs_prediction_{group_name}.png")
    plt.show()

# --- メイン処理 ---
def main():
    group_files = {
        "Group1_Industry_Materials": "Group1_Industry_Materials.csv",
        "Group2_Consumers": "Group2_Consumers.csv",
        "Group3_Tech_Comm_Utilities": "Group3_Tech_Comm_Utilities.csv",
        "Group4_Finance_Health_Energy": "Group4_Finance_Health_Energy.csv"
    }

    results = []

    for group_name, file_path in group_files.items():
        print(f"🔁 Processing {group_name}...")
        train_df, val_df = load_and_prepare_data(file_path)

        training, validation = create_datasets(train_df, val_df)
        best_model_path = train_model(training, validation, group_name)
        val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=16)
        raw_output = predict(best_model_path, val_loader)

        preds_np, acts_np, mae, rmse = evaluate_predictions(raw_output)
        plot_predictions(acts_np, preds_np, group_name)
        results.append({"Group": group_name, "MAE": mae, "RMSE": rmse})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print("✅ Results saved to results.csv")
  

if __name__ == "__main__":
    main()
