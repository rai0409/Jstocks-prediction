Temporal Fusion Transformer (TFT) による株価予測AI

概要
このプロジェクトは、PyTorch Forecasting と Lightning を使用し、
Temporal Fusion Transformer (TFT) に基づいて構築された 株価予測AI モデルです。
TOPIX1000に2010年以前から上場され続けている企業(2025年4月時点）を特徴別に
4つに分類しそのグループの株価データをもとに、将来の対数収益率（log return）を予測します。

将来的な拡張計画

CatBoostでの財務指標や経済指標分析、FinBERTでの感情分析を導入予定です。

これらの手法を組み合わせたハイブリッドモデルの構築により、より高精度な株価予測を目指します。


プロジェクト構成
Jstocks-Github/
├── data/https://drive.google.com/drive/folders/1A-914ta8x5X8TBnizcGE9YR7NpfOa3tj?usp=sharing
│   ├── Group1_Industry_Materials.csv
│   ├── Group2_Consumers.csv
│   ├── Group3_Tech_Comm_Utilities.csv
│   └── Group4_Finance_Health_Energy.csv
├── images/
│   ├── actual_vs_prediction_Group1_Industry_Materials.png
│   ├──actual_vs_prediction_Group2_Consumers.png
│   ├──actual_vs_prediction_Group3_Tech_Comm_Utilities.png
│   └──actual_vs_prediction_Group4_Finance_Health_Energy.png
├── src/
│   └── main.py                # モデルの学習と評価
├── results.csv                # グループごとのMAE / RMSE結果
├── environment.yml            # 必要なPythonパッケージ一覧
├── README.md                  # このドキュメント
└── .gitignore                 # チェックポイントやログの除外設定

環境構築方法
environment.yml を使って仮想環境を作成
conda env create -f environment.yml

環境をアクティベート
conda activate jstocks-env

実行方法
data/ フォルダに以下のようにCSVデータを配置してください：

Group1_Industry_Materials.csv
Group2_Consumers.csv
Group3_Tech_Comm_Utilities.csv
Group4_Finance_Health_Energy.csv

以下のコマンドでメインスクリプトを実行：

bash
コードをコピーする
python src/main.py
実行結果：

各グループの MAE（平均絶対誤差）・RMSE（二乗平均平方根誤差） が results.csv に保存されます

実際の株価 vs 予測の可視化グラフが images/ フォルダに出力されます

出力例

| Group                        | MAE    | RMSE   |
|-----------------------------|--------|--------|
| Group1_Industry_Materials   | 0.0025 | 0.0058 |
| Group2_Consumers            | 0.0012 | 0.0032 |
| Group3_Tech_Comm_Utilities  | 0.0055 | 0.0158 |
| Group4_Finance_Health_Energy| 0.0018 | 0.0031 |

注意事項
.gitignore により checkpoints/ や lightning_logs/ などの学習ログはGitに含まれません

GPUが利用可能な場合は自動的にGPUで学習されます（torch.cuda.is_available() を使用）

ライセンス
このプロジェクトは MIT ライセンスのもとで公開されています。