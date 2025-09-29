import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPool2D, Flatten, Dropout
from datetime import datetime, timedelta
import json
import time
import schedule
import os
import logging
import pickle
import ambient
import requests
import argparse
import glob

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_training_and_upload.log'),
        logging.StreamHandler()
    ]
)

# Ambientクライアントの初期化
am = ambient.Ambient("YOUR_CHANNEL_ID", "YOUR_WRITE_KEY", "YOUR_READ_KEY", "YOUR_USER_KEY")


class AutoTrainerAndUploader:
    def __init__(self, model_dir='./models', ip_address=None):
        """
        自動学習とアップロードを行うクラスの初期化

        Parameters:
        model_dir (str): モデルを保存するディレクトリ
        ip_address (str): M5StackのIPアドレス（指定されている場合は学習後に自動アップロード）
        """
        self.model_dir = model_dir
        self.ip_address = ip_address

        # モデル保存ディレクトリの作成
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # スケーラーの初期化
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # 学習パラメータの設定
        self.time_steps = 24
        self.batch_size = 16
        self.epochs = 200

        # ラグ特徴量の設定
        self.lags = 10  # 過去10ポイントのデータを使用
        self.use_rolling_stats = True  # 移動平均などの統計量を使用するか
        self.rolling_window = 5  # 移動平均の計算ウィンドウサイズ

        logging.info("AutoTrainerAndUploaderを初期化しました")
        if ip_address:
            logging.info(f"学習後のモデルを自動的に {ip_address} にアップロードします")

    def fetch_data_from_ambient(self):
        """
        Ambientデータベースからデータを取得する関数

        Returns:
        DataFrame: 取得したデータ
        """
        try:
            # 現在時刻を取得
            now = datetime.now()
            # 指定された形式（2025-4-20 08:49:59）に変換
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

            # 過去24時間のデータを取得（範囲は適宜調整可能）
            time_delta = 24  # 取得する時間範囲（時間単位）
            start_time = (now - pd.Timedelta(hours=time_delta)).strftime("%Y-%m-%d %H:%M:%S")

            logging.info(f"Ambientからデータ取得: {start_time} から {formatted_time}")

            # Ambientからデータを取得
            d = am.read(
                start='2025-4-22 00:00:00',
                end=formatted_time,
                timeout=30)

            # DataFrameに変換
            df = pd.DataFrame(d)

            # カラム名のマッピング
            column_mapping = {
                'd1': 'temp',  # 温度
                'd2': 'humi',  # 湿度
                'd3': 'atm',  # 気圧
                'd4': 'hot',  # 暑さ指数
                'd5': 'uncon',  # 不快指数（予測対象）
                'd6': 'd6',  # その他のデータ（未使用）
                'created': 'timestamp'  # タイムスタンプ
            }
            df = df.rename(columns=column_mapping)

            # タイムスタンプを日時型に変換
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # 欠損値の確認と処理
            df['atm'] = df['atm'].replace(0, np.nan)
            df = df.ffill()  # 前の値で埋める

            logging.info(f"Ambientから{len(df)}件のデータを取得しました")
            return df

        except Exception as e:
            logging.error(f"データ取得エラー: {e}")
            return None

    def preprocess_data(self, df):
        """
        データの前処理を行う関数

        Parameters:
        df (DataFrame): 元のデータフレーム

        Returns:
        tuple: 訓練データとテストデータのセット
        """
        try:
            # 特徴量とターゲット変数を分離
            X = df[['temp', 'humi', 'atm', 'hot']].copy()
            y = df['uncon']

            # 特徴量エンジニアリング - ラグ特徴量の作成
            for col in X.columns:
                for lag in range(1, self.lags + 1):
                    X.loc[:, f'{col}_lag_{lag}'] = X[col].shift(lag)

            # ローリング統計量（移動平均、標準偏差など）の追加
            if self.use_rolling_stats:
                for col in ['temp', 'humi', 'atm', 'hot']:
                    # 移動平均
                    X.loc[:, f'{col}_rolling_mean_{self.rolling_window}'] = X[col].rolling(
                        window=self.rolling_window).mean()
                    # 移動標準偏差
                    X.loc[:, f'{col}_rolling_std_{self.rolling_window}'] = X[col].rolling(
                        window=self.rolling_window).std()
                    # 移動最大値と最小値の差
                    X.loc[:, f'{col}_rolling_range_{self.rolling_window}'] = X[col].rolling(
                        window=self.rolling_window).max() - \
                                                                             X[col].rolling(
                                                                                 window=self.rolling_window).min()

            # 欠損値を除去（ラグ特徴量を作成したことによる欠損）
            X = X.dropna()
            y = y.loc[X.index]

            # データの正規化
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))

            # 時系列データをCNN用に整形
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.time_steps)

            # トレーニングデータとテストデータに分割（最後の20%をテストデータとする）
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

            # CNNモデルの入力形状に変換
            input_dim = X_train.shape[2]  # 特徴量の数
            X_train = X_train.reshape(X_train.shape[0], self.time_steps, input_dim, 1)
            X_test = X_test.reshape(X_test.shape[0], self.time_steps, input_dim, 1)

            logging.info(f"データ前処理完了: 訓練データ {X_train.shape}, テストデータ {X_test.shape}")
            logging.info(f"特徴量の数: {input_dim}")
            return X_train, y_train, X_test, y_test, input_dim

        except Exception as e:
            logging.error(f"データ前処理エラー: {e}")
            return None, None, None, None, None

    def create_sequences(self, X, y, time_steps):
        """
        時系列データをシーケンスに変換する関数

        Parameters:
        X: 特徴量データ
        y: ターゲットデータ
        time_steps: シーケンスの長さ

        Returns:
        tuple: X_seqとy_seq
        """
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    def build_model(self, input_dim):
        """
        CNNモデルを構築する関数

        Parameters:
        input_dim: 入力特徴量の次元

        Returns:
        Model: 構築したモデル
        """
        inputs = tf.keras.Input(shape=(self.time_steps, input_dim, 1))
        x = tf.keras.layers.Conv2D(8, (4, 3), padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPool2D((3, 3), padding="same")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Conv2D(16, (4, 1), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool2D((3, 1), padding="same")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1)(x)  # 回帰問題なので活性化関数なし

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # モデルのコンパイル
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer='adam',
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

        logging.info("CNNモデルを構築しました")
        return model

    def train_model(self, X_train, y_train, X_test, y_test, input_dim):
        """
        モデルを訓練する関数

        Parameters:
        X_train, y_train: 訓練データ
        X_test, y_test: テストデータ
        input_dim: 入力特徴量の次元

        Returns:
        tuple: 訓練したモデルと履歴
        """
        try:
            # モデルの構築
            model = self.build_model(input_dim)

            # 学習の早期終了
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True
            )

            # モデルの学習
            history = model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                callbacks=[early_stopping],
                verbose=1
            )

            # テストデータでの評価
            y_pred_scaled = model.predict(X_test)

            # スケーリングを元に戻す
            y_test_inv = self.scaler_y.inverse_transform(y_test)
            y_pred_inv = self.scaler_y.inverse_transform(y_pred_scaled)

            # 評価指標の計算
            mse = np.mean((y_test_inv - y_pred_inv) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_inv - y_pred_inv))

            logging.info(f"モデル学習完了 - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            return model, history, mse, rmse, mae

        except Exception as e:
            logging.error(f"モデル学習エラー: {e}")
            return None, None, None, None, None

    def convert_to_tflite(self, model, tflite_path):
        """
        モデルをTFLiteに直接変換する関数

        Parameters:
        model: Kerasモデル
        tflite_path: 保存するTFLiteモデルのパス
        """
        try:
            # TFLiteに変換
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 変換設定
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLiteの組み込み演算を使用
                tf.lite.OpsSet.SELECT_TF_OPS  # TensorFlow演算も使用可能に
            ]

            # 変換実行
            tflite_model = converter.convert()

            # TFLiteモデルを保存
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

            logging.info(f"TFLiteモデルを保存しました: {tflite_path}")
            return tflite_path

        except Exception as e:
            logging.error(f"TFLite変換エラー: {e}")
            return None

    def save_model_and_plot(self, model, history):
        """
        モデルを保存し、学習グラフを生成する関数

        Parameters:
        model: 保存するモデル
        history: 学習履歴

        Returns:
        str: 生成したTFLiteモデルのパス（アップロード用）
        """
        try:
            # 現在日付を取得
            today = datetime.now()
            date_str = today.strftime("%Y%m%d")

            # 日付フォルダを作成
            date_dir = os.path.join(self.model_dir, date_str)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)

            # モデルの保存（.kerasフォーマットで保存）
            model_path = os.path.join(date_dir, f'model_weight_{date_str}.keras')
            model.save(model_path, save_format='keras')

            # バックアップとしてH5形式でも保存
            h5_path = os.path.join(date_dir, f'model_weight_{date_str}.h5')
            model.save(h5_path, save_format='h5')

            # TFLiteモデルへの変換と保存
            tflite_path = os.path.join(date_dir, f'model_weight_{date_str}.tflite')
            tflite_model_path = self.convert_to_tflite(model, tflite_path)

            # 学習グラフの生成と保存
            plt.figure(figsize=(12, 6))

            # 損失のプロット
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True)

            # MAEのプロット
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mean_absolute_error'], label='Training MAE')
            plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
            plt.title('Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True)

            plt.tight_layout()

            # グラフを保存
            plot_path = os.path.join(date_dir, f'learning_curve_{date_str}.png')
            plt.savefig(plot_path)
            plt.close()

            logging.info(f"モデルを保存しました: {model_path}")
            logging.info(f"モデルを保存しました（H5バックアップ）: {h5_path}")
            logging.info(f"学習グラフを保存しました: {plot_path}")

            return tflite_model_path

        except Exception as e:
            logging.error(f"モデル保存エラー: {e}")
            return None

    def find_latest_model(self):
        """./models内の最新の時刻フォルダからモデルファイルを見つける関数"""

        # ./modelsディレクトリの確認
        models_dir = self.model_dir
        if not os.path.exists(models_dir):
            logging.error(f"エラー: '{models_dir}' ディレクトリが見つかりません。")
            return None

        # 時刻フォルダ（YYYYMMDD形式）を見つける
        date_folders = glob.glob(os.path.join(models_dir, "[0-9]" * 8))

        if not date_folders:
            logging.error(f"エラー: '{models_dir}' 内に時刻フォルダ（YYYYMMDD形式）が見つかりません。")
            return None

        # 最新の時刻フォルダを見つける
        latest_folder = max(date_folders)
        folder_date = os.path.basename(latest_folder)

        # フォルダ内のtfliteファイルを検索
        tflite_files = glob.glob(os.path.join(latest_folder, "*.tflite"))

        if not tflite_files:
            logging.error(f"エラー: '{latest_folder}' 内にtfliteファイルが見つかりません。")
            return None

        # model_weight_YYYYMMDD.tflite形式のファイルを優先
        expected_filename = f"model_weight_{folder_date}.tflite"
        expected_path = os.path.join(latest_folder, expected_filename)

        if os.path.exists(expected_path):
            logging.info(f"最新モデルを見つけました: {expected_path}")
            return expected_path
        else:
            # 期待されるファイル名がない場合は、最初のtfliteファイルを使用
            logging.info(f"期待される命名規則のファイルが見つからないため、代わりに {tflite_files[0]} を使用します")
            return tflite_files[0]

    def upload_model(self, ip_address, model_path=None):
        """M5Stackにモデルファイルを自動的にアップロードする関数"""

        # モデルパスが指定されていない場合、最新のモデルを自動検出
        if model_path is None:
            model_path = self.find_latest_model()
            if model_path is None:
                return False

        # URLの設定
        upload_url = f"http://{ip_address}/update"

        # ファイル存在確認
        if not os.path.exists(model_path):
            logging.error(f"エラー: ファイル '{model_path}' が見つかりません。")
            return False

        # ファイルサイズ確認
        file_size = os.path.getsize(model_path) / 1024
        logging.info(f"モデルファイルサイズ: {file_size:.2f} KB")

        # ファイルのオープン
        with open(model_path, 'rb') as model_file:
            files = {'update': model_file}

            logging.info(f"{ip_address} にモデルをアップロード中...")

            try:
                # POSTリクエストでファイルをアップロード
                response = requests.post(upload_url, files=files, timeout=30)

                if response.status_code == 200:
                    logging.info("アップロード成功！")
                    logging.info("M5Stackが再起動中...")
                    time.sleep(5)  # 再起動待機
                    return True
                else:
                    logging.error(f"アップロード失敗。ステータスコード: {response.status_code}")
                    logging.error(f"レスポンス: {response.text}")
                    return False

            except requests.exceptions.RequestException as e:
                logging.error(f"エラー: {e}")
                return False

    def run_training_and_upload(self):
        """
        学習プロセスとアップロードを実行する関数
        """
        logging.info("学習プロセスを開始します")

        # データの取得
        df = self.fetch_data_from_ambient()
        if df is None:
            return

        # データの前処理
        X_train, y_train, X_test, y_test, input_dim = self.preprocess_data(df)
        if X_train is None:
            return

        # モデルの学習
        model, history, mse, rmse, mae = self.train_model(X_train, y_train, X_test, y_test, input_dim)
        if model is None:
            return

        # モデルと学習グラフの保存
        tflite_model_path = self.save_model_and_plot(model, history)

        # IPアドレスが設定されていれば自動アップロード
        if self.ip_address and tflite_model_path:
            logging.info(f"学習したモデルを {self.ip_address} に自動アップロードします")
            upload_result = self.upload_model(self.ip_address, tflite_model_path)
            if upload_result:
                logging.info("モデルのアップロードが完了しました")
            else:
                logging.error("モデルのアップロードに失敗しました")

        logging.info("学習プロセスが完了しました")


def calculate_next_run_time(execution_time):
    """
    次回実行予定時刻を計算する関数

    Parameters:
    execution_time (str): 実行時刻（HH:MM形式）

    Returns:
    datetime: 次回実行予定のdatetimeオブジェクト
    """
    now = datetime.now()
    hour, minute = map(int, execution_time.split(':'))

    # 今日の実行時刻を作成
    today_exec_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # 今日の実行時刻がすでに過ぎている場合は翌日に設定
    if now >= today_exec_time:
        next_run = today_exec_time + timedelta(days=1)
    else:
        next_run = today_exec_time

    return next_run


def run_daily_training_and_upload(execution_time="03:00", ip_address=None):
    """
    毎日指定時刻に学習とアップロードを実行する関数

    Parameters:
    execution_time (str): 実行時刻（HH:MM形式）
    ip_address (str): M5StackのIPアドレス（指定されている場合は学習後に自動アップロード）
    """
    trainer = AutoTrainerAndUploader(ip_address=ip_address)

    # 指定時刻に毎日実行するスケジュールを設定
    schedule.every().day.at(execution_time).do(trainer.run_training_and_upload)

    # 次回の実行予定日時を計算して出力
    next_run = calculate_next_run_time(execution_time)
    logging.info(f"次回の実行予定: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

    logging.info(f"毎日 {execution_time} に学習を実行するスケジュールを設定しました")
    if ip_address:
        logging.info(f"学習後、モデルを {ip_address} に自動アップロードします")

    # 初回実行（即時）
    logging.info("初回学習を即時実行します")
    trainer.run_training_and_upload()

    # 即時実行後に次回の実行予定を更新
    next_run = calculate_next_run_time(execution_time)
    logging.info(f"次回の実行予定: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

    # スケジュールのループ
    while True:
        schedule.run_pending()

        # 1分ごとにチェック
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='M5Stackの機械学習モデルを自動的に学習・アップロードするツール')

    # モード選択オプション
    parser.add_argument('--mode', '-m', choices=['train', 'upload', 'both'], default='both',
                        help='実行モード: train=学習のみ, upload=アップロードのみ, both=学習後アップロード (デフォルト: both)')

    # IPアドレスオプション
    parser.add_argument('--ip', '-i', help='M5StackのIPアドレス (例: 192.168.1.123)')

    # 実行時刻オプション
    parser.add_argument('--time', '-t', default="03:00",
                        help='定期実行時刻 (HH:MM形式, デフォルト: 03:00)')

    # モデルファイルオプション（アップロードのみの場合に使用）
    parser.add_argument('--model-file', '-f',
                        help='アップロードするモデルファイルのパス（指定しない場合は最新のモデルを自動検出）')

    args = parser.parse_args()

    # IPアドレスが必須な場合のチェック
    if args.mode in ['upload', 'both'] and not args.ip:
        parser.error("error: upload/bothモードではIPアドレス (--ip) が必要です")

    # モードに応じた処理
    if args.mode == 'upload':
        # アップロードのみ
        logging.info(f"モデルアップロードモードを開始します")
        trainer = AutoTrainerAndUploader()
        trainer.upload_model(args.ip, args.model_file)

    elif args.mode == 'train':
        # 学習のみ
        logging.info(f"モデル学習モードを開始します")
        run_daily_training_and_upload(args.time, None)

    else:  # 'both'
        # 学習後アップロード
        logging.info(f"学習＆アップロードモードを開始します")
        run_daily_training_and_upload(args.time, args.ip)
