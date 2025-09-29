/**
 * @file ENV_III_TFLite_Prediction_SPIFFS.ino
 * @brief M5Stack + ENV III + TensorFlow Liteを使った快適度予測プログラム（SPIFFS版）
 * @version 2.0 - ラグ特徴量と時系列データ対応
 * 
 * @Hardwares: M5Core + Unit ENV_III
 * @Dependent Libraries:
 * - M5Stack
 * - M5UnitENV
 * - TensorFlowLite
 * - Ambient (オプション)
 */

 #include <Wire.h>
 #include "M5Stack.h"
 #include "M5UnitENV.h"
 #include "Ambient.h"
 
 // SPIFFS関連のライブラリを追加
 #include <SPIFFS.h>
 #include <WebServer.h>
 
 #ifdef min
   #undef min
 #endif
 
 #ifdef max
   #undef max
 #endif
 
 #include <TensorFlowLite_ESP32.h>
 #include "all_ops_resolver.h"
 #include "micro_error_reporter.h"
 #include "micro_interpreter.h"
 #include "schema_generated.h"
 #include "flatbuffers.h"
 
 // モデルデータをインクルード（初回用）
 #include "model.h"
 
 // Ambientの設定
 unsigned int channelId = <your_channel_ID>;
 const char* writeKey = "<your_write_key>";
 
 // WiFi設定
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

 
 // センサーインスタンス
 SHT3X sht3x;
 QMP6988 qmp;
 
 // Ambient設定
 WiFiClient client;
 Ambient ambient;
 
 // WebServerのインスタンス作成
 WebServer server(80);
 
 // センサーデータ格納変数
 float temperature = 0.0;
 float humidity = 0.0;
 float pressure = 0.0;
 float WBGT = 0.0;  // 暑さ指数
 float THI = 0.0;   // 不快指数
 float predicted_comfort = 0.0; // 予測された快適度
 
 // データサンプリング用の変数
 const int time_steps = 24;  // MLモデルで使用する時系列ステップ数
 const int feature_count = 4; // 基本特徴量の数 [temp, humi, atm, hot]
 
 // ラグ特徴量と統計量の設定
 const int lag_count = 10;  // ラグ特徴量の数
 const bool use_rolling_stats = true; // 統計量を使用するか
 const int rolling_window = 5; // 移動平均などの計算ウィンドウサイズ
 
 // 入力次元の計算
 // 基本特徴量 + ラグ特徴量 + ローリング統計（オプション）
 const int input_dim = feature_count + 
                      (lag_count * feature_count) + 
                      (use_rolling_stats ? feature_count * 3 : 0);
 
 // センサーデータ保存配列
 float raw_sensor_data[time_steps][feature_count];  // 元の測定値を保存
 float input_tensor_data[time_steps][input_dim];    // 特徴量エンジニアリング後のデータ
 int data_index = 0;        // 現在のデータインデックス
 bool buffer_filled = false; // バッファが満杯になったかどうか
 
 // TensorFlow Lite用の変数
 tflite::MicroErrorReporter tflErrorReporter;
 tflite::AllOpsResolver tflOpsResolver;
 const tflite::Model* tflModel = nullptr;
 tflite::MicroInterpreter* tflInterpreter = nullptr;
 TfLiteTensor* tflInputTensor = nullptr;
 TfLiteTensor* tflOutputTensor = nullptr;
 
 // MLモデル用のテンソルアリーナサイズ
 constexpr int kTensorArenaSize = 64 * 1024; // ラグ特徴量対応のため拡大
 byte tensorArena[kTensorArenaSize] __attribute__((aligned(16)));
 
 // データの正規化に必要な値（学習時のMin-Maxスケーリングの値）
 // 特徴量ごとのmin-max値（学習データから得られた値）
 const float feature_min[4] = {22.29, 23.13, 1004.84, 17.28}; // temp, humi, atm, hot の最小値
 const float feature_max[4] = {28.52, 57.98, 1012.54, 23.17}; // temp, humi, atm, hot の最大値
 const float target_min = 67.78; // uncon の最小値
 const float target_max = 76.34; // uncon の最大値
 
 // グローバル変数としてバージョンを保持
 uint32_t modelVersion = 1;
 
 // Webサーバーのハンドラー関数
 void handleRoot() {
   String html = "<html><body>";
   html += "<h1>M5Stack Model Manager</h1>";
   html += "<p>Current Model: model.tflite (v" + String(modelVersion) + ")</p>";
   html += "<a href='/upload'>Update Model</a>";
   html += "</body></html>";
   server.send(200, "text/html", html);
 }
 
 void handleUploadForm() {
   String html = "<html><body>";
   html += "<h1>Update Model</h1>";
   html += "<form method='POST' action='/update' enctype='multipart/form-data'>";
   html += "<input type='file' name='update'>";
   html += "<input type='submit' value='Update'>";
   html += "</form>";
   html += "</body></html>";
   server.send(200, "text/html", html);
 }
 
 void handleUpdateComplete() {
   server.sendHeader("Connection", "close");
   server.send(200, "text/plain", "Update complete. Restarting...");
   delay(1000);
   ESP.restart();
 }
 
 void handleUpdateUpload() {
   HTTPUpload& upload = server.upload();
   
   if(upload.status == UPLOAD_FILE_START) {
     // アップロード開始
     Serial.printf("Receiving model file: %s\n", upload.filename.c_str());
     // SPIFFSにファイルを作成
     File modelFile = SPIFFS.open("/model.tflite.new", "w");
     if(!modelFile) {
       Serial.println("Failed to open file for writing");
       return;
     }
   } else if(upload.status == UPLOAD_FILE_WRITE) {
     // データ書き込み
     File modelFile = SPIFFS.open("/model.tflite.new", "a");
     if(modelFile) {
       modelFile.write(upload.buf, upload.currentSize);
       modelFile.close();
     }
   } else if(upload.status == UPLOAD_FILE_END) {
     // アップロード完了
     Serial.printf("Upload complete: %u bytes\n", upload.totalSize);
     
     // バージョン番号をインクリメント
     modelVersion++;
     saveModelVersion(modelVersion);
     
     // 古いモデルファイルを削除し、新しいファイルを有効化
     if(SPIFFS.exists("/model.tflite")) {
       SPIFFS.remove("/model.tflite.bak");
       SPIFFS.rename("/model.tflite", "/model.tflite.bak");
     }
     SPIFFS.rename("/model.tflite.new", "/model.tflite");
   }
 }
 
 // 初回起動時にmodel.hからSPIFFSにモデルファイルを作成する関数
 void createInitialModelFile() {
   File modelFile = SPIFFS.open("/model.tflite", "w");
   if(modelFile) {
     // model.hからのデータをファイルに書き込む
     modelFile.write(model, sizeof(model));
     modelFile.close();
     Serial.println("初期モデルファイルを作成しました");
   } else {
     Serial.println("初期モデルファイルの作成に失敗しました");
   }
 }
 
 // データを正規化する関数
 float normalize(float value, float min_val, float max_val) {
   return (value - min_val) / (max_val - min_val);
 }
 
 // 正規化を元に戻す関数
 float denormalize(float normalized_value, float min_val, float max_val) {
   return normalized_value * (max_val - min_val) + min_val;
 }
 
 // 移動平均を計算する関数
 float calculateRollingMean(float* data, int window_size, int feature_idx) {
   float sum = 0.0;
   int count = 0;
   
   for (int i = 0; i < window_size; i++) {
     // 現在の位置から過去のwindow_size分のデータを使用
     int pos = (data_index - i - 1 + time_steps) % time_steps;
     sum += raw_sensor_data[pos][feature_idx];
     count++;
   }
   
   return (count > 0) ? sum / count : 0.0;
 }
 
 // 移動標準偏差を計算する関数
 float calculateRollingStd(float* data, int window_size, int feature_idx, float mean) {
   float sum_squared_diff = 0.0;
   int count = 0;
   
   for (int i = 0; i < window_size; i++) {
     // 現在の位置から過去のwindow_size分のデータを使用
     int pos = (data_index - i - 1 + time_steps) % time_steps;
     float diff = raw_sensor_data[pos][feature_idx] - mean;
     sum_squared_diff += diff * diff;
     count++;
   }
   
   return (count > 1) ? sqrt(sum_squared_diff / (count - 1)) : 0.0;
 }
 
 // 移動範囲（最大値-最小値）を計算する関数
 float calculateRollingRange(float* data, int window_size, int feature_idx) {
   float min_val = INFINITY;
   float max_val = -INFINITY;
   
   for (int i = 0; i < window_size; i++) {
     // 現在の位置から過去のwindow_size分のデータを使用
     int pos = (data_index - i - 1 + time_steps) % time_steps;
     float val = raw_sensor_data[pos][feature_idx];
     
     if (val < min_val) min_val = val;
     if (val > max_val) max_val = val;
   }
   
   return (min_val != INFINITY && max_val != -INFINITY) ? (max_val - min_val) : 0.0;
 }
 
 // 特徴量エンジニアリング処理（ラグ特徴量と統計量の計算）
 void createFeatures() {
   // 全ての時間ステップについて特徴量を計算
   for (int t = 0; t < time_steps; t++) {
     // 現在のインデックスを計算（循環バッファー内）
     int idx = (data_index - time_steps + t + time_steps) % time_steps;
     
     // 基本特徴量のコピー
     for (int f = 0; f < feature_count; f++) {
       input_tensor_data[t][f] = normalize(raw_sensor_data[idx][f], feature_min[f], feature_max[f]);
     }
     
     // ラグ特徴量の計算
     for (int lag = 1; lag <= lag_count && buffer_filled; lag++) {
       // lag分前のデータのインデックスを計算
       int lag_idx = (idx - lag + time_steps) % time_steps;
       
       for (int f = 0; f < feature_count; f++) {
         // ラグ特徴量を計算してinput_tensor_dataに格納
         int tensor_idx = feature_count + (lag - 1) * feature_count + f;
         input_tensor_data[t][tensor_idx] = normalize(raw_sensor_data[lag_idx][f], feature_min[f], feature_max[f]);
       }
     }
     
     // ローリング統計量の計算（オプション）
     if (use_rolling_stats && buffer_filled) {
       int rolling_offset = feature_count + lag_count * feature_count;
       
       for (int f = 0; f < feature_count; f++) {
         float mean = calculateRollingMean((float*)raw_sensor_data, rolling_window, f);
         float std_dev = calculateRollingStd((float*)raw_sensor_data, rolling_window, f, mean);
         float range = calculateRollingRange((float*)raw_sensor_data, rolling_window, f);
         
         // 計算した統計量をテンソルに追加
         input_tensor_data[t][rolling_offset + f * 3] = normalize(mean, feature_min[f], feature_max[f]);
         input_tensor_data[t][rolling_offset + f * 3 + 1] = std_dev / (feature_max[f] - feature_min[f]); // 標準偏差の正規化
         input_tensor_data[t][rolling_offset + f * 3 + 2] = range / (feature_max[f] - feature_min[f]);   // 範囲の正規化
       }
     }
   }
 }
 
 // モデルを使用した快適度予測関数
 float predict_comfort() {
   // データのバッファーが十分に溜まっていない場合は予測しない
   if (!buffer_filled) {
     return -1.0;
   }
   
   // 特徴量エンジニアリング処理を実行
   createFeatures();
   
   // CNNモデルのための入力テンソル形状変換 [batch_size=1, time_steps, features, channels=1]
   for (int t = 0; t < time_steps; t++) {
     for (int f = 0; f < input_dim; f++) {
       int tensor_idx = t * input_dim + f;
       tflInputTensor->data.f[tensor_idx] = input_tensor_data[t][f];
     }
   }
   
   // 推論を実行
   if (tflInterpreter->Invoke() != kTfLiteOk) {
     Serial.println("Invoke failed!");
     return -1.0;
   }
   
   // 出力を取得して非正規化
   float normalized_output = tflOutputTensor->data.f[0];
   float denormalized_output = denormalize(normalized_output, target_min, target_max);
   
   return denormalized_output;
 }
 
 void updateSensorData() {
   // センサーデータの更新
   if (sht3x.update() && qmp.update()) {
     temperature = sht3x.cTemp;
     humidity = sht3x.humidity;
     pressure = qmp.pressure / 100;
     WBGT = (temperature * 0.003289 + 0.01844) * humidity + (0.6868 * temperature - 2.022);
     THI = (temperature * 0.81 + humidity * 0.01 * (temperature * 0.99 - 14.3) + 46.3);
     
     // 生データを保存
     raw_sensor_data[data_index][0] = temperature;
     raw_sensor_data[data_index][1] = humidity;
     raw_sensor_data[data_index][2] = pressure;
     raw_sensor_data[data_index][3] = WBGT;
     
     // 次のインデックスに移動（循環バッファ）
     data_index = (data_index + 1) % time_steps;
     
     // データバッファが一度満杯になったらフラグを立てる
     if (data_index == 0) {
       buffer_filled = true;
     }
     
     // バッファが満杯になった後は、予測を実行
     if (buffer_filled) {
       predicted_comfort = predict_comfort();
     }
   }
 }
 
 // バージョン番号の保存と読み込み用の関数
 void saveModelVersion(uint32_t version) {
   File versionFile = SPIFFS.open("/model.version", "w");
   if(versionFile) {
     versionFile.write((uint8_t*)&version, sizeof(uint32_t));
     versionFile.close();
   }
 }
 
 uint32_t loadModelVersion() {
   uint32_t version = 1; // デフォルト値
   
   if(SPIFFS.exists("/model.version")) {
     File versionFile = SPIFFS.open("/model.version", "r");
     if(versionFile) {
       versionFile.read((uint8_t*)&version, sizeof(uint32_t));
       versionFile.close();
     }
   } else {
     // 初回時はファイルを作成
     saveModelVersion(version);
   }
   
   return version;
 }
 
 void displayData() {
   M5.Lcd.fillScreen(TFT_BLACK);
   M5.Lcd.setTextFont(2);
   M5.Lcd.setTextSize(1);
   
   // センサーデータ表示
   M5.Lcd.setCursor(20, 0);
   M5.Lcd.print("Temperature:");
   M5.Lcd.setCursor(200, 0);
   M5.Lcd.printf("%2.1f C", temperature);
   
   M5.Lcd.setCursor(20, 20);
   M5.Lcd.print("Humidity:");
   M5.Lcd.setCursor(200, 20);
   M5.Lcd.printf("%2.1f %%", humidity);
   
   M5.Lcd.setCursor(20, 40);
   M5.Lcd.print("Pressure:");
   M5.Lcd.setCursor(200, 40);
   M5.Lcd.printf("%2.0f hPa", pressure);
   
   M5.Lcd.setCursor(20, 60);
   M5.Lcd.print("WBGT:");
   M5.Lcd.setCursor(200, 60);
   M5.Lcd.printf("%2.1f", WBGT);
   
   M5.Lcd.setCursor(20, 80);
   M5.Lcd.print("Current THI:");
   M5.Lcd.setCursor(200, 80);
   M5.Lcd.printf("%2.1f", THI);
   
   // 予測値の表示（データが揃ったら）
   M5.Lcd.setCursor(20, 100);
   M5.Lcd.print("Predicted THI:");
   M5.Lcd.setCursor(200, 100);
   
   if (predicted_comfort > 0) {
     M5.Lcd.printf("%2.1f", predicted_comfort);
     
     // 快適度ステータスの表示
     M5.Lcd.setCursor(50, 120);
     if (predicted_comfort < 70) {
       M5.Lcd.setTextColor(TFT_GREEN, TFT_BLACK);
       M5.Lcd.print("Comfortable");
     } else if (predicted_comfort < 75) {
       M5.Lcd.setTextColor(TFT_YELLOW, TFT_BLACK);
       M5.Lcd.print("Slightly Uncomfortable");
     } else {
       M5.Lcd.setTextColor(TFT_RED, TFT_BLACK);
       M5.Lcd.print("Very Uncomfortable");
     }
     M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
   } else {
     M5.Lcd.print("Collecting data...");
   }
   
   // モデル情報の表示を追加
   M5.Lcd.setCursor(20, 140);
   M5.Lcd.print("Model: ");
   if(SPIFFS.exists("/model.tflite")) {
     File modelFile = SPIFFS.open("/model.tflite", "r");
     M5.Lcd.printf("SPIFFS v%d (%d KB)", modelVersion, modelFile.size() / 1024);
     modelFile.close();
   } else {
     M5.Lcd.print("Embedded");
   }
   
   // 特徴量情報の表示
   M5.Lcd.setCursor(20, 160);
   M5.Lcd.printf("Features: %d basic + %d lag", feature_count, lag_count);
   
   // Wi-Fi接続状態とIPアドレスの表示
   M5.Lcd.setCursor(20, 180);
   M5.Lcd.print("Network: ");
   if (WiFi.status() == WL_CONNECTED) {
     M5.Lcd.setTextColor(TFT_GREEN, TFT_BLACK);
     M5.Lcd.print("Connected");
     M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
     M5.Lcd.setCursor(20, 200);
     M5.Lcd.printf("http://%s", WiFi.localIP().toString().c_str());
   } else {
     M5.Lcd.setTextColor(TFT_RED, TFT_BLACK);
     M5.Lcd.print("Disconnected");
     M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
   }
 
   // メモリ使用量の表示を追加
   M5.Lcd.setCursor(20, 220);
   M5.Lcd.print("Memory: ");
   
   // 使用可能なヒープメモリと合計ヒープメモリの取得
   uint32_t freeHeap = ESP.getFreeHeap();
   uint32_t totalHeap = ESP.getHeapSize();
   uint32_t usedHeap = totalHeap - freeHeap;
   float heapPercentage = (float)usedHeap / totalHeap * 100;
   
   // メモリ使用量を表示
   M5.Lcd.printf("%d/%d KB (%2.1f%%)", usedHeap/1024, totalHeap/1024, heapPercentage);
 }
 
 void sendToAmbient() {
   // Ambientにデータを送信（WiFi接続時のみ）
   if (WiFi.status() == WL_CONNECTED) {
     ambient.set(1, temperature);
     ambient.set(2, humidity);
     ambient.set(3, pressure);
     ambient.set(4, WBGT);
     ambient.set(5, THI);
     
     // 予測値があれば、それも送信
     if (predicted_comfort > 0) {
       ambient.set(6, predicted_comfort);
     }
     
     ambient.send();
   }
 }
 
 void setup() {
   M5.begin(true, false, true, true);
   M5.Lcd.fillScreen(TFT_BLACK);
   M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
   M5.Lcd.setTextDatum(TL_DATUM);
   
   Serial.begin(115200);
   Wire.begin();
   
   // センサーの初期化
   if (!qmp.begin(&Wire, QMP6988_SLAVE_ADDRESS_L, 21, 22, 400000U)) {
     Serial.println("Couldn't find QMP6988");
     M5.Lcd.println("QMP6988 sensor error");
     while (1) delay(10);
   }
   
   if (!sht3x.begin(&Wire, SHT3X_I2C_ADDR, 21, 22, 400000U)) {
     Serial.println("Couldn't find SHT3X");
     M5.Lcd.println("SHT3X sensor error");
     while (1) delay(10);
   }
   
   // SPIFFSの初期化
   if(!SPIFFS.begin(true)) {
     Serial.println("SPIFFSのマウントに失敗しました");
     M5.Lcd.println("SPIFFS mount failed");
   } else {
     Serial.println("SPIFFSのマウントに成功しました");
     
     // モデルファイルの存在確認
     if(!SPIFFS.exists("/model.tflite")) {
       Serial.println("モデルファイルが見つかりません");
       M5.Lcd.println("Model file not found");
       // 初回起動時はmodel.hのデータを使用してファイルを作成
       createInitialModelFile();
 
       // バージョン1として保存
       modelVersion = 1;
       saveModelVersion(modelVersion);
 
     } else {
       Serial.println("モデルファイルを確認しました");
       // 既存のバージョン情報を読み込む
       modelVersion = loadModelVersion();
     }
   }
   
   // Wi-Fiの接続
   WiFi.begin(ssid, password);
   M5.Lcd.setCursor(0, 0);
   M5.Lcd.print("Connecting to WiFi");
   
   int retry_count = 0;
   while (WiFi.status() != WL_CONNECTED && retry_count < 20) {
     delay(500);
     M5.Lcd.print(".");
     retry_count++;
   }
   
   if (WiFi.status() == WL_CONNECTED) {
     M5.Lcd.println("\nWiFi connected");
     M5.Lcd.print("IP: ");
     M5.Lcd.println(WiFi.localIP());
     // Ambientの初期化
     ambient.begin(channelId, writeKey, &client);
     
     // Webサーバーのルート設定
     server.on("/", HTTP_GET, handleRoot);
     server.on("/upload", HTTP_GET, handleUploadForm);
     server.on("/update", HTTP_POST, handleUpdateComplete, handleUpdateUpload);
     
     // Webサーバーの開始
     server.begin();
     Serial.println("Web server started");
     M5.Lcd.println("Web server started");
   } else {
     M5.Lcd.println("\nWiFi connection failed");
     M5.Lcd.println("Continuing without web server");
   }
   
   // TensorFlow Liteモデルの初期化
   M5.Lcd.println("Initializing TF model...");
   
   // SPIFFSからモデルファイルを読み込む
   if(SPIFFS.exists("/model.tflite")) {
     File modelFile = SPIFFS.open("/model.tflite", "r");
     size_t modelSize = modelFile.size();
     
     // モデルデータ用のバッファを確保
     uint8_t* modelBuffer = (uint8_t*)malloc(modelSize);
     
     if(modelBuffer) {
       // ファイルからモデルデータを読み込む
       modelFile.read(modelBuffer, modelSize);
       modelFile.close();
       
       // TFLiteモデルの取得
       tflModel = tflite::GetModel(modelBuffer);
       if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
         M5.Lcd.println("Model schema mismatch!");
         
         // SPIFFSの.bakファイルから復元を試みる
         if(SPIFFS.exists("/model.tflite.bak")) {
           M5.Lcd.println("Trying backup file...");
           SPIFFS.rename("/model.tflite.bak", "/model.tflite");
           ESP.restart(); // 再起動して再試行
         }
         
         // それでもダメなら組み込みモデルで再作成
         M5.Lcd.println("Recreating from embedded model...");
         createInitialModelFile();
         ESP.restart(); // 再起動して再試行
       }
     } else {
       M5.Lcd.println("Failed to allocate model buffer");
       // メモリ確保失敗の場合、組み込みモデルを使用
       tflModel = tflite::GetModel(model);
     }
   } else {
     // モデルファイルがない場合、組み込みモデルを使用
     M5.Lcd.println("Using embedded model");
     tflModel = tflite::GetModel(model);
   }
   
   // インタープリターを作成
   tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, kTensorArenaSize, &tflErrorReporter);
   
   // テンソルのメモリ割り当て
   if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
     M5.Lcd.println("AllocateTensors() failed");
     while (1) delay(10);
   }
   
   // 入出力テンソルへのポインタを取得
   tflInputTensor = tflInterpreter->input(0);
   tflOutputTensor = tflInterpreter->output(0);
   
   // CNN入力テンソルの形状を確認
   Serial.println("TF Lite Input Tensor Info:");
   Serial.printf("Dimensions: %d\n", tflInputTensor->dims->size);
   for (int i = 0; i < tflInputTensor->dims->size; i++) {
     Serial.printf("Dim %d: %d\n", i, tflInputTensor->dims->data[i]);
   }
   Serial.printf("Type: %d\n", tflInputTensor->type);
   
   // センサーデータ配列を初期化
   for (int i = 0; i < time_steps; i++) {
     for (int j = 0; j < feature_count; j++) {
       raw_sensor_data[i][j] = 0.0;
     }
     for (int j = 0; j < input_dim; j++) {
       input_tensor_data[i][j] = 0.0;
     }

    }
  
    M5.Lcd.println("Setup complete!");
    delay(1000);
    M5.Lcd.fillScreen(TFT_BLACK);
  }
  
  void loop() {
    // Webサーバーのリクエスト処理
    if (WiFi.status() == WL_CONNECTED) {
      server.handleClient();
    } else {
      // Wi-Fi切断時は再接続を試みる（オプション）
      static unsigned long lastReconnectAttempt = 0;
      unsigned long currentMillis = millis();
      
      if (currentMillis - lastReconnectAttempt > 60000) { // 1分ごとに再接続を試行
        lastReconnectAttempt = currentMillis;
        WiFi.reconnect();
      }
    }
    
    // センサーデータの更新と表示
    updateSensorData();
    displayData();
    
    // Ambientへのデータ送信
    static unsigned long lastAmbientSend = 0;
    unsigned long currentMillis = millis();
    
    if (currentMillis - lastAmbientSend > 60000) { // 1分ごとにデータを送信
      lastAmbientSend = currentMillis;
      sendToAmbient();
    }
    
    // 短い遅延（Webサーバー応答性のため）
    for (int i = 0; i < 200; i++) {
      if (WiFi.status() == WL_CONNECTED) {
        server.handleClient();
      }
      delay(10);
    }
    
    // ボタン操作のチェック
    M5.update();
    
    // ボタンAを押すとモデル情報を表示
    if (M5.BtnA.wasPressed()) {
      // モデル情報の詳細表示
      M5.Lcd.fillScreen(TFT_BLACK);
      M5.Lcd.setCursor(0, 0);
      M5.Lcd.setTextFont(2);
      M5.Lcd.println("Model Information:");
      M5.Lcd.printf("Version: %d\n", modelVersion);
      
      if(SPIFFS.exists("/model.tflite")) {
        File modelFile = SPIFFS.open("/model.tflite", "r");
        M5.Lcd.printf("File Size: %d KB\n", modelFile.size() / 1024);
        modelFile.close();
      }
      
      M5.Lcd.printf("Input Dimensions: %d x %d\n", time_steps, input_dim);
      M5.Lcd.printf("Features: %d base + %d lag", feature_count, lag_count);
      if (use_rolling_stats) {
        M5.Lcd.printf(" + stats");
      }
      M5.Lcd.println();
      
      M5.Lcd.printf("Memory Usage: %d/%d KB\n", 
                   (ESP.getHeapSize() - ESP.getFreeHeap()) / 1024, 
                   ESP.getHeapSize() / 1024);
      
      M5.Lcd.println("\nPress any button to return");
      
      // ボタンが押されるまで待機
      while(!M5.BtnA.wasPressed() && !M5.BtnB.wasPressed() && !M5.BtnC.wasPressed()) {
        M5.update();
        delay(10);
      }
      
      // 画面をクリア
      M5.Lcd.fillScreen(TFT_BLACK);
    }
    
    // ボタンBを押すとデータ詳細を表示
    if (M5.BtnB.wasPressed()) {
      // センサーデータの詳細表示
      M5.Lcd.fillScreen(TFT_BLACK);
      M5.Lcd.setCursor(0, 0);
      M5.Lcd.setTextFont(2);
      M5.Lcd.println("Sensor Data History:");
      
      // 直近5つのデータを表示
      for (int i = 0; i < 5; i++) {
        int idx = (data_index - i - 1 + time_steps) % time_steps;
        M5.Lcd.printf("%d: T:%.1f H:%.1f P:%.1f W:%.1f\n", 
                     i, 
                     raw_sensor_data[idx][0], 
                     raw_sensor_data[idx][1], 
                     raw_sensor_data[idx][2], 
                     raw_sensor_data[idx][3]);
      }
      
      M5.Lcd.println("\nNormalized Feature Example:");
      if (buffer_filled) {
        // 最新の特徴量データを表示（一部のみ）
        createFeatures();
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 4; j++) {
            M5.Lcd.printf("%.2f ", input_tensor_data[0][i*4+j]);
          }
          M5.Lcd.println();
        }
      } else {
        M5.Lcd.println("Collecting data...");
      }
      
      M5.Lcd.println("\nPress any button to return");
      
      // ボタンが押されるまで待機
      while(!M5.BtnA.wasPressed() && !M5.BtnB.wasPressed() && !M5.BtnC.wasPressed()) {
        M5.update();
        delay(10);
      }
      
      // 画面をクリア
      M5.Lcd.fillScreen(TFT_BLACK);
    }
    
    // ボタンCを押すとWiFi再接続
    if (M5.BtnC.wasPressed()) {
      M5.Lcd.fillScreen(TFT_BLACK);
      M5.Lcd.setCursor(0, 0);
      M5.Lcd.println("Reconnecting WiFi...");
      
      // WiFi接続を再試行
      WiFi.disconnect();
      delay(1000);
      WiFi.begin(ssid, password);
      
      int retry_count = 0;
      while (WiFi.status() != WL_CONNECTED && retry_count < 20) {
        delay(500);
        M5.Lcd.print(".");
        retry_count++;
      }
      
      if (WiFi.status() == WL_CONNECTED) {
        M5.Lcd.println("\nWiFi reconnected!");
        M5.Lcd.printf("IP: %s\n", WiFi.localIP().toString().c_str());
      } else {
        M5.Lcd.println("\nReconnection failed!");
      }
      
      M5.Lcd.println("\nPress any button to return");
      
      // ボタンが押されるまで待機
      while(!M5.BtnA.wasPressed() && !M5.BtnB.wasPressed() && !M5.BtnC.wasPressed()) {
        M5.update();
        delay(10);
      }
      
      // 画面をクリア
      M5.Lcd.fillScreen(TFT_BLACK);
    }
    
    delay(10000);

  }