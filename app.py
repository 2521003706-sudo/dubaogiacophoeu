pip install streamlit yfinance pandas numpy matplotlib scikit-learn tensorflow
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import tensorflow as tf
import yfinance as yf

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Ứng dụng Dự báo Giá Cổ phiếu LSTM", page_icon="📈", layout="wide"
)

# --- PHẦN GIAO DIỆN HEADER: LOGO VÀ TÊN NGƯỜI THỰC HIỆN ---
col_logo, col_info = st.columns([1, 4])

with col_logo:
  # Bạn có thể thay đổi đường dẫn hoặc upload logo trực tiếp
  st.image(
      "https://via.placeholder.com/150",
      width=120,
      caption="Logo Đơn vị/Cá nhân",
  )

with col_info:
  st.title("📈 Ứng dụng Dự báo Giá Cổ phiếu LSTM")
  st.markdown("---")
  # THÔNG TIN NGƯỜI THỰC HIỆN (Có thể chỉnh sửa tên bên dưới)
  st.markdown(
      "**Họ và tên người thực hiện:** Nguyễn Văn A  \n**Mã sinh viên / Đơn vị:"
      "** Khoa Công nghệ Thông tin - Trường Đại học ABC  \n**Giảng viên hướng"
      " dẫn:** TS. Trần Văn B"
  )

st.markdown("---")

# --- SIDEBAR: CẤU HÌNH THAM SỐ ---
st.sidebar.header("⚙️ Tùy chỉnh tham số")
ticker = st.sidebar.text_input("Mã cổ phiếu (Yahoo Finance)", value="VNM.VN")
start_date = st.sidebar.date_input(
    "Ngày bắt đầu", pd.to_datetime("2018-01-01")
)
end_date = st.sidebar.date_input("Ngày kết thúc", pd.to_datetime("2026-10-14"))
epochs = st.sidebar.slider(
    "Số vòng huấn luyện (Epochs)", min_value=1, max_value=20, value=10
)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

run_button = st.sidebar.button("🚀 Chạy Huấn Luyện & Dự Báo")

if run_button:
  # Cố định Seed để ổn định kết quả
  seed = 42
  np.random.seed(seed)
  tf.random.set_seed(seed)
  random.seed(seed)

  with st.spinner(
      f"Đang tải dữ liệu và huấn luyện mô hình cho mã {ticker}..."
  ):
    # 1. Tải dữ liệu
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
      st.error(
          "Không tìm thấy dữ liệu cho mã cổ phiếu này. Vui lòng kiểm tra lại!"
      )
    else:
      # Xử lý MultiIndex nếu có từ yfinance phiên bản mới
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      data = df[["Close"]].values

      # Hiển thị dữ liệu gốc
      st.subheader(f"📊 Dữ liệu lịch sử giá cổ phiếu: {ticker}")
      st.line_chart(df["Close"])

      # Chia train/test
      train_size = int(0.8 * len(data))
      train_data = data[:train_size]
      test_data = data[train_size:]

      # Chuẩn hóa dữ liệu
      scaler = MinMaxScaler()
      train_scaled = scaler.fit_transform(train_data)
      test_scaled = scaler.transform(test_data)

      # Tạo chuỗi sequence
      def create_sequences(data, window_size=60):
        X, y = [], []
        for i in range(window_size, len(data)):
          X.append(data[i - window_size : i])
          y.append(data[i])
        return np.array(X), np.array(y)

      X_train, y_train = create_sequences(train_scaled)
      test_combined = np.vstack((train_scaled[-60:], test_scaled))
      X_test, y_test = create_sequences(test_combined)

      X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
      X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

      val_size = int(0.1 * len(X_train))
      X_val = X_train[-val_size:]
      y_val = y_train[-val_size:]
      X_train_final = X_train[:-val_size]
      y_train_final = y_train[:-val_size]

      # Xây dựng mô hình LSTM
      model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(
              50, return_sequences=True, input_shape=(60, 1)
          ),
          tf.keras.layers.LSTM(50),
          tf.keras.layers.Dense(1),
      ])

      model.compile(optimizer="adam", loss="mean_squared_error")

      early_stop = tf.keras.callbacks.EarlyStopping(
          monitor="val_loss", patience=5, restore_best_weights=True
      )

      # Huấn luyện mô hình
      history = model.fit(
          X_train_final,
          y_train_final,
          validation_data=(X_val, y_val),
          epochs=epochs,
          batch_size=batch_size,
          shuffle=False,
          callbacks=[early_stop],
          verbose=0,
      )

      # Dự báo trên tập Test
      test_pred = model.predict(X_test)
      test_pred = scaler.inverse_transform(test_pred)
      y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

      # Đánh giá mô hình
      rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
      mae = mean_absolute_error(y_test_real, test_pred)
      r2 = r2_score(y_test_real, test_pred)

      st.subheader("📉 Kết quả đánh giá mô hình trên tập Test")
      col1, col2, col3 = st.columns(3)
      col1.metric("RMSE", f"{rmse:.4f}")
      col2.metric("MAE", f"{mae:.4f}")
      col3.metric("R2 Score", f"{r2:.4f}")

      # Biểu đồ Loss
      st.subheader("📉 Biểu đồ Loss (Train vs Validation)")
      fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
      ax_loss.plot(history.history["loss"], label="Train Loss")
      ax_loss.plot(history.history["val_loss"], label="Validation Loss")
      ax_loss.set_title("Loss Train vs Validation")
      ax_loss.legend()
      st.pyplot(fig_loss)

      # Biểu đồ so sánh giá thật và dự báo Test
      st.subheader("📈 So sánh Giá Thật và Giá Dự Báo trên tập Test")
      fig_test, ax_test = plt.subplots(figsize=(12, 5))
      ax_test.plot(y_test_real, label="Giá thật", color="blue")
      ax_test.plot(test_pred, label="Dự báo test", color="orange")
      ax_test.legend()
      st.pyplot(fig_test)

      # Dự báo 10 ngày tới
      last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
      last_60_days = last_60_days.reshape(1, 60, 1)
      future_predictions = []

      for i in range(10):
        next_pred = model.predict(last_60_days)
        future_predictions.append(next_pred[0, 0])
        last_60_days = np.append(
            last_60_days[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1
        )

      future_predictions = np.array(future_predictions).reshape(-1, 1)
      future_predictions = scaler.inverse_transform(future_predictions)

      st.subheader("🔮 Kết quả dự báo 10 ngày tới")
      # Tạo DataFrame hiển thị 10 ngày tới
      future_dates = pd.date_range(
          start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq="B"
      )
      df_future = pd.DataFrame(
          {"Ngày": future_dates, "Giá dự báo": future_predictions.flatten()}
      )
      st.dataframe(df_future)

      # Biểu đồ tổng hợp dự báo 10 ngày tới
      st.subheader("📊 Biểu đồ Xu hướng Giá & 10 ngày tiếp theo")
      fig_future, ax_future = plt.subplots(figsize=(12, 6))
      ax_future.plot(y_test_real, label="Giá thật", color="blue")
      ax_future.plot(test_pred, label="Dự báo test", color="orange")
      future_x = range(len(y_test_real), len(y_test_real) + 10)
      ax_future.plot(
          future_x,
          future_predictions,
          marker="o",
          color="green",
          label="10 ngày tới",
      )
      ax_future.legend()
      st.pyplot(fig_future)
else:
  st.info(
      "👈 Vui lòng thiết lập các tham số ở thanh bên (Sidebar) và bấm nút **'Chạy"
      " Huấn Luyện & Dự Báo'** để bắt đầu."
      )
  
