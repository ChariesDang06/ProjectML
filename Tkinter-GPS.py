import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

# Hàm để khởi tạo Kalman Filter
def initialize_kalman_filter(initial_latitude, initial_longitude):
    F = np.array([[1, 0, 1, 0],  # latitude dựa trên velocity_lat
                  [0, 1, 0, 1],  # longitude dựa trên velocity_lon
                  [0, 0, 1, 0],  # velocity_lat không thay đổi
                  [0, 0, 0, 1]]) # velocity_lon không thay đổi

    Q = np.eye(4) * 0.01  # Nhiễu hệ thống (có thể điều chỉnh)
    H = np.array([[1, 0, 0, 0],  # latitude được đo trực tiếp
                  [0, 1, 0, 0]]) # longitude được đo trực tiếp
    R = np.eye(2) * 0.1  # Nhiễu đo lường (có thể điều chỉnh)
    x0 = np.array([initial_latitude, initial_longitude, 0, 0])  # Trạng thái ban đầu
    P0 = np.eye(4) * 1.0  # Hiệp phương sai ban đầu

    return F, H, Q, R, x0, P0

# Hàm để dự đoán và cập nhật Kalman Filter
def kalman_filter_step(F, H, Q, R, P, x, z):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_updated = x_pred + K @ y
    P_updated = (np.eye(len(P)) - K @ H) @ P_pred
    return x_updated, P_updated

# Hàm để xử lý một tệp CSV
def process_csv_file(file_path):
    data = pd.read_csv(file_path)
    
    if 'Time' in data.columns:
        time_data = data['Time'].values  # Lưu trữ cột thời gian
    else:
        time_data = np.arange(len(data))  # Nếu không có, tạo một dãy số thay thế

    gps_measurements = data[['Latitude', 'Longitude']].values
    
    F, H, Q, R, x, P = initialize_kalman_filter(gps_measurements[0, 0], gps_measurements[0, 1])

    filtered_positions = []
    for z in gps_measurements:
        x, P = kalman_filter_step(F, H, Q, R, P, x, z)
        filtered_positions.append(x[:2])

    filtered_positions = np.array(filtered_positions)
    output = pd.DataFrame(filtered_positions, columns=['Latitude', 'Longitude'])
    output['Time'] = time_data
    return data, output

# Hàm để vẽ biểu đồ so sánh giữa dữ liệu gốc và dữ liệu đã lọc
def plot_comparison(raw_data, filtered_data):
    fig, ax = plt.subplots()
    ax.scatter(raw_data['Latitude'], raw_data['Longitude'], label='Raw GPS Data', c='red')
    ax.scatter(filtered_data['Latitude'], filtered_data['Longitude'], label='Filtered GPS Data', c='blue')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_title('GPS Data: Raw vs. Kalman Filtered')
    ax.legend()
    return fig

# Hàm để chọn và xử lý tệp CSV
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        raw_data, filtered_data = process_csv_file(file_path)
        fig = plot_comparison(raw_data, filtered_data)

        # Xóa canvas cũ nếu có
        for widget in frame.winfo_children():
            widget.destroy()

        # Hiển thị biểu đồ lên giao diện Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("GPS Data Kalman Filter")
root.geometry("800x600")

# Tạo nút chọn file và khung để hiển thị biểu đồ
button = tk.Button(root, text="Open Text File", command=open_file)
button.pack(side=tk.TOP, pady=10)

# Tạo nút để đóng ứng dụng
button_close = tk.Button(root, text="Close", command=root.quit)
button_close.pack(side=tk.TOP, pady=10)

frame = tk.Frame(root)
frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Bắt đầu vòng lặp chính của ứng dụng Tkinter
root.mainloop()
