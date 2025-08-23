import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 配置
csv_path = "/tmp/uwb_logs/uwb_distance_series.csv"  # 替换成你的路径
output_path = "uwb_distances_scatter.png"

# 读取数据
df = pd.read_csv(csv_path)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 提取距离列（d0 到 dN）
distance_cols = [col for col in df.columns if col.startswith("d")]

# 绘制
plt.figure(figsize=(20, 8))

for col in distance_cols:
    plt.plot(df['Timestamp'], df[col], marker='o', linestyle='None', label=col, markersize=3, alpha=0.7)

# 时间格式美化
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()

# 样式设置
plt.title("UWB Raw Distance Points (d0 - d15)")
plt.xlabel("Timestamp")
plt.ylabel("Distance (m)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(ncol=4, fontsize='small', loc='upper right')
plt.tight_layout()

# 保存 & 展示
plt.savefig(output_path, dpi=200)
plt.show()
