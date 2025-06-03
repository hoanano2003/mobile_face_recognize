# import tensorflow as tf
# print("Available devices:")
# for device in tf.config.list_physical_devices():
#     print(device)


import psutil

print("Số lõi CPU vật lý:", psutil.cpu_count(logical=False))
print("Số lõi CPU logic (bao gồm cả hyper-threading):", psutil.cpu_count(logical=True))
print("Phần trăm sử dụng CPU tổng thể:", psutil.cpu_percent(interval=1))
mem = psutil.virtual_memory()
print(f"Tổng bộ nhớ: {mem.total / (1024**3):.2f} GB")
print(f"Bộ nhớ còn trống: {mem.available / (1024**3):.2f} GB")
print(f"Tỉ lệ sử dụng bộ nhớ: {mem.percent}%")
for proc in psutil.process_iter(['pid', 'name', 'username']):
    print(proc.info)