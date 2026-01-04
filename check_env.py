import torch
import os

print(f"Đang chạy trên đường dẫn: {os.getcwd()}")
print(f"Cuda Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Sẵn sàng huấn luyện trên ổ D!")
else:
    print("Cảnh báo: Chưa nhận diện được GPU.")