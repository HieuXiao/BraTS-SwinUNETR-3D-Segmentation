import os
import glob
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, CacheDataset, Dataset
from src.transforms import get_train_transforms, get_val_transforms

class BraTSDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 1, 
        num_workers: int = 4,
        cache_rate: float = 1.0
    ):
        """
        Quản lý dữ liệu BraTS.
        args:
            data_dir: Đường dẫn đến thư mục chứa dữ liệu BraTS (đã giải nén).
            batch_size: Kích thước lô (với GPU 4GB, nên để là 1).
            num_workers: Số luồng CPU để tải dữ liệu (thường set bằng số nhân CPU).
            cache_rate: Tỷ lệ dữ liệu lưu vào RAM (1.0 là lưu hết). Giúp train nhanh hơn.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        
        self.train_files = []
        self.val_files = []

    def setup(self, stage=None):
        """
        Bước quan trọng: Tìm file, ghép 4 modality lại và chia tập Train/Val.
        """
        # Tìm tất cả các thư mục con (mỗi thư mục là 1 bệnh nhân)
        # Cấu trúc BraTS thường là: Dataset/BraTS-GLI-XXXXX/
        search_path = os.path.join(self.data_dir, "*")
        patient_dirs = sorted(glob.glob(search_path))
        
        # Lọc chỉ lấy các thư mục (tránh lấy nhầm file lạ)
        patient_dirs = [d for d in patient_dirs if os.path.isdir(d)]
        
        if len(patient_dirs) == 0:
            raise FileNotFoundError(f"Không tìm thấy dữ liệu nào trong {self.data_dir}")

        data_list = []
        for pat_path in patient_dirs:
            # Lấy ID bệnh nhân từ tên thư mục (ví dụ: BraTS-GLI-00001)
            pat_id = os.path.basename(pat_path)
            
            # Định nghĩa tên file chuẩn theo format BraTS 2023/2024
            # [cite_start]Cần load 4 modalities: T1, T1ce, T2, FLAIR [cite: 16, 17, 18, 19, 20]
            t1 = os.path.join(pat_path, f"{pat_id}-t1n.nii.gz")
            t1ce = os.path.join(pat_path, f"{pat_id}-t1c.nii.gz")
            t2 = os.path.join(pat_path, f"{pat_id}-t2w.nii.gz")
            flair = os.path.join(pat_path, f"{pat_id}-t2f.nii.gz")
            seg = os.path.join(pat_path, f"{pat_id}-seg.nii.gz")

            # Kiểm tra xem file có tồn tại không
            if os.path.exists(t1) and os.path.exists(seg):
                data_list.append({
                    "image": [t1, t1ce, t2, flair], # MONAI sẽ tự stack 4 file này thành 1 ảnh 4 kênh
                    "label": seg
                })
        
        print(f"--> Tìm thấy {len(data_list)} bệnh nhân hợp lệ.")

        # Chia tập dữ liệu: 80% Train - 20% Val
        train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)
        self.train_files = train_list
        self.val_files = val_list
        print(f"--> Train set: {len(self.train_files)} | Val set: {len(self.val_files)}")

    def train_dataloader(self):
        """
        Tạo DataLoader cho huấn luyện.
        Sử dụng CacheDataset để nạp dữ liệu vào RAM, giúp training nhanh hơn.
        """
        # Sử dụng transform từ src/transforms.py
        # [cite_start]Giảm crop size xuống 64x64x64 để phù hợp GPU 4GB [cite: 75, 90]
        train_transforms = get_train_transforms(roi_size=(64, 64, 64))
        
        # Nếu RAM thấp (<32GB), hãy đổi CacheDataset thành Dataset thường
        # Hoặc giảm cache_rate xuống (ví dụ 0.5)
        train_ds = CacheDataset(
            data=self.train_files, 
            transform=train_transforms, 
            cache_rate=self.cache_rate, 
            num_workers=self.num_workers
        )
        
        return DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True # Giúp chuyển dữ liệu sang GPU nhanh hơn
        )

    def val_dataloader(self):
        """
        Tạo DataLoader cho kiểm thử.
        """
        val_transforms = get_val_transforms()
        
        val_ds = CacheDataset(
            data=self.val_files, 
            transform=val_transforms, 
            cache_rate=self.cache_rate, 
            num_workers=self.num_workers
        )
        
        return DataLoader(
            val_ds, 
            batch_size=1, # Validation luôn để batch_size=1 để đánh giá từng ca
            shuffle=False, 
            num_workers=self.num_workers
        )