# PEP Preprocessing Pipeline

Pipeline tiền xử lý ảnh gồm 3 bước:

1. Global deskew: căn chỉnh ảnh bị nghiêng.
2. Crop: cắt vùng ảnh cần thiết.
3. Main preprocess: chuẩn hóa nền, tăng tương phản và làm nổi bật chi tiết.

## Yêu cầu

- Windows
- Python 3.10 trở lên
- Git

## Cài đặt

Clone repository:

```powershell
git clone <URL_REPOSITORY>
cd mocbanorg
```

Tạo và kích hoạt môi trường ảo:

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Cài thư viện:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Chạy toàn bộ pipeline với cấu hình mặc định:

```
python preprocess.py
```

Tùy chọn dòng lệnh:

```
python preprocess.py `
  --input datasets `
  --output results `
  --csv results/processing_results.csv `
  --mode full `
  --step 1 1 1
```
