# فاز ۱: راه‌اندازی پایه و زیرساخت
## Phase 1: Infrastructure Setup (Days 1-2)

## 🎯 اهداف این فاز
- ایجاد ساختار حرفه‌ای پروژه
- راه‌اندازی سیستم‌های لاگینگ و کانفیگ
- توسعه data pipeline قوی
- تست اولیه سیستم‌ها

## 📋 لیست کارهای فاز ۱

### روز ۱: راه‌اندازی ساختار پروژه

#### ✅ کارهای انجام شده:
1. **ساختار پوشه‌ها**: ایجاد کامل ساختار پروژه
2. **Requirements**: تعریف تمام dependencies مورد نیاز
3. **Configuration System**: سیستم JSON-based config
4. **Logging System**: سیستم پیشرفته لاگینگ با try-catch
5. **Data Infrastructure**: کلاس‌های dataset، preprocessing، و data loading

#### 🔧 مراحل نصب و راه‌اندازی:

```bash
# 1. ایجاد virtual environment
python -m venv floorplan_env
source floorplan_env/bin/activate  # MacOS/Linux
# floorplan_env\Scripts\activate    # Windows

# 2. نصب dependencies
pip install -r requirements.txt

# 3. تست سیستم لاگینگ
python config/logging_config.py

# 4. بررسی ساختار
ls -la
```

### روز ۲: راه‌اندازی داده‌ها و تست‌های اولیه

#### 📁 آماده‌سازی داده‌ها:
1. **داده‌های خام**: قرار دادن 2680 تصویر در `data/raw/`
2. **Labels**: قرار دادن فایل‌های label در `data/raw/labels/`
3. **تست data loading**: اجرای تست‌های اولیه

#### 🧪 تست‌های مرحله ۱:

```python
# تست 1: Configuration System
import json
with open('config/config.json', 'r') as f:
    config = json.load(f)
print("✅ Config loaded successfully")

# تست 2: Logging System  
from config.logging_config import get_logger
logger = get_logger("test")
logger.info("✅ Logging system working")

# تست 3: Data Pipeline (پس از اضافه کردن داده‌ها)
from data import DataManager
data_manager = DataManager(config)
data_manager.setup_datasets()
print("✅ Data pipeline working")
```

## 🎓 نکات یادگیری برای مصاحبه:

### 1. **سیستم Configuration**
- **سوال احتمالی**: چرا از JSON برای config استفاده کردید؟
- **پاسخ**: JSON ساده، قابل خواندن و به راحتی قابل تغییر است. برای production می‌توان از YAML یا حتی Hydra استفاده کرد.

### 2. **Logging Architecture**
- **سوال احتمالی**: چگونه error handling را پیاده‌سازی کردید؟
- **پاسخ**: استفاده از try-catch در تمام قسمت‌ها + logging سطح‌بندی شده + file handlers جداگانه برای errors

### 3. **Data Pipeline Design**
- **سوال احتمالی**: چطور scalability را در نظر گرفتید؟
- **پاسخ**: استفاده از DataLoader با num_workers، lazy loading، و مدیریت memory بهینه

## 🔄 آماده‌سازی برای فاز ۲:

### پیش‌نیازها:
- [ ] تست موفق تمام سیستم‌های فاز ۱
- [ ] آماده بودن dataset با حداقل ۱۰۰ sample
- [ ] اجرای موفق data loading pipeline

### نکات مهم:
1. **Git Commits**: هر روز چندین commit با پیام‌های توصیفی
2. **Documentation**: نوشتن docstring برای تمام functions
3. **Testing**: تست manual تمام components

## 📊 معیارهای موفقیت فاز ۱:
- ✅ ساختار کامل پروژه
- ✅ سیستم config و logging فعال
- ✅ Data pipeline بدون خطا
- ✅ حداقل ۵ commit با پیام‌های مناسب
- ✅ Documentation اولیه

---

## 🚀 آماده برای فاز ۲: توسعه مدل پایه

در فاز بعدی:
- ایجاد architecture های مدل
- پیاده‌سازی training pipeline
- ایجاد evaluation metrics
- شروع testing framework

**نکته**: در هر مرحله، تمرکز بر quality over quantity و آمادگی برای توضیح design decisions در مصاحبه! 