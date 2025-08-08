# 🚀 راهنمای شروع سریع
# Quick Start Guide

## ⚡ شروع فوری (۵ دقیقه)

### ۱. Clone و Setup
```bash
# اگر از GitHub کلون می‌کنید
git clone [your-repo-url]
cd floorplan_detection

# اگر locally شروع می‌کنید (مثل الان)
python setup_project.py
```

### ۲. Virtual Environment
```bash
# ایجاد virtual environment
python -m venv floorplan_env

# فعال‌سازی
# macOS/Linux:
source floorplan_env/bin/activate

# Windows:
floorplan_env\Scripts\activate
```

### ۳. نصب Dependencies
```bash
pip install -r requirements.txt
```

### ۴. تست سیستم
```bash
# تست configuration
python -c "import json; print('✅ Config OK' if json.load(open('config/config.json')) else '❌')"

# تست logging
python config/logging_config.py

# تست unittest ها
python -m pytest tests/ -v
```

---

## 📁 آماده‌سازی داده‌ها

### ساختار مورد انتظار:
```
data/
├── raw/
│   ├── image1.jpg        # 2680 floorplan images
│   ├── image2.png
│   └── ...
└── raw/labels/
    ├── image1.txt        # Corresponding labels
    ├── image2.txt        # Format: class_id x_center y_center width height
    └── ...
```

### فرمت Label Files:
```
# هر خط: class_id x_center y_center width height
# مثال (coordinates نرمال شده بین 0-1):
0 0.5 0.3 0.2 0.1    # room at center-left
1 0.8 0.2 0.05 0.15  # window at top-right  
2 0.2 0.9 0.1 0.05   # door at bottom-left
```

### تست Data Pipeline:
```python
# پس از اضافه کردن داده‌ها
python -c "
from data import DataManager
import json
config = json.load(open('config/config.json'))
dm = DataManager(config)
dm.setup_datasets()
stats = dm.get_dataset_stats()
print('✅ Data pipeline working!')
print(f'Total images: {stats[\"overall\"][\"total_images\"]}')
print(f'Classes: {list(stats[\"overall\"][\"class_counts\"].keys())}')
"
```

---

## 🛠 مراحل توسعه (روزانه)

### امروز (روز ۱):
- [x] ✅ ساختار پروژه
- [x] ✅ Configuration system  
- [x] ✅ Logging system
- [x] ✅ Data infrastructure
- [ ] 📁 اضافه کردن dataset
- [ ] 🧪 تست data pipeline

### فردا (روز ۲):
- [ ] 🏗️ مدل base architecture
- [ ] 🎯 YOLOv8 implementation
- [ ] 🔄 Training pipeline
- [ ] 📊 Basic metrics

---

## 🚨 مشکلات رایج و راه‌حل

### ۱. Import Error
```bash
# مشکل: ModuleNotFoundError
# راه‌حل: اضافه کردن PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# یا اجرا از root directory
python -m data.dataset  # instead of cd data && python dataset.py
```

### ۲. GPU Problems
```python
# بررسی GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# اگر GPU نیست، config را تنظیم کنید:
# در config.json: "device": "cpu"
```

### ۳. Memory Issues
```python
# کاهش batch size در config.json
{
  "training": {
    "batch_size": 8,  # کاهش از 16
    "num_workers": 2  # کاهش workers
  }
}
```

---

## 📋 Checklist روز ۱

### Technical:
- [ ] ✅ Virtual environment created
- [ ] ✅ Dependencies installed  
- [ ] ✅ Configuration validated
- [ ] ✅ Logging system tested
- [ ] 📁 Dataset added (when available)
- [ ] 🧪 Data pipeline tested

### Git & Documentation:
- [ ] 📝 Git repo initialized
- [ ] 💾 Initial commit
- [ ] 📚 README.md reviewed
- [ ] 🎯 Phase1_Setup_Guide.md studied

### Learning:
- [ ] 🧠 Understanding project structure
- [ ] 💡 Learning configuration system
- [ ] 🔍 Understanding data pipeline
- [ ] 📖 Reading interview preparation notes

---

## 🔄 آماده برای مرحله بعد

### Prerequisites for Phase 2:
1. ✅ All Phase 1 systems working
2. 📁 Dataset available (even 100 samples for testing)
3. 🧪 Successful data loading test
4. 💾 Clean git history with descriptive commits

### اگر مشکلی پیش آمد:
1. 🔍 Check logs در `logs/` directory
2. 🧪 Run individual tests: `python tests/test_config.py`
3. 📞 Review error messages carefully
4. 🔄 Try `python setup_project.py` again

---

## 🎯 هدف روز ۱
**یک foundation قوی برای development بعدی + درک عمیق از architecture decisions برای مصاحبه**

✨ **Success Metric**: توانایی توضیح دلیل هر architectural choice در مصاحبه! 