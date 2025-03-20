# 🤖 Crypto Trading Bot

บอทเทรดคริปโตอัตโนมัติที่ใช้ Machine Learning ในการทำนายทิศทางราคา

## 📋 คุณสมบัติ

- ใช้ XGBoost ในการทำนายทิศทางราคา
- เทรนหลายโมเดลและใช้ Majority Voting
- คำนวณ Technical Indicators หลากหลายตัว
- บันทึกประวัติการเทรดและสรุปผลการเทรด
- ปรับแต่งค่าต่างๆ ได้ผ่านไฟล์ config

## 🚀 การติดตั้ง

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. สร้างไฟล์ `config.json` และใส่ API Key ของ Binance:
```json
{
    "api": {
        "key": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET"
    },
    ...
}
```

## 📝 การใช้งาน

1. รันบอท:
```bash
python src/trading_bot.py
```

2. กด Ctrl+C เพื่อหยุดการทำงาน

## ⚙️ การตั้งค่า

ปรับแต่งค่าต่างๆ ได้ในไฟล์ `config.json`:

- `trading.symbol`: คู่เหรียญที่จะเทรด (เช่น "BTCUSDT")
- `trading.interval`: ช่วงเวลาในการเทรด (เช่น "1m", "5m", "1h")
- `trading.quantity`: ปริมาณต่อการเทรด
- `trading.confidence_threshold`: ความมั่นใจขั้นต่ำในการเทรด (0-1)
- `model.num_models`: จำนวนโมเดลที่ใช้
- `model.force_train`: บังคับให้เทรนโมเดลใหม่ทุกครั้ง

## 📊 โครงสร้างโปรเจค

```
├── src/
│   ├── config/         # การตั้งค่าต่างๆ
│   ├── models/         # โมเดล ML
│   ├── services/       # บริการต่างๆ
│   ├── utils/          # ฟังก์ชันช่วยเหลือ
│   └── trading_bot.py  # ไฟล์หลัก
├── config.json         # ไฟล์ config
└── requirements.txt    # dependencies
```

## ⚠️ คำเตือน

การเทรดคริปโตมีความเสี่ยงสูง อาจสูญเสียเงินลงทุนได้ ควรศึกษาให้ดีก่อนใช้งาน

## Features ที่ใช้ในการทำนาย
- ราคา OHLCV
- RSI
- MACD
- Bollinger Bands
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)

## การปรับแต่ง
- ปรับ `confidence_threshold` ใน `execute_trade()` เพื่อควบคุมความเสี่ยง
- เพิ่มหรือลด features ใน `prepare_features()`
- ปรับพารามิเตอร์ของ XGBoost ใน `train_model()` 