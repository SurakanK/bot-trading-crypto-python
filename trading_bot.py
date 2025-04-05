import os
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ta
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
import joblib
import pathlib
import json
from scipy import stats
import pickle
import warnings

# โหลด environment variables และ config
load_dotenv()

def load_config():
    """โหลดการตั้งค่าจากไฟล์ config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด config.json: {str(e)}")
        return None

CONFIG = load_config()

class TradingBot:
    def __init__(self):
        if CONFIG is None:
            raise Exception("❌ ไม่สามารถเริ่มต้นบอทได้เนื่องจากไม่มีไฟล์ config.json")
            
        self.symbol = CONFIG['trading']['symbol']
        self.interval = CONFIG['trading']['interval']
        self.use_testnet = CONFIG['trading']['use_testnet']
        self.num_models = CONFIG['model']['num_models']
        
        # กำหนดค่าคงที่สำหรับการเทรด
        self.SIDE_BUY = SIDE_BUY
        self.SIDE_SELL = SIDE_SELL
        self.ORDER_TYPE_MARKET = ORDER_TYPE_MARKET
        
        # เพิ่มการติดตาม crypto balance
        self.base_currency = self.symbol[:-4]  # เช่น BTC จาก BTCUSDT
        self.quote_currency = self.symbol[-4:]  # เช่น USDT จาก BTCUSDT
        self.base_balance = 0  # จำนวน crypto ที่ถือครอง
        
        self.models_dir = pathlib.Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # เก็บโมเดลตามจำนวนที่กำหนดใน config
        self.models = []
        self.scalers = []
        for i in range(self.num_models):
            model_path = self.models_dir / f'xgboost_{self.symbol}_{self.interval}_{i}.model'
            scaler_path = self.models_dir / f'scaler_{self.symbol}_{self.interval}_{i}.pkl'
            self.models.append(None)
            self.scalers.append(None)
            try:
                if model_path.exists() and scaler_path.exists():
                    model = xgb.XGBClassifier()
                    model.load_model(str(model_path))
                    scaler = joblib.load(scaler_path)
                    self.models[i] = model
                    self.scalers[i] = scaler
            except Exception as e:
                print(f"ℹ️ ไม่พบโมเดลที่ {i} หรือเกิดข้อผิดพลาด: {str(e)}")
        
        self.trade_history_path = self.models_dir / f'trade_history_{self.symbol}_{self.interval}.json'
        
        # เก็บข้อมูลการเทรด
        self.initial_balance = 0
        self.current_balance = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit_loss = 0
        self.trade_history = []
        
        print(f"\n{'='*50}")
        print(f"เริ่มต้นบอทเทรด {self.symbol} ที่ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Timeframe: {self.interval}")
        print(f"{'='*50}\n")
        
        if self.use_testnet:
            self.client = Client(
                os.getenv('BINANCE_API_KEY'),
                os.getenv('BINANCE_API_SECRET'),
                testnet=True
            )
            print("🔧 กำลังใช้งาน Binance Testnet")
            # แสดงข้อมูลบัญชี testnet
            account = self.client.get_account()
            print("\n📊 ข้อมูลบัญชี Testnet:")
            
            # หาเงินเริ่มต้นในบัญชี
            quote_currency = self.symbol[len(self.symbol)-4:]  # เช่น USDT จาก BTCUSDT
            for asset in account['balances']:
                if float(asset['free']) > 0 or float(asset['locked']) > 0:
                    print(f"💰 สินทรัพย์: {asset['asset']}")
                    print(f"   จำนวนที่ใช้ได้: {asset['free']}")
                    print(f"   จำนวนที่ล็อค: {asset['locked']}\n")
                    if asset['asset'] == quote_currency:
                        self.initial_balance = float(asset['free']) + float(asset['locked'])
                        self.current_balance = self.initial_balance
            
            print(f"\n💵 เงินเริ่มต้น: {self.initial_balance:.2f} {quote_currency}")
        else:
            self.client = Client(
                os.getenv('BINANCE_API_KEY'),
                os.getenv('BINANCE_API_SECRET')
            )
            print("⚠️ กำลังใช้งาน Binance Production API - โปรดระวัง!")
        
        # เคลียร์ประวัติการเทรดเก่า
        self.clear_trade_history()
        
        self.load_trade_history()
        
    def get_historical_data(self):
        """ดึงข้อมูลราคาย้อนหลัง"""
        lookback_days = CONFIG['model']['lookback_days']
        print(f"\n📈 กำลังดึงข้อมูลราคาย้อนหลัง {lookback_days} วัน...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        klines = self.client.get_historical_klines(
            self.symbol,
            self.interval,
            start_time.strftime("%d %b %Y %H:%M:%S"),
            end_time.strftime("%d %b %Y %H:%M:%S")
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # แปลงข้อมูลให้เป็นตัวเลข
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ ดึงข้อมูลสำเร็จ: {len(df)} แท่งเทียน")
        return df
    
    def add_technical_indicators(self, df):
        """เพิ่ม technical indicators"""
        print("\n🔄 กำลังคำนวณ Technical Indicators...")
        
        # RSI หลายช่วงเวลา
        print("   ➕ RSI")
        df['RSI_1'] = ta.momentum.RSIIndicator(df['close'], window=1).rsi()
        df['RSI_3'] = ta.momentum.RSIIndicator(df['close'], window=3).rsi()
        
        # MACD
        print("   ➕ MACD")
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        # Bollinger Bands
        print("   ➕ Bollinger Bands")
        bollinger = ta.volatility.BollingerBands(df['close'], window=20)  # เปลี่ยนเป็น window=20 แท่งเทียน
        df['BB_high_1'] = bollinger.bollinger_hband()
        df['BB_low_1'] = bollinger.bollinger_lband()
        df['BB_mid_1'] = bollinger.bollinger_mavg()
        df['BB_width_1'] = (df['BB_high_1'] - df['BB_low_1']) / df['BB_mid_1']
        
        # Moving Averages ระยะสั้น
        print("   ➕ Moving Averages")
        df['SMA_1'] = ta.trend.sma_indicator(df['close'], window=1)
        df['SMA_3'] = ta.trend.sma_indicator(df['close'], window=3)
        df['EMA_1'] = ta.trend.ema_indicator(df['close'], window=1)
        df['EMA_3'] = ta.trend.ema_indicator(df['close'], window=3)
        
        # Stochastic Oscillator
        print("   ➕ Stochastic")
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=3, smooth_window=2)
        df['STOCH_k'] = stoch.stoch()
        df['STOCH_d'] = stoch.stoch_signal()
        
        # ATR (Average True Range)
        print("   ➕ ATR")
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # ROC (Rate of Change)
        print("   ➕ ROC")
        df['ROC'] = ta.momentum.ROCIndicator(df['close'], window=1).roc()

        # Volume Indicators
        print("   ➕ Volume Indicators")
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

        # VWAP (Volume Weighted Average Price)
        print("   ➕ VWAP")
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Bid-Ask Spread (จำลอง)
        df['Bid_Ask_Spread'] = df['high'] - df['low']

        # Order Book Imbalance (จำลองจากปริมาณการซื้อขาย)
        df['Order_Book_Imbalance'] = (df['volume'] - df['volume'].rolling(window=3, min_periods=1).mean()) / df['volume'].rolling(window=3,min_periods=1).std()

        # Time of Day (0-23)
        df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.hour

        print("✅ คำนวณ Technical Indicators เสร็จสิ้น")
        return df
    
    def prepare_features(self, df):
        """เตรียม features สำหรับโมเดล"""
        print("\n🔧 กำลังเตรียมข้อมูลสำหรับโมเดล...")
        
        # เพิ่ม technical indicators
        df = self.add_technical_indicators(df)
        
        # สร้าง target variable (1 ถ้าราคาขึ้น, 0 ถ้าราคาลง)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # เลือก features ที่จะใช้
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'RSI_1', 'RSI_3',  # ลด RSI_21 เพราะช้าเกินไป เพิ่ม RSI_10 เพื่อตอบสนองเร็วขึ้น
            'MACD', 'MACD_signal', 'MACD_diff',
            'BB_high_1', 'BB_low_1', 'BB_mid_1', 'BB_width_1',  # ลดจาก 20 period เหลือ 3
            'SMA_1', 'SMA_3', 'EMA_1', 'EMA_3',  # แทนที่ SMA_20/50, EMA_20/50 ด้วยค่าระยะสั้น
            'STOCH_k', 'STOCH_d',
            'ATR', 'ROC', 'OBV',
            'VWAP',  # ค่าเฉลี่ยน้ำหนักตามปริมาณการซื้อขาย
            'Bid_Ask_Spread',  # วัดความลึกของตลาด
            'Order_Book_Imbalance',  # เปรียบเทียบออเดอร์ฝั่งซื้อกับฝั่งขาย
            'time_of_day'  # ระบุช่วงเวลาของวัน เช่น เช้า เที่ยง บ่าย ดึก
        ]
        
        # ลบแถวที่มีค่า NaN
        df = df.dropna()
        
        X = df[feature_columns]
        y = df['target']
        
        print(f"✅ เตรียมข้อมูลเสร็จสิ้น: {len(X)} ตัวอย่าง, {len(feature_columns)} features")
        return X, y
    
    def save_model(self):
        """บันทึกโมเดลและ scaler ลงไฟล์"""
        for i in range(self.num_models):
            if self.models[i] is not None and self.scalers[i] is not None:
                model_path = self.models_dir / f'xgboost_{self.symbol}_{self.interval}_{i}.model'
                scaler_path = self.models_dir / f'scaler_{self.symbol}_{self.interval}_{i}.pkl'
                self.models[i].save_model(str(model_path))
                joblib.dump(self.scalers[i], scaler_path)
        print("✅ บันทึกโมเดลทั้งหมดเรียบร้อย")
    
    def load_model(self):
        """โหลดโมเดลและ scaler จากไฟล์"""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                print("\n📂 กำลังโหลดโมเดลที่บันทึกไว้...")
                self.model = xgb.XGBClassifier()
                self.model.load_model(str(self.model_path))
                self.scaler = joblib.load(self.scaler_path)
                print("✅ โหลดโมเดลสำเร็จ")
                return True
        except Exception as e:
            print(f"ℹ️ ไม่พบโมเดลที่บันทึกไว้ หรือเกิดข้อผิดพลาด: {str(e)}")
        return False

    def save_trade_history(self):
        """บันทึกประวัติการเทรด"""
        trade_data = {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_profit_loss': self.total_profit_loss,
            'trade_history': self.trade_history
        }
        
        with open(self.trade_history_path, 'w') as f:
            json.dump(trade_data, f, indent=4)
    
    def load_trade_history(self):
        """โหลดประวัติการเทรด"""
        try:
            if self.trade_history_path.exists():
                with open(self.trade_history_path, 'r') as f:
                    trade_data = json.load(f)
                    self.initial_balance = trade_data['initial_balance']
                    self.current_balance = trade_data['current_balance']
                    self.total_trades = trade_data['total_trades']
                    self.winning_trades = trade_data['winning_trades']
                    self.total_profit_loss = trade_data['total_profit_loss']
                    self.trade_history = trade_data['trade_history']
                print("\n📈 โหลดประวัติการเทรดสำเร็จ")
                self.print_trading_summary()
        except Exception as e:
            print(f"ℹ️ ไม่พบประวัติการเทรด หรือเกิดข้อผิดพลาด: {str(e)}")
    
    def print_trading_summary(self):
        """แสดงสรุปผลการเทรด"""
        quote_currency = self.symbol[len(self.symbol)-4:]
        
        # นับจำนวนการเทรดจาก trade_history
        buy_trades = len([t for t in self.trade_history if t['type'] == 'BUY'])
        sell_trades = len([t for t in self.trade_history if t['type'] == 'SELL'])
        self.total_trades = min(buy_trades, sell_trades)  # นับเป็นคู่ (ซื้อ-ขาย)
        
        # คำนวณ win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        losing_trades = self.total_trades - self.winning_trades
        
        print("\n📊 สรุปผลการเทรด:")
        print(f"   💰 เงินเริ่มต้น: {self.initial_balance:.2f} {quote_currency}")
        print(f"   💵 เงินปัจจุบัน: {self.current_balance:.2f} {quote_currency}")
        print(f"   📈 กำไร/ขาดทุนรวม: {self.total_profit_loss:.2f} {quote_currency} ({((self.current_balance/self.initial_balance)-1)*100:.2f}%)")
        print(f"   🎯 จำนวนเทรดทั้งหมด: {buy_trades} ซื้อ, {sell_trades} ขาย")
        print(f"   ✅ เทรดกำไร: {self.winning_trades} ครั้ง")
        print(f"   ❌ เทรดขาดทุน: {losing_trades} ครั้ง")
        print(f"   📊 Win Rate: {win_rate:.2f}%")

    def train_model(self, X, y, force_train=False):
        """เทรนโมเดล XGBoost ตามจำนวนที่กำหนด"""
        if all(model is not None for model in self.models) and not force_train:
            print("\n✅ ใช้โมเดลที่มีอยู่")
            return

        print(f"\n🤖 กำลังเทรนโมเดล XGBoost {self.num_models} ตัว...")
        
        # สร้าง XGBoost parameters จาก config
        xgb_params = {
            'n_estimators': CONFIG['xgboost_params']['n_estimators'],
            'max_depth': CONFIG['xgboost_params']['max_depth'],
            'learning_rate': CONFIG['xgboost_params']['learning_rate'],
            'objective': CONFIG['xgboost_params']['objective'],
            'eval_metric': CONFIG['xgboost_params']['eval_metric'],
            'subsample': CONFIG['xgboost_params']['subsample'],
            'colsample_bytree': CONFIG['xgboost_params']['colsample_bytree'],
            'min_child_weight': CONFIG['xgboost_params']['min_child_weight'],
            'gamma': CONFIG['xgboost_params']['gamma']
        }
        
        for i in range(self.num_models):
            print(f"\nเทรนโมเดลที่ {i+1}/{self.num_models}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=CONFIG['model']['test_size'],
                random_state=CONFIG['model']['random_state'] + i
            )
            
            # Scale features
            if self.scalers[i] is None:
                self.scalers[i] = StandardScaler()
            X_train_scaled = self.scalers[i].fit_transform(X_train)
            X_test_scaled = self.scalers[i].transform(X_test)
            
            # สร้างและเทรนโมเดลด้วยพารามิเตอร์จาก config
            self.models[i] = xgb.XGBClassifier(**xgb_params)
            
            # เทรนโมเดลด้วย early stopping จาก config
            early_stopping = CONFIG['xgboost_params'].get('early_stopping_rounds', 25)
            print(f"   ⏳ Early stopping rounds: {early_stopping}")
            
            self.models[i].fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=early_stopping,
                verbose=False
            )
            
            # ประเมินผลโมเดล
            train_score = self.models[i].score(X_train_scaled, y_train)
            test_score = self.models[i].score(X_test_scaled, y_test)
            print(f"   Model {i+1} - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
        
        # บันทึกโมเดล
        self.save_model()
        print("\n✨ เทรนโมเดลเสร็จสิ้น - ใช้พารามิเตอร์ใหม่ทั้งหมด")
    
    def predict_next_movement(self, current_data):
        """ทำนายทิศทางราคาถัดไปโดยใช้หลายโมเดล"""
        if not all(model is not None for model in self.models):
            raise Exception("❌ โมเดลยังไม่ได้ถูกเทรนครบ")
        
        predictions = []
        probabilities = []
        
        for i in range(self.num_models):
            # Scale features
            scaled_data = self.scalers[i].transform(current_data)
            
            # ทำนาย
            pred = self.models[i].predict(scaled_data)
            prob = self.models[i].predict_proba(scaled_data)
            
            predictions.append(pred[0])
            probabilities.append(prob[0])
        
        # ใช้เสียงส่วนใหญ่ในการตัดสินใจ
        predictions = np.array(predictions)
        counts = np.bincount(predictions)
        final_prediction = np.argmax(counts)
        
        # คำนวณความมั่นใจจากเฉพาะโมเดลที่ทำนายตรงกับ final_prediction
        matching_probabilities = []
        for i, pred in enumerate(predictions):
            if pred == final_prediction:
                matching_probabilities.append(probabilities[i][final_prediction])
        
        # คำนวณค่าเฉลี่ยความมั่นใจ
        avg_confidence = np.mean(matching_probabilities)
        
        # สร้าง array ความน่าจะเป็นที่มีขนาดเท่ากับจำนวนคลาส (2)
        avg_probability = np.zeros(2)
        avg_probability[final_prediction] = avg_confidence
        avg_probability[1 - final_prediction] = 1 - avg_confidence
        
        return final_prediction, avg_probability
    
    def execute_trade(self, prediction, probability, quantity):
        """ทำการเทรดตามการทำนาย"""
        buy_threshold = CONFIG['trading']['buy_threshold']
        sell_threshold = CONFIG['trading']['sell_threshold']
        
        print(f"\n🎯 ผลการทำนาย:")
        print(f"   ทิศทาง: {'ขึ้น ⬆️' if prediction == 1 else 'ลง ⬇️'}")
        print(f"   ความมั่นใจ: {probability[prediction] * 100:.2f}%")
        
        # ตรวจสอบว่ามีการเทรดที่เปิดอยู่หรือไม่
        has_open_trade = any(trade.get('is_open', False) for trade in self.trade_history)
        
        if prediction == 1 and probability[prediction] > buy_threshold:
            # ถ้ามีการเทรดที่เปิดอยู่ ให้ข้ามการซื้อ
            if has_open_trade:
                print("\n⚠️ มีการเทรดที่เปิดอยู่ ต้องรอขายก่อน")
                return
                
            print(f"\n🟢 ส่งคำสั่งซื้อ {quantity} {self.symbol}")
            try:
                # เปิด order ซื้อ
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=self.SIDE_BUY,
                    type=self.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # บันทึกราคาที่ซื้อ
                entry_price = float(order['fills'][0]['price'])
                
                # คำนวณมูลค่าการเทรดและอัพเดทยอดเงิน/crypto
                trade_value = entry_price * quantity
                self.current_balance -= trade_value  # หักเงิน USDT
                self.base_balance += quantity  # เพิ่ม crypto
                
                # บันทึกการเทรด
                trade_info = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'BUY',
                    'price': entry_price,
                    'quantity': quantity,
                    'value': trade_value,
                    'balance': self.current_balance,
                    'crypto_balance': self.base_balance,
                    'prediction': 'ราคาจะขึ้น',
                    'confidence': probability[prediction] * 100,
                    'is_open': True,
                    'order_id': order['orderId']
                }
                self.trade_history.append(trade_info)
                self.save_trade_history()
                
                print(f"✅ คำสั่งซื้อสำเร็จที่ราคา {entry_price:.2f} {self.quote_currency}")
                print(f"💵 มูลค่าการเทรด: {trade_value:.2f} {self.quote_currency}")
                print(f"💰 เงินคงเหลือ: {self.current_balance:.2f} {self.quote_currency}")
                print(f"🪙 {self.base_currency} คงเหลือ: {self.base_balance:.8f}")
                
            except Exception as e:
                print(f"❌ เกิดข้อผิดพลาดในการซื้อ: {str(e)}")
            
        elif prediction == 0 and probability[prediction] > sell_threshold:
            # ตรวจสอบว่ามีการเทรดที่เปิดอยู่หรือไม่
            if not has_open_trade:
                print("\n⚠️ ไม่มีการเทรดที่เปิดอยู่")
                return
                
            # ตรวจสอบว่ามี crypto พอขายไหม
            if self.base_balance < quantity:
                print(f"\n⚠️ มี {self.base_currency} ไม่พอขาย (ต้องการ {quantity}, มี {self.base_balance})")
                return
                
            print(f"\n🔴 ส่งคำสั่งขาย {quantity} {self.symbol}")
            try:
                # เปิด order ขาย
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=self.SIDE_SELL,
                    type=self.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # บันทึกราคาที่ขาย
                exit_price = float(order['fills'][0]['price'])
            
                # คำนวณมูลค่าการเทรดและอัพเดทยอดเงิน/crypto
                trade_value = exit_price * quantity
                self.current_balance += trade_value  # เพิ่มเงิน USDT
                self.base_balance -= quantity  # ลด crypto
                
                # บันทึกการเทรด
                # หาการเทรดที่เปิดอยู่ล่าสุด
                open_trade = next((trade for trade in reversed(self.trade_history) if trade.get('is_open', False)), None)
                if open_trade:
                    # คำนวณกำไร/ขาดทุน
                    entry_value = open_trade['price'] * quantity
                    profit_loss = trade_value - entry_value
                    
                    trade_info = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'SELL',
                        'price': exit_price,
                        'quantity': quantity,
                        'value': trade_value,
                        'balance': self.current_balance,
                        'crypto_balance': self.base_balance,
                        'prediction': 'ราคาจะลง',
                        'confidence': probability[prediction] * 100,
                        'is_open': False,
                        'entry_price': open_trade['price'],
                        'profit_loss': profit_loss,
                        'order_id': order['orderId']
                    }
                    self.trade_history.append(trade_info)
                    
                    # อัพเดทสถิติการเทรด
                    self.total_trades += 1
                    if profit_loss > 0:
                        self.winning_trades += 1
                    self.total_profit_loss += profit_loss
                    
                    # ปิดการเทรดที่เปิดอยู่
                    open_trade['is_open'] = False
                    
                    self.save_trade_history()
                    
                    print(f"✅ คำสั่งขายสำเร็จที่ราคา {exit_price:.2f} {self.quote_currency}")
                    print(f"💵 มูลค่าการเทรด: {trade_value:.2f} {self.quote_currency}")
                    print(f"📊 กำไร/ขาดทุน: {profit_loss:.2f} {self.quote_currency}")
                    print(f"💰 เงินคงเหลือ: {self.current_balance:.2f} {self.quote_currency}")
                    print(f"🪙 {self.base_currency} คงเหลือ: {self.base_balance:.8f}")
                
            except Exception as e:
                print(f"❌ เกิดข้อผิดพลาดในการขาย: {str(e)}")
        else:
            print("\n⚪️ ไม่มีการเทรด (ความมั่นใจต่ำกว่า threshold)")
    
    def run(self):
        """รันบอทเทรด"""
        trade_quantity = CONFIG['trading']['trade_quantity']
        retrain_every_n_cycles = CONFIG['model']['retrain_every_n_cycles']
        trade_interval = CONFIG['trading']['trade_interval']
        
        # คำนวณเวลารอจาก trade_interval
        trade_interval_time = 60  # ค่าเริ่มต้น 1 นาที
        if trade_interval.endswith('s'):
            trade_interval_time = int(trade_interval[:-1])
        elif trade_interval.endswith('m'):
            trade_interval_time = int(trade_interval[:-1]) * 60
        elif trade_interval.endswith('h'):
            trade_interval_time = int(trade_interval[:-1]) * 60 * 60
        
        print(f"\n🚀 เริ่มต้นบอทเทรด...")
        print(f"   ปริมาณการเทรดต่อครั้ง: {trade_quantity} {self.symbol}")
        print(f"   เทรนโมเดลใหม่ทุก {retrain_every_n_cycles} นาที")
        print(f"   รอบการเทรด: ทุก {trade_interval}")
        
        # แสดงสรุปผลการเทรดก่อนเริ่ม
        self.print_trading_summary()
        
        cycle = 1
        while True:
            try:
                print(f"\n{'='*50}")
                print(f"📍 รอบที่ {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                
                # ดึงข้อมูลและเทรนโมเดล
                df = self.get_historical_data()
                X, y = self.prepare_features(df)
                
                # เทรนโมเดลใหม่ทุก N รอบ
                force_train = (cycle % retrain_every_n_cycles) == 0
                if force_train:
                    print("\n🔄 ถึงเวลาเทรนโมเดลใหม่")
                self.train_model(X, y, force_train=force_train)
                
                # ดึงข้อมูลล่าสุดสำหรับการทำนาย
                current_data = X.iloc[-1:].copy()
                
                # ทำนายและเทรด
                prediction, probability = self.predict_next_movement(current_data)
                self.execute_trade(prediction, probability, trade_quantity)
                
                # แสดงสรุปผลการเทรดทุก 10 รอบ
                if cycle % 10 == 0:
                    self.print_trading_summary()
                
                cycle += 1
                time.sleep(trade_interval_time)
                
            except Exception as e:
                print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
                time.sleep(trade_interval_time)
                continue

    def clear_trade_history(self):
        """เคลียร์ประวัติการเทรดทั้งหมด"""
        print("\n🗑️ กำลังเคลียร์ประวัติการเทรด...")
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit_loss = 0
        self.trade_history = []
        self.save_trade_history()
        print("✅ เคลียร์ประวัติการเทรดเรียบร้อย")

if __name__ == "__main__":
    bot = TradingBot()
    bot.run() 