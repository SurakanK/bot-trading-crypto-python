import streamlit as st
import pandas as pd
import json
import pathlib
import time
import os
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")

def load_trade_history():
    """โหลดประวัติการเทรดจากไฟล์"""
    try:
        trade_history_path = pathlib.Path('models/trade_history_BTCUSDT_1m.json')
        if trade_history_path.exists():
            with open(trade_history_path, 'r') as f:
                trade_data = json.load(f)
            return trade_data
        else:
            st.warning("ยังไม่มีประวัติการเทรด")
            return None
    except Exception as e:
        st.error(f"ไม่สามารถโหลดประวัติการเทรดได้: {str(e)}")
        return None

def check_file_changes():
    """ตรวจสอบการเปลี่ยนแปลงของไฟล์ trade history"""
    trade_history_file = 'models/trade_history_BTCUSDT_1m.json'
    
    try:
        if os.path.exists(trade_history_file):
            with open(trade_history_file, 'r') as f:
                current_data = json.load(f)
                
            if not hasattr(st.session_state, 'last_trade_count'):
                st.session_state.last_trade_count = len(current_data['trade_history'])
            
            current_trade_count = len(current_data['trade_history'])
            
            if current_trade_count != st.session_state.last_trade_count:
                st.session_state.last_trade_count = current_trade_count
                return True
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการตรวจสอบไฟล์: {str(e)}")
    
    return False

def main():
    st.title("🤖 Trading Bot Dashboard")
    
    # ตรวจสอบการเปลี่ยนแปลงและรีเฟรชถ้ามีการเทรดใหม่
    if check_file_changes():
        st.experimental_rerun()
    
    # สร้าง columns สำหรับแสดงข้อมูลสรุป
    col1, col2, col3, col4 = st.columns(4)
    
    # โหลดข้อมูลการเทรด
    trade_data = load_trade_history()
    
    if trade_data:
        # แสดงข้อมูลสรุป
        with col1:
            st.metric(
                "💰 เงินในบัญชี (USDT)", 
                f"{trade_data['current_balance']:.2f}",
                f"{((trade_data['current_balance']/trade_data['initial_balance'])-1)*100:.2f}%"
            )
        
        with col2:
            win_rate = (trade_data['winning_trades'] / trade_data['total_trades'] * 100) if trade_data['total_trades'] > 0 else 0
            st.metric("📊 Win Rate", f"{win_rate:.2f}%")
        
        with col3:
            st.metric(
                "📈 กำไร/ขาดทุน (USDT)", 
                f"{trade_data['total_profit_loss']:.2f}"
            )
        
        # แสดง Crypto Balance
        with col4:
            if len(trade_data['trade_history']) > 0:
                last_trade = trade_data['trade_history'][-1]
                if 'crypto_balance' in last_trade:
                    st.metric(
                        "🪙 BTC Balance",
                        f"{last_trade['crypto_balance']:.8f}",
                        help="จำนวน BTC ที่มีในบัญชี"
                    )
        
        # แสดงกราฟราคาและการเทรด
        if trade_data['trade_history']:
            df = pd.DataFrame(trade_data['trade_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = go.Figure()
            
            # เพิ่มจุดซื้อ (สีเขียว)
            buy_points = df[df['type'] == 'BUY']
            if not buy_points.empty:
                fig.add_trace(go.Scatter(
                    x=buy_points['timestamp'],
                    y=buy_points['price'],
                    mode='markers',
                    name='ซื้อ',
                    marker=dict(color='green', size=10)
                ))
            
            # เพิ่มจุดขาย (สีแดง)
            sell_points = df[df['type'] == 'SELL']
            if not sell_points.empty:
                fig.add_trace(go.Scatter(
                    x=sell_points['timestamp'],
                    y=sell_points['price'],
                    mode='markers',
                    name='ขาย',
                    marker=dict(color='red', size=10)
                ))
            
            fig.update_layout(
                title='การเทรดทั้งหมด',
                xaxis_title='เวลา',
                yaxis_title='ราคา (USDT)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # แสดงตารางประวัติการเทรด
            st.subheader("📝 ประวัติการเทรด")
            df_display = df[['timestamp', 'type', 'price', 'quantity', 'value', 'balance', 'prediction', 'confidence']]
            
            # แปลง prediction เป็น emoji
            df_display['prediction'] = df_display['prediction'].map({
                'ราคาจะขึ้น': '⬆️',
                'ราคาจะลง': '⬇️'
            })
            
            df_display = df_display.rename(columns={
                'timestamp': 'เวลา',
                'type': 'ประเภท',
                'price': 'ราคา',
                'quantity': 'ปริมาณ',
                'value': 'มูลค่า',
                'balance': 'เงินคงเหลือ',
                'prediction': 'ทำนาย',
                'confidence': 'ความน่าจะเป็น'
            })
            df_display['ความน่าจะเป็น'] = df_display['ความน่าจะเป็น'].map('{:.2f}%'.format)
            st.dataframe(
                df_display.sort_values('เวลา', ascending=False)
            )
    else:
        st.info("กำลังรอข้อมูลการเทรด...")
    
    # เพิ่มปุ่ม Refresh
    if st.button("🔄 Refresh"):
        st.experimental_rerun()

    # เพิ่ม auto refresh ทุก 5 วินาที
    time.sleep(5)
    st.experimental_rerun()

if __name__ == "__main__":
    main() 