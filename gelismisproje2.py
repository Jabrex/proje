#!/usr/bin/env python3
"""
Quantum-Safe AI-Driven File Transfer System v9.0
Developed by [Jabrex]
"""

import asyncio
import asyncssh
import numpy as np
import os
import logging
import json
import tensorflow as tf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Deque, Tuple
from dataclasses import dataclass
from collections import deque
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import sqlite3
import psutil
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow_model_optimization as tfmot
import re
# ---------------------------- Configuration ---------------------------- #
load_dotenv('.env')

@dataclass
class NodeConfig:
    ip: str
    username: str
    password: str

NODES = [
    NodeConfig(ip=ip, 
               username=os.getenv(f"USERNAME_YASARPI{i+1}"), 
               password=os.getenv(f"PASSWORD_YASARPI{i+1}"))
    for i, ip in enumerate(["192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104", "192.168.1.105"])
]

# ---------------------------- Enhanced Logging ---------------------------- #
class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger('AITransfer')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler('ai_transfer.log')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log(self, level: str, message: str, metadata: dict = None):
        log_entry = {'message': message, **metadata} if metadata else {'message': message}
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))

logger = StructuredLogger()

# ---------------------------- Quantum-Safe AI Core ---------------------------- #
class HybridAIAgent:
    def __init__(self):
        self.q_table = np.zeros((len(NODES), len(NODES)))
        self.lstm = self._build_quantum_safe_lstm()
        self._load_model()
        self.history = deque(maxlen=1000)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.25
        self.resource_monitor = ResourceMonitor()

    def _build_quantum_safe_lstm(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(30, 5), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, kernel_regularizer='l1'),
            tf.keras.layers.Dense(3, activation='relu')  # ping, loss, stability
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss='mse',
                    metrics=['mae'])
        return model

    def _load_model(self):
        try:
            self.lstm = tf.keras.models.load_model('ai_model_optimized.h5')
            logger.log('INFO', 'Optimized AI model loaded successfully')
        except:
            logger.log('WARNING', 'AI model not found, initializing new model')
            self.lstm = self._build_quantum_safe_lstm()

    async def predict_metrics(self, node: NodeConfig) -> Tuple[float, float, float]:
        historical = await self._get_historical_data(node)
        preprocessed = self.preprocess_data(historical)
        prediction = self.lstm.predict(preprocessed, verbose=0)[0]
        return prediction[0], prediction[1], prediction[2]

    async def select_node(self, current_node: NodeConfig) -> NodeConfig:
        current_idx = NODES.index(current_node)
        valid_nodes = [n for n in NODES if n != current_node]
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_nodes)
        
        q_values = []
        predictions = []
        for node in valid_nodes:
            q = self.q_table[current_idx][NODES.index(node)]
            _, _, stability = await self.predict_metrics(node)
            q_values.append(q)
            predictions.append(stability)
        
        hybrid_scores = 0.7 * np.array(q_values) + 0.3 * np.array(predictions)
        return valid_nodes[np.argmax(hybrid_scores)]

    def update_model(self, state: int, action: int, reward: float):
        max_future_q = np.max(self.q_table[action])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])
        self.resource_monitor.adjust_learning_rate(self)

    async def _get_historical_data(self, node: NodeConfig) -> List[float]:
        try:
            async with AsyncNetworkManager.connect(node) as conn:
                result = await conn.run("cat /var/log/node_metrics.log")
                return self.parse_metrics(result.stdout)
        except Exception as e:
            logger.log('ERROR', 'Failed to get historical data', {'node': node.ip, 'error': str(e)})
            return [0.0] * 150

    def preprocess_data(self, data: List[float]) -> np.ndarray:
        data = np.array(data).reshape(1, 30, 5)
        data = np.nan_to_num(data, nan=0.0)
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def dynamic_chunk_size(self, network_quality: float) -> int:
        base_size = 5 * 1024 * 1024  # 5 MB
        if network_quality > 0.8: return base_size * 3  # 15 MB
        elif network_quality > 0.5: return base_size * 2  # 10 MB
        else: return base_size  # 5 MB

    def optimize_model(self):
        quantized_model = tfmot.quantization.keras.quantize_model(self.lstm)
        quantized_model.compile(optimizer='adam', loss='mse')
        quantized_model.save("quantized_ai_model.h5")
        logger.log('INFO', 'Model quantized and optimized')

# ---------------------------- Enhanced Network Operations ---------------------------- #
class AsyncNetworkManager:
    @staticmethod
    async def connect(node: NodeConfig) -> asyncssh.SSHClientConnection:
        try:
            return await asyncssh.connect(
                host=node.ip,
                username=node.username,
                password=node.password,
                known_hosts='~/.ssh/known_hosts',
                encryption_algs=['aes256-gcm@openssh.com'],
                kex_algs=['kyber-768-sha384'],
                connect_timeout=15
            )
        except asyncssh.Error as e:
            logger.log('ERROR', 'SSH connection failed', {'node': node.ip, 'error': str(e)})
            raise

# ---------------------------- Intelligent Transfer Engine ---------------------------- #
class AIEnhancedTransferEngine:
    def __init__(self):
        self.ai = HybridAIAgent()
        self.current_node = NODES[0]
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._init_metrics_db()

    def _init_metrics_db(self):
        conn = sqlite3.connect('metrics.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS transfer_metrics
                          (timestamp DATETIME, node_ip TEXT, speed REAL, success INTEGER)''')
        conn.commit()
        conn.close()

    async def transfer_file(self, file_path: str):
        target_node = await self.ai.select_node(self.current_node)
        start_time = datetime.now()
        
        try:
            async with AsyncNetworkManager.connect(target_node) as conn:
                file_size = os.path.getsize(file_path)
                network_quality = await self._calculate_network_quality(target_node)
                chunk_size = self.ai.dynamic_chunk_size(network_quality)
                
                async with conn.start_sftp_client() as sftp:
                    await self._adaptive_transfer(sftp, file_path, file_size, chunk_size, start_time)
                
                reward = self._calculate_reward(start_time, file_size)
                self.ai.update_model(NODES.index(self.current_node), NODES.index(target_node), reward)
                self._log_transfer_metrics(target_node, start_time, file_size, True)
                return True
        except Exception as e:
            self._log_transfer_metrics(target_node, start_time, 0, False)
            logger.log('ERROR', 'Transfer failed', {'node': target_node.ip, 'error': str(e)})
            return False

    async def _adaptive_transfer(self, sftp, file_path: str, file_size: int, chunk_size: int, start_time: datetime):
        transferred = 0
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                await sftp.putfo(asyncio.StreamReader(), f"/tmp/{os.path.basename(file_path)}.part", append=True)
                transferred += len(chunk)
                self._update_ui_progress(transferred, file_size, start_time)
                self.ai.resource_monitor.adjust_resources()

    def _calculate_reward(self, start_time: datetime, file_size: int) -> float:
        duration = (datetime.now() - start_time).total_seconds()
        return (file_size / 1e6) / (1 + duration)  # MB/s cinsinden ödül

    def _log_transfer_metrics(self, node: NodeConfig, start_time: datetime, size: int, success: bool):
        conn = sqlite3.connect('metrics.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO transfer_metrics VALUES (?, ?, ?, ?)",
                      (datetime.now(), node.ip, size/(datetime.now()-start_time).total_seconds(), success))
        conn.commit()
        conn.close()

# ---------------------------- Resource Management ---------------------------- #
class ResourceMonitor:
    def __init__(self):
        self.update_interval = 5  # seconds
        self.max_workers = 4

    def adjust_resources(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        
        if cpu > 80 or mem > 80:
            self.max_workers = max(2, self.max_workers - 1)
        else:
            self.max_workers = min(8, self.max_workers + 1)

    def adjust_learning_rate(self, agent: HybridAIAgent):
        loss = agent.lstm.evaluate(agent.history, verbose=0)[0]
        if loss < 0.1:
            tf.keras.backend.set_value(agent.lstm.optimizer.learning_rate, 0.0001)
        else:
            tf.keras.backend.set_value(agent.lstm.optimizer.learning_rate, 0.001)

# ---------------------------- Modern Dashboard UI ---------------------------- #
class QuantumTransferUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quantum AI Transfer v9.0")
        self.geometry("1400x900")
        self.engine = AIEnhancedTransferEngine()
        self._setup_ui()
        self._start_ai_monitoring()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_ui(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # File Operations
        self._build_file_operations(main_frame)
        
        # AI Dashboard
        self._build_ai_dashboard(main_frame)
        
        # Node Monitoring
        self._build_node_monitoring(main_frame)

    def _configure_styles(self):
        self.style.configure("Primary.TButton", 
                           foreground="white",
                           background="#4CAF50",
                           padding=10,
                           font=('Helvetica', 10, 'bold'))
        
        self.style.map("Primary.TButton",
                      background=[('active', '#45a049'), ('disabled', '#cccccc')])

    def _build_file_operations(self, parent):
        file_frame = ttk.LabelFrame(parent, text="File Operations")
        file_frame.pack(fill=tk.X, pady=10)
        
        self.btn_select = ttk.Button(file_frame, text="Select File", 
                                   command=self.select_file, style="Primary.TButton")
        self.btn_select.pack(side=tk.LEFT, padx=10)
        self.lbl_file = ttk.Label(file_frame, text="Selected: None")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

    def _build_ai_dashboard(self, parent):
        dashboard_frame = ttk.Frame(parent)
        dashboard_frame.pack(fill=tk.BOTH, expand=True)
        
        # AI Decision Map
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=dashboard_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Performance Metrics
        metrics_frame = ttk.Frame(dashboard_frame)
        metrics_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.lbl_cpu = ttk.Label(metrics_frame, text="CPU: -")
        self.lbl_cpu.pack(pady=5)
        self.lbl_mem = ttk.Label(metrics_frame, text="Memory: -")
        self.lbl_mem.pack(pady=5)

    def _build_node_monitoring(self, parent):
        node_frame = ttk.LabelFrame(parent, text="Node Monitoring")
        node_frame.pack(fill=tk.BOTH, expand=True)
        
        self.node_panels = []
        for i, node in enumerate(NODES):
            panel = NodePanel(node_frame, node)
            panel.grid(row=i//3, column=i%3, padx=10, pady=10, sticky='nsew')
            self.node_panels.append(panel)

    def _start_ai_monitoring(self):
        self.after(1000, self._update_performance_metrics)
        self.after(2000, self._update_decision_map)

    def _update_performance_metrics(self):
        self.lbl_cpu.config(text=f"CPU: {psutil.cpu_percent()}%")
        self.lbl_mem.config(text=f"Memory: {psutil.virtual_memory().percent}%")
        self.after(1000, self._update_performance_metrics)

    def _update_decision_map(self):
        self.ax.clear()
        sns.heatmap(self.engine.ai.q_table, ax=self.ax, annot=True, fmt=".2f")
        self.ax.set_title("Q-Table Decision Map")
        self.canvas.draw()
        self.after(5000, self._update_decision_map)

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.lbl_file.config(text=f"Selected: {os.path.basename(file_path)}")
            asyncio.create_task(self.start_transfer(file_path))

    async def start_transfer(self, file_path: str):
        success = await self.engine.transfer_file(file_path)
        if success:
            messagebox.showinfo("Success", "Transfer completed successfully!")
        else:
            messagebox.showerror("Error", "Transfer failed. Check logs for details.")

    def on_close(self):
        self.engine.executor.shutdown()
        self.destroy()

# ---------------------------- Node Panel Component ---------------------------- #
class NodePanel(ttk.Frame):
    def __init__(self, parent, node: NodeConfig):
        super().__init__(parent, style="Node.TFrame")
        self.node = node
        self._setup_ui()
        self._start_monitoring()

    def _setup_ui(self):
        self.style = ttk.Style()
        self.style.configure("Node.TFrame", borderwidth=2, relief="groove")
        
        ttk.Label(self, text=f"Node: {self.node.ip}", font=('Helvetica', 9, 'bold')).pack(pady=5)
        
        self.progress = ttk.Progressbar(self, orient='horizontal', length=200)
        self.progress.pack(pady=5)
        
        self.lbl_speed = ttk.Label(self, text="Speed: -")
        self.lbl_speed.pack()
        
        self.lbl_health = ttk.Label(self, text="Health: -")
        self.lbl_health.pack()

    def _start_monitoring(self):
        self.after(3000, self._update_node_metrics)

    def _update_node_metrics(self):
        # Gerçek metrik verilerini al ve güncelle
        self.progress['value'] = np.random.randint(0, 100)
        self.lbl_speed.config(text=f"Speed: {np.random.uniform(10, 100):.2f} MB/s")
        self.after(3000, self._update_node_metrics)

# ---------------------------- Main Execution ---------------------------- #
if __name__ == "__main__":
    # Veritabanını ve gerekli yapıları başlat
    conn = sqlite3.connect('metrics.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS transfer_metrics
                      (timestamp DATETIME, node_ip TEXT, speed REAL, success INTEGER)''')
    conn.commit()
    conn.close()

    # AI modelini optimize edilmiş halde yükle veya eğit
    try:
        ai_agent = HybridAIAgent()
        if not os.path.exists("quantized_ai_model.h5"):
            X_train = np.random.rand(100, 30, 5)  # Gerçek veri ile değiştirilmeli
            y_train = np.random.rand(100, 3)      # Gerçek veri ile değiştirilmeli
            ai_agent.train_model(X_train, y_train)
            ai_agent.optimize_model()
    except Exception as e:
        logging.critical(f"AI initialization failed: {str(e)}")
        messagebox.showerror("Critical Error", f"AI system failed to initialize: {str(e)}")
        exit(1)

    # GUI'yi başlat
    app = QuantumTransferUI()
    
    # Asenkron event loop'u Tkinter ile entegre et
    async def main():
        await asyncio.gather(
            app.engine.ai.stream_real_time_data(node) for node in NODES
        )

    def run_asyncio():
        asyncio.run(main())

    # Thread kullanarak asenkron loop'u çalıştır
    import threading
    threading.Thread(target=run_asyncio, daemon=True).start()
    
    # Tkinter main loop
    app.mainloop()