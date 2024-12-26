#!/usr/bin/env python3

import os
import math
import json
import time
import socket
import threading
import subprocess
import paramiko
import argparse
import tkinter as tk
from tkinter import filedialog
import logging
import psutil
import hashlib
import uuid
import glob
import random  # Rastgele sayı işlemleri için
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from threading import Thread, Lock

#############################################################################
# Özel Hata Sınıfları ve Eksik Fonksiyonlar (Stub)
#############################################################################

class SecurityError(Exception):
    """Güvenlikle ilgili özel hata."""
    pass

class IntegrityError(Exception):
    """Dosya bütünlüğüyle ilgili özel hata."""
    pass

def get_device_load(node):
    """Örnek CPU yükü fonksiyonu (stub).
       Gerçek ortamda SSH/API vb. ile gerçek değer alınmalı."""
    return random.uniform(0, 100)

def get_device_memory(node):
    """Örnek bellek durumu fonksiyonu (stub)."""
    return {'used_percent': random.uniform(0, 100)}

def get_network_load(node):
    """Örnek ağ kullanımı fonksiyonu (stub)."""
    return random.uniform(0, 100)

def get_ping_time(src_node, dst_node):
    """Örnek ping süresi fonksiyonu (stub)."""
    return random.uniform(1, 100)

def measure_bandwidth(src_node, dst_node):
    """Örnek bant genişliği ölçüm fonksiyonu (stub)."""
    return random.uniform(0, 1000)

def scp_upload(source_path, dest_path, dest_node):
    """Örnek SCP yükleme fonksiyonu (stub).
       Gerçek sistemde paramiko SFTP veya scp vb. kullanılmalı."""
    return True

#############################################################################
# Raspberry Pi Configuration
#############################################################################
devices = {
    "pi100": {
        "ip": "192.168.1.100",
        "username": "yasarpi1",
        "password": "yasarpi1"
    },
    "pi101": {
        "ip": "192.168.1.101",
        "username": "yasarpi2",
        "password": "yasarpi2"
    },
    "pi102": {
        "ip": "192.168.1.102",
        "username": "yasarpi3",
        "password": "yasarpi3"
    },
    "pi103": {
        "ip": "192.168.1.103",
        "username": "yasarpi4",
        "password": "yasarpi4"
    },
    "pi104": {
        "ip": "192.168.1.104",
        "username": "yasarpi5",
        "password": "yasarpi5"
    }
}

graph_neighbors = {
    "pi100": ["pi101", "pi102"],
    "pi101": ["pi100", "pi103", "pi104"],
    "pi102": ["pi100", "pi103", "pi104"],
    "pi103": ["pi101", "pi102", "pi104"],
    "pi104": ["pi101", "pi102", "pi103"]
}

UDP_PORT = 698

#############################################################################
# Enhanced SARSA Agent
#############################################################################
class EnhancedSarsaAgent:
    def __init__(self, node_id, alpha=0.1, gamma=0.9, epsilon=0.1, buffer_size=1000):
        self.node_id = node_id
        self.q_table = {}
        self.alpha = alpha  # Öğrenme oranı
        self.initial_alpha = alpha
        self.gamma = gamma  # İndirim faktörü
        self.epsilon = epsilon  # Keşif oranı
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        
        # Experience Replay için buffer
        self.experience_buffer = []
        self.buffer_size = buffer_size
        
        # Metrik ağırlıkları (Örnek kullanım)
        self.weights = {
            'rtt': 0.3,
            'cpu_load': 0.3,
            'memory': 0.2,
            'bandwidth': 0.2
        }
        
        self.q_file_path = f"enhanced_qtable_{node_id}.json"
        self.stats_file_path = f"sarsa_stats_{node_id}.json"
        self.load_q_table()
        
        # İstatistikler
        self.stats = {
            'updates': 0,
            'avg_reward': 0.0,
            'total_reward': 0.0,
            'successful_routes': 0
        }

    def get_q_value(self, state, action):
        return self.q_table.get((state[0], state[1], action), 0.0)

    def calculate_reward(self, metrics):
        """Çoklu metrik bazlı ödül hesaplama (kullanılabilir)."""
        if not metrics or None in metrics.values():
            return -10.0
            
        reward = (
            self.weights['rtt'] * (1.0 / (1.0 + metrics['rtt'])) +
            self.weights['cpu_load'] * (1.0 - metrics['cpu_load'] / 100.0) +
            self.weights['memory'] * (1.0 - metrics['memory'] / 100.0) +
            self.weights['bandwidth'] * (metrics['bandwidth'] / 1000.0)
        )
        
        return reward * 10  # Ölçeklendirme

    def select_action(self, current_node, end_node, possible_next_nodes, metrics=None):
        """Epsilon-greedy politikası ile aksiyon seçimi"""
        if not possible_next_nodes:
            return None
            
        if random.random() < self.epsilon:
            # Keşif: Rastgele seçim
            return random.choice(possible_next_nodes)
        
        # Sömürü: En iyi Q-değerine sahip aksiyonu seç
        state = (current_node, end_node)
        return max(possible_next_nodes, key=lambda x: self.get_q_value(state, x))

    def update_q_value(self, state, action, reward, next_state, next_action):
        """SARSA güncelleme"""
        current_q = self.get_q_value(state, action)
        next_q = 0.0 if next_action is None else self.get_q_value(next_state, next_action)
        
        # Adaptif öğrenme oranı
        self.alpha = self.initial_alpha / (1 + self.stats['updates'] * 0.001)
        
        # Q-değeri güncelleme
        td_error = reward + self.gamma * next_q - current_q
        new_q = current_q + self.alpha * td_error
        
        self.q_table[(state[0], state[1], action)] = new_q
        
        # İstatistik güncelleme
        self.stats['updates'] += 1
        self.stats['total_reward'] += reward
        self.stats['avg_reward'] = self.stats['total_reward'] / self.stats['updates']
        
        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Experience buffer'a ekle
        self.store_experience(state, action, reward, next_state, next_action)
        
        # Periyodik kayıt
        if self.stats['updates'] % 100 == 0:
            self.save_q_table()
            self.save_stats()

    def store_experience(self, state, action, reward, next_state, next_action):
        """Experience replay için deneyim saklama"""
        experience = (state, action, reward, next_state, next_action)
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

    def replay_experiences(self, batch_size=32):
        """Experience replay ile öğrenme"""
        if len(self.experience_buffer) < batch_size:
            return
        batch = random.sample(self.experience_buffer, batch_size)
        for experience in batch:
            state, action, reward, next_state, next_action = experience
            self.update_q_value(state, action, reward, next_state, next_action)

    def save_q_table(self):
        """Q-tablosunu JSON formatında kaydet"""
        try:
            serializable_q = {
                f"{k[0]},{k[1]},{k[2]}": v 
                for k, v in self.q_table.items()
            }
            with open(self.q_file_path, "w") as f:
                json.dump(serializable_q, f, indent=2)
        except Exception as e:
            print(f"[SARSA-{self.node_id}] Q-table kaydetme hatası: {e}")

    def load_q_table(self):
        """Kaydedilmiş Q-tablosunu yükle"""
        if os.path.isfile(self.q_file_path):
            try:
                with open(self.q_file_path, "r") as f:
                    loaded = json.load(f)
                self.q_table = {
                    tuple(k.split(",")): v 
                    for k, v in loaded.items()
                }
            except Exception as e:
                print(f"[SARSA-{self.node_id}] Q-table yükleme hatası: {e}")
                self.q_table = {}

    def save_stats(self):
        """İstatistikleri kaydet"""
        try:
            with open(self.stats_file_path, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"[SARSA-{self.node_id}] Stats kaydetme hatası: {e}")

#############################################################################
# Enhanced OLSR Protocol
#############################################################################
class EnhancedOLSRNode(threading.Thread):
    def __init__(self, node_id, hello_interval=2, tc_interval=5):
        super().__init__()
        self.node_id = node_id
        self.ip = devices[node_id]["ip"]
        self.hello_interval = hello_interval
        self.tc_interval = tc_interval
        self.stop_flag = False
        
        # Temel topoloji bilgisi
        self.topology = {}  # {node: set(neighbors)}
        self.link_quality = {}  # {(node1, node2): quality_metrics}
        self.mpr_set = set()  # Seçilen MPR'ler
        self.mpr_selectors = set()  # Bu düğümü MPR seçenler
        
        # Metrik takibi
        self.link_metrics = {
            'delay': {},      # {(node1, node2): delay}
            'bandwidth': {},  # {(node1, node2): bandwidth}
            'stability': {}   # {(node1, node2): uptime}
        }
        
        # MPR seçim parametreleri
        self.mpr_coverage = 2
        self.mpr_update_interval = 10
        
        self.initialize_topology()
        self.setup_networking()
        self.setup_logging()

    def setup_networking(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", UDP_PORT))
        self.sock.settimeout(1.0)

    def setup_logging(self):
        self.logger = logging.getLogger(f"OLSR-{self.node_id}")
        handler = logging.FileHandler(f"olsr_{self.node_id}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def initialize_topology(self):
        for node in devices:
            self.topology[node] = set()
            if node in graph_neighbors[self.node_id]:
                self.topology[self.node_id].add(node)
                self.topology[node].add(self.node_id)
                self.link_quality[(self.node_id, node)] = {
                    'quality': 1.0,
                    'last_updated': time.time()
                }

    def calculate_link_quality(self, neighbor):
        delay = self.link_metrics['delay'].get((self.node_id, neighbor), float('inf'))
        bandwidth = self.link_metrics['bandwidth'].get((self.node_id, neighbor), 0)
        stability = self.link_metrics['stability'].get((self.node_id, neighbor), 0)
        
        delay_factor = 1.0 / (1.0 + delay/100.0)
        bw_factor = min(1.0, bandwidth/1000.0)
        stability_factor = min(1.0, stability/3600.0)
        
        quality = 0.4 * delay_factor + 0.4 * bw_factor + 0.2 * stability_factor
        return quality

    def select_mprs(self):
        one_hop = self.topology[self.node_id]
        two_hop = set()
        
        for n in one_hop:
            two_hop.update(self.topology[n])
        two_hop = two_hop - one_hop - {self.node_id}
        
        selected_mprs = set()
        covered = set()
        
        # 1. Benzersiz erişim sağlayan düğümleri seç
        for n in one_hop:
            unique = set(self.topology[n]) & two_hop
            unique = unique - covered
            if unique:
                selected_mprs.add(n)
                covered.update(unique)
        
        # 2. Kalan düğümler için en iyi MPR'leri seç
        uncovered = two_hop - covered
        while uncovered and len(selected_mprs) < len(one_hop):
            best_mpr = None
            best_coverage = set()
            
            for n in one_hop - selected_mprs:
                new_covered = set(self.topology[n]) & uncovered
                quality = self.calculate_link_quality(n)
                coverage_score = len(new_covered) * quality
                
                if not best_mpr or coverage_score > len(best_coverage):
                    best_mpr = n
                    best_coverage = new_covered
            
            if not best_mpr:
                break
                
            selected_mprs.add(best_mpr)
            covered.update(best_coverage)
            uncovered = uncovered - best_coverage
        
        self.mpr_set = selected_mprs
        self.logger.info(f"Updated MPR set: {selected_mprs}")

    def send_hello(self):
        msg = {
            "type": "HELLO",
            "sender": self.node_id,
            "neighbors": list(self.topology[self.node_id]),
            "mpr_set": list(self.mpr_set),
            "link_quality": {
                str(k): v for k, v in self.link_quality.items()
                if self.node_id in k
            }
        }
        self.broadcast_message(msg)

    def send_tc(self):
        msg = {
            "type": "TC",
            "sender": self.node_id,
            "topology": {
                node: list(neighbors) 
                for node, neighbors in self.topology.items()
            },
            "mpr_selectors": list(self.mpr_selectors),
            "metrics": {
                "delay": {str(k): v for k, v in self.link_metrics['delay'].items()},
                "bandwidth": {str(k): v for k, v in self.link_metrics['bandwidth'].items()},
                "stability": {str(k): v for k, v in self.link_metrics['stability'].items()}
            }
        }
        self.broadcast_message(msg)

    def handle_hello(self, msg):
        sender = msg["sender"]
        neighbors = set(msg["neighbors"])
        sender_mprs = set(msg["mpr_set"])
        
        self.topology[sender] = neighbors
        
        for k, v in msg["link_quality"].items():
            node1, node2 = eval(k)
            self.link_quality[(node1, node2)] = v
            self.link_quality[(node2, node1)] = v
        
        if self.node_id in sender_mprs:
            self.mpr_selectors.add(sender)
        else:
            self.mpr_selectors.discard(sender)

    def handle_tc(self, msg):
        sender = msg["sender"]
        
        for node, neighbors in msg["topology"].items():
            self.topology[node] = set(neighbors)
        
        for metric_type, values in msg["metrics"].items():
            for k, v in values.items():
                node1, node2 = eval(k)
                self.link_metrics[metric_type][(node1, node2)] = v
                self.link_metrics[metric_type][(node2, node1)] = v

    def run(self):
        last_hello = 0
        last_tc = 0
        last_mpr = 0
        
        while not self.stop_flag:
            try:
                current_time = time.time()
                if current_time - last_hello >= self.hello_interval:
                    self.send_hello()
                    last_hello = current_time
                
                if current_time - last_tc >= self.tc_interval:
                    self.send_tc()
                    last_tc = current_time
                
                if current_time - last_mpr >= self.mpr_update_interval:
                    self.select_mprs()
                    last_mpr = current_time
                
                try:
                    data, addr = self.sock.recvfrom(8192)
                    msg = json.loads(data.decode())
                    
                    if msg["type"] == "HELLO":
                        self.handle_hello(msg)
                    elif msg["type"] == "TC":
                        self.handle_tc(msg)
                        
                except socket.timeout:
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)
        
        self.sock.close()

    def broadcast_message(self, msg):
        try:
            data = json.dumps(msg).encode()
            self.sock.sendto(data, ('<broadcast>', UDP_PORT))
        except Exception as e:
            self.logger.error(f"Broadcast error: {e}")

    def stop(self):
        self.stop_flag = True

    def get_full_topology(self):
        return self.topology

#############################################################################
# Enhanced Router - Rota Seçim Sürecini Açıklamalı Loglama ile
#############################################################################
class EnhancedRouter:
    def __init__(self, node_id, olsr_node, sarsa_agent):
        self.node_id = node_id
        self.olsr_node = olsr_node
        self.sarsa_agent = sarsa_agent
        
        self.active_routes = {}
        self.backup_routes = {}
        self.route_history = []
        self.max_history = 1000
        
        self.node_loads = {}
        self.link_loads = {}
        
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger(f"Router-{self.node_id}")
        handler = logging.FileHandler(f"router_{self.node_id}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def find_multiple_routes(self, source, destination, max_routes=3):
        routes = []
        topology = self.olsr_node.get_full_topology()
        
        def find_paths(current, dest, visited, path):
            if current == dest:
                routes.append(path[:])
                return
            if len(routes) >= max_routes:
                return
            for next_node in topology[current]:
                if next_node not in visited:
                    visited.add(next_node)
                    find_paths(next_node, dest, visited, path + [next_node])
                    visited.remove(next_node)
        
        visited = {source}
        find_paths(source, destination, visited, [source])
        
        # Rotaları maliyetlerine göre skorla ve açıklama ekle
        scored_routes = []
        for route in routes:
            cost, explanation = self.calculate_path_cost(route, return_explanation=True)
            scored_routes.append({
                "route": route,
                "cost": cost,
                "explanation": explanation  # Adım adım cost verileri
            })
        
        # Maliyet tabanlı sıralama
        scored_routes.sort(key=lambda x: x["cost"])
        
        # Tüm rotalar için log veya JSON kaydı
        self.log_route_choices(scored_routes, source, destination)
        
        # En iyi (maliyet açısından) ilk max_routes rotayı döndür
        return [sr["route"] for sr in scored_routes[:max_routes]]

    def log_route_choices(self, scored_routes, source, destination):
        """Seçilen ve seçilmeyen rotaları açıklamalı şekilde logla."""
        if not scored_routes:
            self.logger.info(f"No routes found from {source} to {destination}")
            return
        
        best_route = scored_routes[0]
        best_cost = best_route["cost"]
        
        log_message = [
            f"--- Route Decision from {source} to {destination} ---",
            f"Best route: {best_route['route']} (cost={best_cost:.2f})"
        ]
        
        for idx, route_info in enumerate(scored_routes):
            route = route_info["route"]
            cost = route_info["cost"]
            explanation = route_info["explanation"]
            
            if idx == 0:
                reason = "SELECTED: Lowest cost"
            else:
                reason = "NOT SELECTED: Higher cost"
            
            step_details = []
            for step in explanation:
                step_msg = (
                    f"{step['from']} -> {step['to']} | "
                    f"lat={step['latency']:.1f}ms, "
                    f"bw={step['bandwidth']:.1f}Mbps, "
                    f"cpu={step['cpu_load']:.1f}%, "
                    f"mem={step['memory_load']:.1f}%, "
                    f"q_val={step['q_value']:.2f}, "
                    f"hop_cost={step['hop_cost']:.2f}"
                )
                step_details.append(step_msg)
            
            log_message.append(
                f"Route {idx+1}: {route}, cost={cost:.2f}, reason={reason}\n"
                f"    Steps:\n      " + "\n      ".join(step_details)
            )
        
        # Log dosyasına yaz
        self.logger.info("\n".join(log_message))
        
        # İstersen JSON dosyasına da kaydedebilirsin (yorum kaldırarak):
        # with open(f"route_decision_{self.node_id}_{source}_to_{destination}.json", "w") as f:
        #     json.dump(scored_routes, f, indent=2, default=str)

    def calculate_path_cost(self, path, metrics=None, return_explanation=False):
        if metrics is None:
            metrics = {}
        if not path or len(path) < 2:
            if return_explanation:
                return float('inf'), [{"error": "Invalid path"}]
            return float('inf')
        
        total_cost = 0
        step_explanations = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_hop = path[i + 1]
            
            link_load = self.link_loads.get((current, next_hop), {})
            latency = link_load.get('latency', 1000)
            bandwidth = link_load.get('bandwidth', 0.1)
            
            node_load = self.node_loads.get(next_hop, {})
            cpu_load = node_load.get('cpu', 100)
            memory_load = node_load.get('memory', 100)
            
            q_value = self.sarsa_agent.get_q_value((current, path[-1]), next_hop)
            
            hop_cost = (
                0.3 * (latency / 100) +
                0.2 * (1 - bandwidth / 1000) +
                0.2 * (cpu_load / 100) +
                0.2 * (memory_load / 100) +
                0.1 * (1 - q_value)
            )
            
            if return_explanation:
                step_explanations.append({
                    "from": current,
                    "to": next_hop,
                    "latency": latency,
                    "bandwidth": bandwidth,
                    "cpu_load": cpu_load,
                    "memory_load": memory_load,
                    "q_value": q_value,
                    "hop_cost": hop_cost
                })
            
            total_cost += hop_cost
        
        if return_explanation:
            return total_cost, step_explanations
        return total_cost

    def update_route_metrics(self):
        for node in devices:
            if node != self.node_id:
                try:
                    cpu_load = float(get_device_load(node))
                    memory_info = get_device_memory(node)
                    network_load = get_network_load(node)
                    
                    self.node_loads[node] = {
                        'cpu': cpu_load,
                        'memory': memory_info['used_percent'],
                        'network': network_load
                    }
                except Exception as e:
                    self.logger.warning(f"Metric update failed for {node}: {e}")
                
                if node in self.olsr_node.topology[self.node_id]:
                    try:
                        rtt = get_ping_time(self.node_id, node)
                        bw = measure_bandwidth(self.node_id, node)
                        self.link_loads[(self.node_id, node)] = {
                            'latency': rtt,
                            'bandwidth': bw
                        }
                    except Exception as e:
                        self.logger.warning(f"Link metric update failed for {node}: {e}")

    def select_best_route(self, source, destination):
        if destination in self.active_routes:
            active_route = self.active_routes[destination]
            if time.time() - active_route['last_used'] < 30:
                if self.is_route_healthy(active_route['path']):
                    return active_route['path']
        
        routes = self.find_multiple_routes(source, destination)
        if not routes:
            return None
        
        best_route = routes[0]
        self.active_routes[destination] = {
            'path': best_route,
            'metrics': {},
            'last_used': time.time()
        }
        
        self.backup_routes[destination] = routes[1:]
        
        self.route_history.append({
            'timestamp': time.time(),
            'source': source,
            'destination': destination,
            'path': best_route,
            'metrics': self.node_loads.copy()
        })
        
        if len(self.route_history) > self.max_history:
            self.route_history.pop(0)
        
        return best_route

    def is_route_healthy(self, path):
        if not path or len(path) < 2:
            return False
        for i in range(len(path) - 1):
            current = path[i]
            next_hop = path[i+1]
            
            link_metrics = self.link_loads.get((current, next_hop), {})
            if link_metrics.get('latency', 1000) > 500:
                return False
            node_metrics = self.node_loads.get(next_hop, {})
            if node_metrics.get('cpu', 100) > 90:
                return False
            if node_metrics.get('memory', 100) > 90:
                return False
        return True

    def handle_route_failure(self, failed_path):
        destination = failed_path[-1]
        if destination in self.active_routes:
            del self.active_routes[destination]
        
        if destination in self.backup_routes and self.backup_routes[destination]:
            backup_route = self.backup_routes[destination].pop(0)
            if self.is_route_healthy(backup_route):
                self.active_routes[destination] = {
                    'path': backup_route,
                    'metrics': {},
                    'last_used': time.time()
                }
                self.logger.info(f"Switched to backup route for {destination}: {backup_route}")
                return backup_route
        
        return self.select_best_route(self.node_id, destination)

    def get_route_statistics(self):
        stats = {
            'total_routes': len(self.route_history),
            'active_routes': len(self.active_routes),
            'backup_routes': sum(len(routes) for routes in self.backup_routes.values()),
            'average_path_length': 0,
            'most_used_nodes': {},
            'route_success_rate': 0
        }
        
        if self.route_history:
            path_lengths = [len(r['path']) for r in self.route_history]
            stats['average_path_length'] = sum(path_lengths) / len(path_lengths)
            node_usage = {}
            for route in self.route_history:
                for node in route['path']:
                    node_usage[node] = node_usage.get(node, 0) + 1
            stats['most_used_nodes'] = dict(sorted(
                node_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        
        return stats

#############################################################################
# Enhanced System Manager
#############################################################################
class EnhancedSystemManager:
    def __init__(self, node_id):
        self.node_id = node_id
        self.ip = devices[node_id]["ip"]
        self.username = devices[node_id]["username"]
        self.password = devices[node_id]["password"]
        
        self.metrics = {
            'cpu': [],
            'memory': [],
            'network': [],
            'disk': [],
            'temperature': []
        }
        self.metric_window = 3600
        
        self.ssh_connections = {}
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.blacklist = set()
        
        self.active_transfers = {}
        self.transfer_history = []
        
        self.setup_logging()
        self.initialize_monitoring()
        
    def setup_logging(self):
        self.logger = logging.getLogger(f"System-{self.node_id}")
        fh = logging.FileHandler(f"system_{self.node_id}.log")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def initialize_monitoring(self):
        Thread(target=self.monitor_system_metrics, daemon=True).start()
        Thread(target=self.monitor_network_health, daemon=True).start()
        Thread(target=self.cleanup_old_data, daemon=True).start()

    def monitor_network_health(self):
        while True:
            # Örneğin ağ istatistikleri ölçülebilir.
            time.sleep(10)

    def get_system_metrics(self):
        """Raspberry Pi veya benzeri bir cihaza uygun şekilde sistem metrikleri alır."""
        try:
            cpu_temp_str = os.popen("vcgencmd measure_temp").read().replace("temp=","").replace("'C\n","")
            try:
                cpu_temp = float(cpu_temp_str)
            except ValueError:
                cpu_temp = random.uniform(30, 60)
            
            cpu_load = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'load': cpu_load,
                    'temperature': cpu_temp
                },
                'memory': {
                    'used_percent': memory.percent,
                    'available': memory.available,
                    'swap_used': swap.used
                },
                'disk': {
                    'used_percent': disk.percent,
                    'free': disk.free
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Metric collection error: {e}")
            return None

    def monitor_system_metrics(self):
        while True:
            try:
                metrics = self.get_system_metrics()
                if metrics:
                    current_time = time.time()
                    self.metrics['cpu'].append({'timestamp': current_time, 'value': metrics['cpu']})
                    self.metrics['memory'].append({'timestamp': current_time, 'value': metrics['memory']})
                    self.metrics['disk'].append({'timestamp': current_time, 'value': metrics['disk']})
                    self.metrics['temperature'].append({'timestamp': current_time, 'value': metrics['cpu']['temperature']})
                    self.metrics['network'].append({'timestamp': current_time, 'value': metrics['network']})
                    
                    self.check_critical_conditions(metrics)
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(30)

    def check_critical_conditions(self, metrics):
        warnings = []
        if metrics['cpu']['load'] > 80:
            warnings.append(f"High CPU usage: {metrics['cpu']['load']}%")
        if metrics['cpu']['temperature'] > 70:
            warnings.append(f"High CPU temperature: {metrics['cpu']['temperature']}°C")
        if metrics['memory']['used_percent'] > 85:
            warnings.append(f"High memory usage: {metrics['memory']['used_percent']}%")
        if metrics['disk']['used_percent'] > 90:
            warnings.append(f"High disk usage: {metrics['disk']['used_percent']}%")
        
        if warnings:
            self.logger.warning("Critical conditions detected: " + ", ".join(warnings))
            self.take_corrective_action(warnings)

    def take_corrective_action(self, warnings):
        for warning in warnings:
            if "CPU" in warning:
                self.optimize_cpu_usage()
            elif "memory" in warning:
                self.optimize_memory_usage()
            elif "disk" in warning:
                self.cleanup_disk_space()

    def optimize_cpu_usage(self):
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                if proc.info['cpu_percent'] > 50:
                    processes.append(proc.info)
            if processes:
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
                pid = processes[0]['pid']
                os.system(f"renice +10 -p {pid}")
        except Exception as e:
            self.logger.error(f"CPU optimization error: {e}")

    def optimize_memory_usage(self):
        try:
            os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
            non_critical_services = ["bluetooth", "cups", "avahi-daemon"]
            for service in non_critical_services:
                os.system(f"systemctl stop {service}")
        except Exception as e:
            self.logger.error(f"Memory optimization error: {e}")

    def cleanup_disk_space(self):
        try:
            os.system("journalctl --vacuum-time=2d")
            os.system("rm -rf /tmp/*")
            for old_file in glob.glob("/home/pi/transfer_temp_*"):
                if time.time() - os.path.getctime(old_file) > 86400:
                    os.remove(old_file)
        except Exception as e:
            self.logger.error(f"Disk cleanup error: {e}")

    def secure_file_transfer(self, source_path, dest_node, dest_path):
        transfer_id = str(uuid.uuid4())
        try:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")
            if dest_node in self.blacklist:
                raise SecurityError(f"Destination node {dest_node} is blacklisted")
            
            checksum = self.calculate_checksum(source_path)
            self.active_transfers[transfer_id] = {
                'source': source_path,
                'destination': f"{dest_node}:{dest_path}",
                'start_time': time.time(),
                'status': 'in_progress',
                'checksum': checksum
            }
            
            success = scp_upload(source_path, dest_path, dest_node)
            if success:
                remote_checksum = self.get_remote_checksum(dest_node, dest_path)
                if checksum != remote_checksum:
                    raise IntegrityError("Checksum verification failed")
                
                self.active_transfers[transfer_id]['status'] = 'completed'
                self.transfer_history.append(self.active_transfers[transfer_id])
            return success
        except Exception as e:
            self.logger.error(f"File transfer error: {e}")
            self.active_transfers[transfer_id]['status'] = 'failed'
            self.active_transfers[transfer_id]['error'] = str(e)
            return False
        finally:
            if transfer_id in self.active_transfers:
                if self.active_transfers[transfer_id]['status'] == 'completed':
                    del self.active_transfers[transfer_id]

    def calculate_checksum(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_remote_checksum(self, node, file_path):
        try:
            ssh = self.get_ssh_connection(node)
            stdin, stdout, stderr = ssh.exec_command(f"sha256sum {file_path}")
            output = stdout.read().decode().strip()
            if output:
                return output.split()[0]
            return None
        except Exception as e:
            self.logger.error(f"Remote checksum error: {e}")
            return None

    def get_ssh_connection(self, node):
        if node in self.ssh_connections:
            try:
                self.ssh_connections[node].exec_command('echo 1')
                self.ssh_connections[node].last_used = time.time()
                return self.ssh_connections[node]
            except:
                del self.ssh_connections[node]
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                devices[node]["ip"],
                username=devices[node]["username"],
                password=devices[node]["password"],
                timeout=5
            )
            ssh.last_used = time.time()
            self.ssh_connections[node] = ssh
            return ssh
        except Exception as e:
            self.logger.error(f"SSH connection error to {node}: {e}")
            self.failed_attempts[node] = self.failed_attempts.get(node, 0) + 1
            if self.failed_attempts[node] >= self.max_failed_attempts:
                self.blacklist.add(node)
                self.logger.warning(f"Node {node} blacklisted due to multiple failed attempts")
            raise

    def check_node_health(self, node):
        if node in self.blacklist:
            return False
        return True

    def cleanup_old_data(self):
        while True:
            try:
                current_time = time.time()
                for category in self.metrics:
                    self.metrics[category] = [
                        m for m in self.metrics[category]
                        if current_time - m['timestamp'] < self.metric_window
                    ]
                self.transfer_history = [
                    t for t in self.transfer_history
                    if current_time - t['start_time'] < 86400
                ]
                for node in list(self.ssh_connections.keys()):
                    last_used = getattr(self.ssh_connections[node], 'last_used', 0)
                    if current_time - last_used > 300:
                        self.ssh_connections[node].close()
                        del self.ssh_connections[node]
                time.sleep(3600)
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                time.sleep(3600)

#############################################################################
# Network Manager
#############################################################################
class NetworkManager:
    def __init__(self, node_id):
        self.node_id = node_id
        self.system_manager = EnhancedSystemManager(node_id)
        self.sarsa_agent = EnhancedSarsaAgent(node_id)
        self.olsr_node = EnhancedOLSRNode(node_id)
        self.router = EnhancedRouter(node_id, self.olsr_node, self.sarsa_agent)
        
        self.setup_logging()
        self.initialize_components()
        
    def setup_logging(self):
        self.logger = logging.getLogger(f"NetworkManager-{self.node_id}")
        handler = logging.FileHandler(f"network_manager_{self.node_id}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def initialize_components(self):
        self.olsr_node.start()
        self.logger.info("OLSR protocol started")
        
    def transfer_file(self, source_path: str, destination_node: str):
        try:
            route = self.router.select_best_route(self.node_id, destination_node)
            if not route:
                raise Exception("No valid route found")
            self.logger.info(f"Selected route: {route}")
            
            current_path = source_path
            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]
                if not self.system_manager.check_node_health(next_node):
                    new_route = self.router.handle_route_failure(route)
                    if not new_route:
                        raise Exception("No alternative route available")
                    route = new_route
                    continue
                
                temp_path = f"/home/{devices[next_node]['username']}/transfer_temp"
                success = self.system_manager.secure_file_transfer(
                    current_path, next_node, temp_path
                )
                if not success:
                    raise Exception(f"Transfer failed at {next_node}")
                
                reward = self.calculate_transfer_reward(current_node, next_node)
                self.sarsa_agent.update_q_value(
                    (current_node, destination_node),
                    next_node,
                    reward,
                    (next_node, destination_node),
                    route[i+2] if i+2 < len(route) else None
                )
                current_path = temp_path
            
            self.logger.info(f"File transfer completed to {destination_node}")
            return True
        except Exception as e:
            self.logger.error(f"File transfer failed: {e}")
            return False

    def calculate_transfer_reward(self, current_node: str, next_node: str) -> float:
        try:
            metrics = self.system_manager.get_system_metrics()
            if not metrics:
                return 0
            link_quality = self.olsr_node.calculate_link_quality(next_node)
            reward = (
                0.4 * link_quality +
                0.3 * (1 - metrics['cpu']['load']/100) +
                0.3 * (1 - metrics['memory']['used_percent']/100)
            )
            return reward * 10
        except Exception as e:
            self.logger.error(f"Reward calculation error: {e}")
            return -1

    def cleanup(self):
        try:
            self.olsr_node.stop()
            self.system_manager.cleanup_old_data()
            self.sarsa_agent.save_q_table()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

#############################################################################
# GUI Application
#############################################################################
class EnhancedApp:
    def __init__(self, root, network_manager):
        self.root = root
        self.root.title(f"Enhanced Network Manager - {network_manager.node_id}")
        self.network_manager = network_manager
        
        self.setup_ui()
        
    def setup_ui(self):
        # File Selection
        self.frame_file = tk.LabelFrame(self.root, text="File Transfer")
        self.frame_file.pack(padx=10, pady=5, fill="x")
        
        self.btn_select = tk.Button(
            self.frame_file, text="Select File", command=self.select_file
        )
        self.btn_select.pack(pady=5)
        
        self.lbl_file = tk.Label(self.frame_file, text="No file selected")
        self.lbl_file.pack(pady=5)
        
        # Network Status
        self.frame_status = tk.LabelFrame(self.root, text="Network Status")
        self.frame_status.pack(padx=10, pady=5, fill="x")
        
        self.txt_status = tk.Text(self.frame_status, height=5, width=50)
        self.txt_status.pack(padx=5, pady=5)
        
        # Actions
        self.frame_actions = tk.LabelFrame(self.root, text="Actions")
        self.frame_actions.pack(padx=10, pady=5, fill="x")
        
        self.btn_send = tk.Button(
            self.frame_actions,
            text="Send to pi104",
            command=self.send_file
        )
        self.btn_send.pack(pady=5)
        
        # Logs
        self.frame_logs = tk.LabelFrame(self.root, text="Logs")
        self.frame_logs.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.txt_log = tk.Text(self.frame_logs, height=10, width=50)
        self.txt_log.pack(padx=5, pady=5, fill="both", expand=True)
        
        self.selected_file = None
        self.update_status()

    def select_file(self):
        f = filedialog.askopenfilename(
            title="Select File to Transfer",
            filetypes=[("All Files", "*.*")]
        )
        if f:
            self.selected_file = f
            self.lbl_file.config(text=f"Selected: {os.path.basename(f)}")
            self.log(f"File selected: {f}")

    def send_file(self):
        if not self.selected_file:
            self.log("Please select a file first")
            return
        self.log("Starting file transfer...")
        success = self.network_manager.transfer_file(self.selected_file, "pi104")
        if success:
            self.log("File transfer completed successfully")
        else:
            self.log("File transfer failed")
        self.update_status()

    def update_status(self):
        """Ağ durumunu güncelle"""
        try:
            metrics = self.network_manager.system_manager.get_system_metrics()
            if not metrics:
                self.root.after(5000, self.update_status)
                return
            status = (
                f"CPU Load: {metrics['cpu']['load']:.1f}%\n"
                f"Memory: {metrics['memory']['used_percent']:.1f}%\n"
                f"Active Routes: {len(self.network_manager.router.active_routes)}\n"
                f"Network Status: {self.get_network_status()}"
            )
            self.txt_status.delete(1.0, tk.END)
            self.txt_status.insert(tk.END, status)
            self.root.after(5000, self.update_status)
        except Exception as e:
            self.log(f"Status update error: {e}")

    def get_network_status(self) -> str:
        try:
            active_nodes = 0
            for node in graph_neighbors[self.network_manager.node_id]:
                try:
                    ssh_conn = self.network_manager.system_manager.get_ssh_connection(node)
                    if ssh_conn:
                        active_nodes += 1
                except:
                    pass
            return f"Online: {active_nodes}/{len(graph_neighbors[self.network_manager.node_id])}"
        except:
            return "Status Unknown"
            
    def log(self, message: str):
        self.txt_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.txt_log.see(tk.END)

#############################################################################
# Main Function
#############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node", 
        type=str, 
        required=True,
        help="Node ID (pi100/pi101/pi102/pi103/pi104)"
    )
    args = parser.parse_args()
    
    if args.node not in devices:
        print(f"Invalid node ID: {args.node}")
        return
    
    network_manager = NetworkManager(args.node)
    
    # GUI sadece pi100 üzerinde çalışır, diğer nodelar background thread
    if args.node == "pi100":
        root = tk.Tk()
        app = EnhancedApp(root, network_manager)
        root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, network_manager))
        root.mainloop()
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            network_manager.cleanup()

def on_closing(root, network_manager):
    network_manager.cleanup()
    root.destroy()

if __name__ == "__main__":
    main()
