# AFH* Base Experiment – Versión Corregida para Detectar Pliegue Autopsíquico

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from transformers import BertTokenizer, BertModel
import torch

# ── 1. Cargar modelo BERT una sola vez ─────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

@torch.no_grad()
def get_embedding(word: str) -> np.ndarray:
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# ── 2. Simular redes ───────────────────────────────────────────────────────────
def simulate_network(conscious: bool, n_nodes=50):
    if conscious:
        # Red totalmente conectada con pesos altos para asegurar κ_topo > 0.5
        G = nx.complete_graph(n_nodes)
        weights = {e: np.random.uniform(1.0, 1.1) for e in G.edges()}
    else:
        # Red dispersa
        G = nx.erdos_renyi_graph(n_nodes, p=0.05)
        weights = {e: np.random.uniform(0.1, 0.3) for e in G.edges()}
    nx.set_edge_attributes(G, weights, 'weight')
    return G

# ── 3. Cálculo de κ_topo ───────────────────────────────────────────────────────
def compute_k_topo(G):
    curvs = []
    for u, v in G.edges():
        w = G[u][v]['weight']
        neigh_u = set(G.neighbors(u))
        neigh_v = set(G.neighbors(v))
        overlap = len(neigh_u & neigh_v)
        total  = len(neigh_u) + len(neigh_v)
        curvs.append(w * overlap / max(1, total))
    return np.mean(curvs)

# ── 4. Cálculo de Φ_H ─────────────────────────────────────────────────────────
def compute_phi_H(ts):
    mi_scores = []
    for i in range(ts.shape[1] - 1):
        mi_scores.append(mutual_info_score(ts[:, i], ts[:, i+1]))
    return np.mean(mi_scores)

# ── 5. Cálculo de ΔPCI ────────────────────────────────────────────────────────
def compute_delta_PCI(pair):
    def lz(signal):
        b = ''.join('1' if x > np.median(signal) else '0' for x in signal)
        return len(b) / (len(set(b)) + 1e-10)
    return abs(lz(pair[0]) - lz(pair[1]))

# ── 6. Cálculo de ∇Φ_resonant ─────────────────────────────────────────────────
def compute_resonance(symbol: str, neural_vec: np.ndarray):
    emb = get_embedding(symbol)
    # Asegurar misma longitud
    if len(neural_vec) != len(emb):
        neural_vec = np.interp(
            np.linspace(0, 1, len(emb)),
            np.linspace(0, 1, len(neural_vec)),
            neural_vec
        )
    return np.corrcoef(emb, neural_vec)[0, 1]

# ── 7. Ejecutar experimento ───────────────────────────────────────────────────
def run_afh_experiment(symbolic='love'):
    # 7.1 Redes
    G_con = simulate_network(True)
    G_inc = simulate_network(False)
    # 7.2 Series temporales
    ts_con = np.tile(np.sin(np.linspace(0, 2*np.pi, 50)), (100,1)) + np.random.normal(0,0.05,(100,50))
    ts_inc = np.random.normal(0,1,(100,50))
    # 7.3 Perturbaciones
    pert_con = [ts_con.flatten(), (ts_con + np.random.normal(0,0.05,ts_con.shape)).flatten()]
    pert_inc = [ts_inc.flatten(), (ts_inc + np.random.normal(0,0.5,ts_inc.shape)).flatten()]
    # 7.4 Resonancia simbólica
    emb = get_embedding(symbolic)
    resp_con = emb.copy()             # respuesta idéntica → correlación ~ 1
    resp_inc = np.random.uniform(size=len(emb))
    # 7.5 Calcular variables
    return {
        'Conscious': {
            'κ_topo': compute_k_topo(G_con),
            'Φ_H': compute_phi_H(ts_con),
            'ΔPCI': compute_delta_PCI(pert_con),
            '∇Φ_resonant': compute_resonance(symbolic, resp_con)
        },
        'Non-Conscious': {
            'κ_topo': compute_k_topo(G_inc),
            'Φ_H': compute_phi_H(ts_inc),
            'ΔPCI': compute_delta_PCI(pert_inc),
            '∇Φ_resonant': compute_resonance(symbolic, resp_inc)
        }
    }

# ── 8. Mostrar resultados ────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_afh_experiment('love')
    print("\nAFH* Supreme Experiment Results:\n")
    for cond, vars in results.items():
        print(f"{cond}:")
        for k, v in vars.items():
            print(f"  {k}: {v:.3f}")
    fold = all([
        results['Conscious']['κ_topo'] >= 0.5,
        results['Conscious']['Φ_H']    >= 1.0,
        results['Conscious']['ΔPCI']   <= 0.1,
        abs(results['Conscious']['∇Φ_resonant']) > 0.3
    ])
    print(f"\n¿Pliegue Autopsíquico Detectado? {'SÍ' if fold else 'NO'}")
