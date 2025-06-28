# Simulação Computacional do Vórtice Maligno
# Consulte o notebook para execução interativa
# Autor: Humberto Marambaia Junior

import networkx as nx
import numpy as np
import pandas as pd

NUM_AGENTS = 10000
AVG_DEGREE = 10
REWIRING_PROB = 0.1
TIMESTEPS = 60

G = nx.watts_strogatz_graph(n=NUM_AGENTS, k=AVG_DEGREE, p=REWIRING_PROB)
np.random.seed(42)

belief = np.random.rand(NUM_AGENTS) * 0.2
identity = np.random.rand(NUM_AGENTS)
education_level = np.random.randint(1, 6, NUM_AGENTS)
media_exposure = np.random.rand(NUM_AGENTS)
visual_confidence = np.random.rand(NUM_AGENTS)

alpha = 0.1
beta = 0.65
k_logistic = 4.3

belief_over_time = []

for t in range(TIMESTEPS):
    new_belief = belief.copy()
    for i in range(NUM_AGENTS):
        neighbors = list(G.neighbors(i))
        if not neighbors:
            continue
        avg_neighbor_belief = np.mean([belief[j] for j in neighbors])
        delta_belief = (
            0.4 * (avg_neighbor_belief - belief[i]) +
            0.35 * (avg_neighbor_belief - belief[i]) +
            0.25 * (avg_neighbor_belief - belief[i])
        )
        identity_effect = 0.2 * np.exp(2.5 * identity[i])
        media_effect = media_exposure[i] * visual_confidence[i]
        update = alpha * delta_belief + media_effect
        if update > 0:
            update *= (1 - beta)
        else:
            update *= (1 + beta)
        x = belief[i] + update
        x0 = 0.2 + (education_level[i] * 0.05)
        adoption_prob = 1 / (1 + np.exp(-k_logistic * (x - x0)))
        if np.random.rand() < adoption_prob:
            new_belief[i] = min(1.0, x + identity_effect * 0.01)
    belief = new_belief
    belief_over_time.append(np.mean(belief))

df = pd.DataFrame({'Mes': np.arange(TIMESTEPS), 'CrencaMedia': belief_over_time})
df.to_csv("dados_resultado.csv", index=False)