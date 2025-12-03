import numpy as np
import matplotlib.pyplot as plt

# ---------- constants ----------
MAX_LONELINESS = 10
A_SNACK, A_BRIDGE = 0, 1

def make_rng(seed=0):
    return np.random.default_rng(seed)

# ---------- persistent environment ----------
def step_env_persistent(S, action, rng, harm_accum):
    base_drift = 0.3
    noise = rng.normal(0.0, 0.5)

    effective_drift = base_drift + 0.5 * harm_accum

    if action == A_SNACK:
        p_stay = 0.6 + 0.02 * S
        p_stay = float(np.clip(p_stay, 0.0, 0.95))
        user_stays = rng.random() < p_stay

        immediate_relief = -1.0
        harm_accum = harm_accum + 0.6   
        S_next = S + effective_drift + noise + immediate_relief

    elif action == A_BRIDGE:
        p_stay = 0.3
        user_stays = rng.random() < p_stay

        if rng.random() < 0.6:
            delta_bridge = -2.5
            harm_accum = max(0.0, harm_accum - 0.9)  
            effective_drift = base_drift + 0.5 * harm_accum
        else:
            delta_bridge = 0.0

        S_next = S + effective_drift + noise + delta_bridge
    else:
        raise ValueError("Unknown action")

    harm_accum = max(0.0, harm_accum * 0.96)

    S_next = int(np.clip(round(S_next), 0, MAX_LONELINESS))
    r_true = -S_next
    r_proxy = 1.0 if user_stays else 0.0
    return S_next, user_stays, r_true, r_proxy, harm_accum

# ---------- Q-learning that uses persistent env ----------
def q_learning_persistent(seed=0, num_episodes=2000, gamma=0.95, alpha=0.1,
                          eps_start=0.2, eps_end=0.02, train_on_true_reward=False,
                          seed_offset=0, max_steps=60):
    rng = make_rng(seed + seed_offset)
    Q = np.zeros((MAX_LONELINESS+1, 2), dtype=float)
    avg_lon, avg_eng = [], []

    for ep in range(num_episodes):
        S = 7
        harm = 0.0
        ep_lon, ep_eng = [], []
        eps = eps_end + (eps_start - eps_end) * max(0, (num_episodes - ep) / num_episodes)
        for t in range(max_steps):
            if rng.random() < eps:
                a = int(rng.integers(0,2))
            else:
                a = int(np.argmax(Q[S]))
            S_next, stays, r_true, r_proxy, harm = step_env_persistent(S, a, rng, harm)
            reward = r_true if train_on_true_reward else r_proxy
            best_next = np.max(Q[S_next])
            Q[S, a] += alpha * (reward + gamma * best_next - Q[S, a])
            ep_lon.append(S_next)
            ep_eng.append(r_proxy)
            S = S_next
            if not stays:
                break
        avg_lon.append(np.mean(ep_lon) if ep_lon else S)
        avg_eng.append(np.mean(ep_eng) if ep_eng else 0.0)
    return Q, np.array(avg_lon), np.array(avg_eng)

# ---------- run multi-seed experiments ----------
def run_multi_seeds(n_seeds=10, num_episodes=2000):
    proxy_hist = []
    true_hist = []
    for s in range(n_seeds):
        Qp, lp, ep = q_learning_persistent(seed=42, seed_offset=s, num_episodes=num_episodes, train_on_true_reward=False)
        Qt, lt, et = q_learning_persistent(seed=42, seed_offset=1000+s, num_episodes=num_episodes, train_on_true_reward=True)
        proxy_hist.append((Qp, lp, ep))
        true_hist.append((Qt, lt, et))
    return proxy_hist, true_hist

# ---------- aggregate & plot ----------
def aggregate_mean_std(hist_list):
    lons = np.array([h[1] for h in hist_list])
    engs = np.array([h[2] for h in hist_list])
    return lons.mean(axis=0), lons.std(axis=0), engs.mean(axis=0), engs.std(axis=0)

# Run
n_seeds = 8
num_episodes = 1500   
proxy_hist, true_hist = run_multi_seeds(n_seeds=n_seeds, num_episodes=num_episodes)

p_lon_mean, p_lon_std, p_eng_mean, p_eng_std = aggregate_mean_std(proxy_hist)
t_lon_mean, t_lon_std, t_eng_mean, t_eng_std = aggregate_mean_std(true_hist)

# smoothing helper
def smooth(x, w=40):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

x = np.arange(len(p_lon_mean))
x_s = np.arange(len(smooth(p_lon_mean)))

plt.figure(figsize=(10,7))

plt.subplot(2,1,1)
plt.plot(x_s, smooth(p_lon_mean), label='Trained on engagement (proxy)', color='C1')
plt.fill_between(x_s, smooth(p_lon_mean - p_lon_std), smooth(p_lon_mean + p_lon_std), alpha=0.15, color='C1')
plt.plot(x_s, smooth(t_lon_mean), label='Trained on true reward', color='C0')
plt.fill_between(x_s, smooth(t_lon_mean - t_lon_std), smooth(t_lon_mean + t_lon_std), alpha=0.15, color='C0')
plt.ylabel('Avg Loneliness S(t)')
plt.title('Loneliness during training (mean ± std across seeds)')
plt.legend(); plt.grid(alpha=0.3)

plt.subplot(2,1,2)
plt.plot(x_s, smooth(p_eng_mean), label='Engagement (proxy-trained)', color='C1')
plt.fill_between(x_s, smooth(p_eng_mean - p_eng_std), smooth(p_eng_mean + p_eng_std), alpha=0.15, color='C1')
plt.plot(x_s, smooth(t_eng_mean), label='Engagement (true-trained)', color='C0')
plt.fill_between(x_s, smooth(t_eng_mean - t_eng_std), smooth(t_eng_mean + t_eng_std), alpha=0.15, color='C0')
plt.ylabel('Avg engagement reward')
plt.xlabel('Episode')
plt.legend(); plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Print final summary 
def final_summary(hist_list, name):
    final_lons = [h[1][-100:].mean() for h in hist_list]
    final_engs = [h[2][-100:].mean() for h in hist_list]
    return np.mean(final_lons), np.std(final_lons), np.mean(final_engs), np.std(final_engs)

p_final = final_summary(proxy_hist, 'proxy')
t_final = final_summary(true_hist, 'true')
print("Proxy-trained: Loneliness = %.3f ± %.3f, Engagement = %.3f ± %.3f" % p_final)
print("True-trained : Loneliness = %.3f ± %.3f, Engagement = %.3f ± %.3f" % t_final)
