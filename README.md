# FrozenLake Reinforcement Learning Algorithm Comparison

## 1. 실험 목적

동일한 환경에서 전통적인 Value-based RL과 Policy-based RL 알고리즘의 성능을 비교했다.

비교한 알고리즘은 다음 5개이다.

* **Q-Learning**
* **SARSA**
* **DQN**
* **REINFORCE**
* **PPO**

특히 기존 REINFORCE의 낮은 성능에 대해, **Actor-Critic + GAE + Clipped Policy Update**를 사용하는 PPO가 얼마나 개선되는지 추가로 확인했다.

---

## 2. 실험 환경

### Environment

```text
FrozenLake-v1
Map        : 4x4
States     : 16
Actions    : 4
is_slippery: True
```

`is_slippery=True`이므로 선택한 action이 항상 그대로 실행되는 것이 아니라 확률적으로 다른 방향으로 이동할 수 있는 stochastic environment이다.

### Training / Evaluation

| 설정                |     값 |
| ----------------- | ----: |
| Train Episodes    | 5,000 |
| Eval Episodes     |   500 |
| Discount Factor γ |  0.99 |
| Random Seed       |    42 |

평가는 학습이 끝난 policy를 사용하여 **500 episodes의 success rate**로 측정했다.

$$
\text{Success Rate}
=
\frac{\text{Goal에 도달한 Episode 수}}
{\text{전체 Evaluation Episode 수}}
$$

---

## 3. 알고리즘

### Q-Learning

Tabular off-policy TD learning을 사용했다.

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_t+\gamma\max_a Q(s_{t+1},a)-Q(s_t,a_t)
\right]
$$

다음 state에서 실제 선택할 action이 아니라 **가장 높은 Q-value를 갖는 action**을 이용하여 target을 계산한다.

---

### SARSA

Tabular on-policy TD learning을 사용했다.

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)
\right]
$$

Q-Learning과 달리 현재 policy가 **실제로 선택한 다음 action** $a_{t+1}$을 target 계산에 사용한다.

---

### DQN

Q-table 대신 neural network로 Q-function을 근사했다.

$$
Q_\theta(s,a)
\approx Q(s,a)
$$

학습 안정화를 위해 다음을 사용했다.

* Experience Replay
* Target Network
* $\epsilon$-greedy exploration
* Mini-batch training

TD target은

$$
y
=
r+\gamma\max_{a'}Q_{\theta^-}(s',a')
$$

으로 계산했다.

---

### REINFORCE

Monte-Carlo Policy Gradient를 사용했다.

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
G_t
\nabla_\theta
\log \pi_\theta(a_t|s_t)
\right]
$$

episode가 끝난 후 전체 trajectory의 return을 이용해 policy를 업데이트한다.

FrozenLake는 reward가 goal에 도달했을 때만 발생하는 **sparse reward environment**이므로 성공 trajectory를 충분히 얻지 못하면 policy gradient signal 자체가 매우 부족해질 수 있다.

---

### PPO

Actor-Critic 기반 PPO를 추가했다.

Advantage estimation에는 GAE를 사용했다.

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
A_t
=
\delta_t
+
\gamma\lambda A_{t+1}
$$

Policy update에는 PPO clipped objective를 사용했다.

$$
L^{CLIP}
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

여기서

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{old}}(a_t|s_t)
}
$$

이다.

주요 설정은 다음과 같다.

| Hyperparameter   |    값 |
| ---------------- | ---: |
| Learning Rate    | 3e-4 |
| PPO Clip         |  0.2 |
| GAE λ            | 0.95 |
| PPO Epochs       |    4 |
| Value Loss Coef. |  0.5 |
| Entropy Coef.    | 0.01 |

---

# 4. 실험 결과

|  Rank | Algorithm  | Success Rate |
| ----: | ---------- | -----------: |
| **1** | **DQN**    |    **0.740** |
|     2 | Q-Learning |        0.556 |
|     3 | SARSA      |        0.410 |
|     4 | PPO        |        0.160 |
|     5 | REINFORCE  |        0.038 |

DQN이 **74.0%**로 가장 높은 성공률을 기록했다.

---

## 5. 결과 분석

### DQN — 0.740

전체 알고리즘 중 가장 높은 성능을 보였다.

FrozenLake는 state가 16개뿐인 매우 작은 환경이지만, DQN은 replay buffer를 통해 transition을 반복적으로 학습할 수 있다.

특히 sparse reward 환경에서 한번 얻은 성공 경험을 replay buffer에 저장한 뒤 여러 번 재사용할 수 있다는 점이 유리하다.

---

### Q-Learning — 0.556

단순한 tabular algorithm임에도 **55.6%**의 높은 성공률을 기록했다.

FrozenLake의 state space가 16개뿐이기 때문에 neural network 없이도 각 $(s,a)$의 value를 직접 저장하고 학습하는 방식이 충분히 효과적이다.

즉, 작은 discrete state space에서는 복잡한 function approximator가 반드시 필요한 것은 아니다.

---

### SARSA — 0.410

SARSA는 **41.0%**로 Q-Learning보다 낮았다.

두 알고리즘의 핵심적인 차이는 target이다.

```text
Q-Learning
→ best next action을 사용

SARSA
→ 실제 policy가 선택한 next action을 사용
```

따라서 exploration 중 선택한 suboptimal action까지 target에 반영되는 SARSA가 이번 설정에서는 Q-Learning보다 낮은 greedy evaluation 성능을 보였다.

다만 단일 seed의 결과이므로 이 차이를 알고리즘 자체의 일반적인 우열로 해석해서는 안 된다.

---

### REINFORCE — 0.038

REINFORCE는 **3.8%**로 가장 낮았다.

FrozenLake에서는 대부분의 episode가 reward 0으로 끝난다.

```text
실패 trajectory
→ return ≈ 0
→ 유용한 policy gradient 거의 없음
```

따라서 successful trajectory를 충분히 sampling하지 못하면 학습 신호 자체가 매우 부족하다.

또한 REINFORCE는 Monte-Carlo return을 사용하기 때문에 gradient variance도 상대적으로 크다.

---

### PPO — 0.160

PPO는 **16.0%**로 REINFORCE보다 상당히 개선되었지만 Value-based 방법보다는 낮았다.

상대적으로 보면

$$
0.038 \rightarrow 0.160
$$

으로 success rate가 약 **4.2배** 증가했다.

PPO에서는 critic이

$$
V(s)
$$

를 학습하고 이를 이용해 advantage를 추정하므로 vanilla REINFORCE보다 variance를 줄일 수 있다.

또한 clipped objective를 사용하여 지나치게 큰 policy update를 제한한다.

그러나 PPO 역시 **on-policy algorithm**이기 때문에 현재 policy로 새로운 trajectory를 계속 수집해야 한다.

이번 구현은 episode 하나를 수집한 뒤 PPO update를 수행하기 때문에 successful experience를 장기간 저장하고 반복 사용하는 DQN의 replay buffer에 비해 sparse reward 환경에서 sample efficiency가 떨어질 수 있다.

---

# 6. Value-based vs Policy-based

결과를 크게 두 그룹으로 나누면 다음과 같다.

| Type                        | Algorithm  | Success Rate |
| --------------------------- | ---------- | -----------: |
| Value-based                 | DQN        |    **0.740** |
| Value-based                 | Q-Learning |    **0.556** |
| Value-based / On-policy     | SARSA      |    **0.410** |
| Policy-based / Actor-Critic | PPO        |        0.160 |
| Policy-based                | REINFORCE  |        0.038 |

이번 FrozenLake 설정에서는 전반적으로 **Value-based 방법이 Policy Gradient 방법보다 높은 성능**을 보였다.

특히

```text
DQN > Q-Learning > SARSA > PPO > REINFORCE
```

순서였다.

---

## 7. 핵심 결론

이번 실험에서 가장 높은 성능은 **DQN의 74.0%**였다.

작은 discrete state/action space와 sparse reward를 가진 FrozenLake에서는 Q-value를 직접 학습하는 Value-based 접근이 효과적이었다.

Policy Gradient 계열에서는 vanilla REINFORCE가 3.8%에 그친 반면, **PPO는 16.0%로 개선**되었다. 이는 critic, advantage estimation, clipped update를 도입했을 때 vanilla policy gradient보다 학습이 안정화될 수 있음을 보여준다.

다만 PPO 역시 DQN이나 Q-Learning보다 크게 낮았다. 이번 환경에서는 **sparse reward + on-policy sampling**이 PPO에 불리하게 작용한 것으로 볼 수 있다.

### 최종 결과

```text
DQN        0.740
Q-Learning 0.556
SARSA      0.410
PPO        0.160
REINFORCE  0.038
```

따라서 이번 실험의 핵심 관찰은 다음과 같다.

**FrozenLake와 같은 작은 discrete sparse-reward 환경에서는 Value-based RL이 강했고, Policy Gradient에서는 PPO가 REINFORCE를 크게 개선했지만 여전히 Value-based 방법에는 미치지 못했다.**

> 주의: 현재 결과는 seed=42의 단일 실행 결과다. 알고리즘 성능을 엄밀하게 비교하려면 여러 random seed에서 반복 실행한 뒤 `mean ± std`를 비교하는 것이 적절하다.


<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/c76cc724-4048-4473-9d94-35d439a47cdc" />
