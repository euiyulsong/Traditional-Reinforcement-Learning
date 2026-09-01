# DQN / Q-Learning / SARSA / Policy Gradient 비교 실험

## 1. 실험 목적

동일한 강화학습 환경에서 다음 4가지 알고리즘을 학습하고 성능을 비교한다.

* **Q-Learning**
* **SARSA**
* **DQN**
* **Policy Gradient (REINFORCE)**

## 2. 환경

오픈소스 **Gymnasium의 `FrozenLake-v1`** 환경을 사용했다.

```text
State: 16개
Action: 4개
Reward: 목표 지점 도착 시 1
is_slippery=True
```

미끄러짐을 활성화하여 action을 선택해도 예상과 다른 방향으로 이동할 수 있는 stochastic environment로 구성했다.

## 3. 알고리즘 비교

| Algorithm  | 학습 대상             | 방식         | 핵심                             |
| ---------- | ----------------- | ---------- | ------------------------------ |
| Q-Learning | Q-Table           | Off-policy | 다음 state의 최대 Q값 사용             |
| SARSA      | Q-Table           | On-policy  | 실제 선택한 다음 action의 Q값 사용        |
| DQN        | Neural Q-function | Off-policy | Q-Learning을 Neural Network로 근사 |
| REINFORCE  | Policy Network    | On-policy  | Policy를 직접 최적화                 |

### Q-Learning

```text
target = r + γ max Q(s', a)
```

현재 행동 정책과 관계없이 **가장 좋은 다음 action**을 기준으로 학습한다.

### SARSA

```text
target = r + γ Q(s', a')
```

현재 policy가 **실제로 선택한 다음 action**을 이용해 학습한다.

### DQN

```text
State
  ↓
Neural Network
  ↓
Q(s,a)
```

Q-table 대신 Neural Network로 Q-function을 근사하고 다음을 추가했다.

* Experience Replay
* Target Network
* ε-greedy exploration

### Policy Gradient

```text
State
  ↓
Policy Network
  ↓
π(a|s)
```

Q-value를 학습하지 않고 **action probability를 직접 학습**했다.

REINFORCE의 objective는 대략 다음과 같다.

```text
loss = -log π(a|s) × Return
```

## 4. 평가

모든 모델을 동일한 `FrozenLake-v1`에서 학습한 뒤 **500 episode의 success rate**로 평가한다.

또한 training reward의 moving average를 그려 **학습 속도와 안정성**도 비교한다.

## 5. 핵심 관계

```text
Value-based
    │
    ├── Q-Learning (Off-policy)
    │       │
    │       └── + Neural Network
    │           + Replay Buffer
    │           + Target Network
    │                ↓
    │               DQN
    │
    └── SARSA (On-policy)


Policy-based
    │
    └── Policy Gradient
            ↓
        π(a|s)를 직접 학습
```

`FrozenLake`처럼 state가 매우 작은 환경에서는 **Q-Learning/SARSA의 Q-table만으로 충분**할 수 있다. DQN의 장점은 state space가 커져 Q-table을 직접 유지하기 어려워질 때 더 뚜렷해진다.
