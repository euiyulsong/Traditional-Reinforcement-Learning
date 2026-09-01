# RL 알고리즘 비교 실험 결과

## 실험 환경

* **Environment:** Gymnasium `FrozenLake-v1`
* **States:** 16
* **Actions:** 4
* **Stochastic:** `is_slippery=True`
* **평가:** 학습 완료 후 500 episodes의 성공률 비교

## 결과

| Algorithm  | Success Rate |
| ---------- | -----------: |
| **SARSA**  |    **0.740** |
| **DQN**    |    **0.740** |
| Q-Learning |        0.624 |
| REINFORCE  |        0.038 |

## 해석

### SARSA — 74.0%

가장 좋은 성능을 보였다.

```text
target = r + γ Q(s', a')
```

현재 policy가 실제로 선택할 다음 action까지 고려하는 **on-policy** 방식이다. `FrozenLake`가 미끄러지는 stochastic 환경이기 때문에 안정적인 정책을 학습한 것으로 볼 수 있다.

### DQN — 74.0%

SARSA와 동일하게 **74.0%**를 기록했다.

```text
Q-Table
   ↓ Neural Network로 근사
DQN
```

Q-Learning 계열이지만 Q-table 대신 Neural Network와 Replay Buffer, Target Network를 사용한다.

다만 FrozenLake는 state가 16개뿐이라 **DQN이 필요한 만큼 복잡한 문제는 아니다.**

### Q-Learning — 62.4%

SARSA/DQN보다 낮은 **62.4%**를 기록했다.

```text
target = r + γ max Q(s', a)
```

실제 다음 행동이 아니라 항상 최적의 다음 행동을 가정하는 **off-policy** 방식이다.

이번처럼 action 결과에 randomness가 있는 환경에서는 SARSA와 성능 차이가 발생할 수 있다.

### REINFORCE — 3.8%

성능이 매우 낮았다.

REINFORCE는 Q-value 대신 policy를 직접 학습한다.

```text
π(a|s)
```

하지만 reward가 **goal 도착 시에만 1인 sparse reward 환경**이라 대부분 episode에서 학습 신호를 거의 얻지 못한다.

따라서 이 결과만 보고 **Policy Gradient 자체가 나쁘다고 결론 내리면 안 된다.**

## 결론

```text
SARSA       74.0%  ███████████████
DQN         74.0%  ███████████████
Q-Learning  62.4%  ████████████
REINFORCE    3.8%  █
```

이번 실험에서는 **SARSA ≈ DQN > Q-Learning >>> REINFORCE** 순이었다.

핵심적으로 확인한 것은 다음과 같다.

* **Q-Learning:** Off-policy + Q-table
* **SARSA:** On-policy + Q-table
* **DQN:** Q-Learning을 Neural Network로 확장
* **REINFORCE:** Value가 아니라 Policy를 직접 학습
* 작은 discrete state에서는 Q-table만으로도 충분하다.
* Sparse reward에서는 vanilla REINFORCE가 학습하기 어렵다.

> 따라서 다음 실험은 `CartPole-v1`처럼 continuous state 환경에서 DQN과 Policy Gradient를 비교하면 Deep RL을 사용하는 이유가 더 명확하게 나타난다.
