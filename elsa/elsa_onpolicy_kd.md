# ELSA + On-Policy Knowledge Distillation

## ELSA 알고리즘

$$\min_{w,\,z}\; \mathcal{L}(w) \quad \text{s.t.}\quad w = z,\quad z \in \mathcal{S}_s$$

---

**Algorithm 1: ELSA with On-Policy KD**

---

**Input:** Dense model $\pi_T$ (teacher), prompt dataset $\mathcal{D}$, sparsity $s$, penalty $\lambda$

**Initialize:** $w \leftarrow \theta_\text{dense}$, $z \leftarrow \text{Proj}_{\mathcal{S}_s}(w)$, $u \leftarrow 0$

---

**for** $t = 1, \ldots, T$ **do**

&emsp;1. Sample prompt $x \sim \mathcal{D}$

&emsp;2. Generate $y \sim \pi_w(\cdot \mid x)$ &emsp; *(student, no gradient)*

&emsp;3. Compute task loss $\mathcal{L}(w)$ &emsp; ← **On-Policy KD loss** (see below)

&emsp;4. **w-update** (proximal gradient via Adam):
$$g \leftarrow \nabla_w \mathcal{L}(w) + \lambda(w - z + u), \qquad w \leftarrow w - \eta\,\text{Adam}(g)$$

&emsp;5. **if** $t \bmod \texttt{interval} = 0$: &emsp; *(z, u update)*
$$z \leftarrow \text{Proj}_{\mathcal{S}_s}(w + u), \qquad u \leftarrow u + (w - z)$$

**end for**

**return** $\text{Proj}_{\mathcal{S}_s}(w)$

---

## On-Policy KD Loss

기존 ELSA의 task loss $\mathcal{L}(w)$는 NTP(next-token prediction)였습니다. 이를 on-policy KD loss로 대체합니다.

$$\mathcal{L}_\text{KD}(w) = \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_w(\cdot|x)}\!\left[\sum_{t} D\!\left(\pi_w(\cdot \mid y_{<t}, x) \;\|\; \pi_T(\cdot \mid y_{<t}, x)\right)\right]$$

**NTP와의 차이:** 학습 시퀀스 $y$를 외부 데이터에서 가져오지 않고, **현재 student $\pi_w$가 직접 생성**합니다. 이를 통해 sparse 모델의 실제 분포와 학습 분포 간의 mismatch(covariate shift)를 제거합니다.

## KL Divergence 방향

$D(\cdot \| \cdot)$으로 KL divergence를 사용합니다. 방향에 따라 성질이 다릅니다.

| | Forward KL: $D(\pi_T \| \pi_w)$ | Reverse KL: $D(\pi_w \| \pi_T)$ |
|---|---|---|
| 최소화 대상 | student가 teacher 분포를 커버 | student가 teacher의 최빈값에 집중 |
| 성질 | mean-seeking | mode-seeking |

현재는 **Reverse KL**을 사용합니다.

$$\mathcal{L}_\text{KD}(w) = \mathbb{E}\!\left[\sum_t \sum_v \pi_w(v) \log \frac{\pi_w(v)}{\pi_T(v)}\right]$$

계산 효율을 위해 teacher 기준 top-$k$ 토큰($k=50$)에 대해서만 KL을 계산합니다.