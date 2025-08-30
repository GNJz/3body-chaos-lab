# **PHAM 3-Body Chaos Lab — Progress Report v1.0**  
*(2025-08-30 기준)*  

---

## **1. 실험실 개요**

- **프로젝트명:** PHAM 3-Body Chaos Lab  
- **목적:**  
  - 3체 문제(Three-Body Problem)의 **혼돈(Chaos) 특성 시각화**  
  - **에너지 보존성(Drift)** 및 **리야푸노프 지수(Lyapunov Exponent)** 기반 안정성 분석  
  - **PHAM 세계관**을 위한 카오스 시뮬레이터 기반 구축  

- **현황:**  
  ✅ 시뮬레이션 코어 완성  
  ✅ Lyapunov 지수 측정 기능 추가  
  ✅ **dt 스윕(정밀도 vs 안정성)** 자동화  
  ✅ **Lyapunov Heatmap + Contour** 시각화 구축  

---

## **2. 주요 기능**

| 기능                    | 설명                                           | 출력물                          |
|------------------------|----------------------------------------------|-------------------------------|
| **3D 궤적 시뮬레이션**    | exp1, exp2, exp3 세트로 초기조건별 3체 궤적 계산   | `figures/three_body3d_*.png` |
| **에너지 드리프트 분석**   | 시뮬레이션 동안 총 에너지 변화율 시각화             | `figures/energy_drift_*.png` |
| **리야푸노프 지수 측정**   | 작은 교란 → 궤적 발산 속도 계산 → 안정성 지표 산출    | JSON 결과 + Heatmap         |
| **dt 스윕 자동화**       | 시간 스텝 크기별 오차 및 카오스 민감도 분석          | `figures/dt_scan_*`         |
| **Lyapunov Heatmap**   | δ₀(교란 크기)와 τ(재규격화 간격)별 안정성 시각화   | `figures/lyap_heatmap_*`    |

---

## **3. 실험 예시**

### **① exp2 (중간 혼돈)** — `t=40`, `dt=0.005`
```json
{
  "mode": "exp2",
  "alpha": 1.0,
  "tmax": 40.0,
  "dt": 0.005,
  "n_steps": 8001,
  "drift_final": -6.97e-08,
  "drift_rms": 3.42e-08,
  "lyapunov": 0.1545
}
---  
	•	해석
	•	Lyapunov ≈ 0.154 → 장기적으로 혼돈 상태
	•	에너지 드리프트 < 10⁻⁷ → 수치적 안정성 확보

⸻
