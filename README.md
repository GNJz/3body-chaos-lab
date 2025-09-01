# PHAM 3-Body Chaos Lab — Progress Report v1.0
*(2025-08-30 기준)*

---

## **1. 실험실 개요**
- **프로젝트명:** PHAM 3-Body Chaos Lab
- **목적:**
  - 3체 문제(Three-Body Problem)의 혼돈(Chaos) 특성 시각화
  - 에너지 보존성(Drift)과 리야푸노프 지수(Lyapunov Exponent) 기반 안정성 분석
  - PHAM 세계관용 카오스 시뮬레이터 기반 구축
- **현황:**
  ✅ 시뮬레이션 코어 완성  
  ✅ Lyapunov 지수 측정 기능 추가  
  ✅ dt 스윕(정밀도 vs 안정성) 자동화  
  ✅ Lyapunov Heatmap + Contour 시각화 구축  

---

## **2. 주요 기능**

| 기능 | 설명 | 출력물 |
|------|------|--------|
| **3D 궤적 시뮬레이션** | exp1, exp2, exp3 초기조건별 3체 궤적 계산 | `figures/three_body3d_*.png` |
| **에너지 드리프트 분석** | 시뮬레이션 동안 총 에너지 변화율 시각화 | `figures/energy_drift_*.png` |
| **리야푸노프 지수 측정** | 작은 교란 → 궤적 발산 속도 → 안정성 지표 | JSON 결과 + Heatmap |
| **dt 스윕 자동화** | 시간 스텝 크기별 오차 및 카오스 민감도 분석 | `figures/dt_scan_*` |
| **Lyapunov Heatmap/Contour** | δ₀(교란 크기)와 τ(재규격화 간격)별 안정성 시각화 | `figures/lyap_heatmap_*` |

---

## **3. 실험 예시**

### **① exp2 (중간 혼돈)** — `t=40`, `dt=0.005`

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

#### 해석
- Lyapunov ≈ **0.154** → 장기적으로 혼돈 상태
- 에너지 드리프트 < **10⁻⁷** → 수치적 안정성 확보

---

### **② Lyapunov Heatmap — exp2**
- **δ₀ 범위:** 1e-10 ~ 1e-7  
- **τ 범위:** 0.5 ~ 3.0  
- **결과:** 안정성 구간 vs 혼돈 구간 **명확히 분리 확인 완료**

---

## **4. 현재 상태**
✅ 실험실 코어 완성  
✅ 시각화 기능 완전 동작  
✅ dt / τ / δ₀ 파라미터 실험 가능  
⚡ PHAM 세계관과 연계 준비 완료  

---

## **5. 다음 단계 제안**

| 단계 | 목표 | 예상 산출물 |
|------|------|------------------------|
| **Step 1** | 초기조건 커스터마이즈 | exp4, exp5 추가 실험 |
| **Step 2** | Phase Diagram 생성 | 파라미터 전역 스윕 → 카오스 맵 |
| **Step 3** | PHAM 세계관 통합 | Kaos Map → 팜틀란티스 연결 시각화 |
| **Step 4** | 실험 매뉴얼 작성 | Markdown 매뉴얼 + 코드 가이드 |

---
