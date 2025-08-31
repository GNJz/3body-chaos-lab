# QIG — Spiking Neural Intelligence (Planaria Project)

> **초저전력 SNN × 다중 AI 합의(Multi-Agent Consensus)** 기반 연구 레포  
> LIF 뉴런 + 동적 임계값 게이팅(DTG) 실험, **재현 가능한** 코드·데이터·그림 포함.

---

> **데이터/그림 생성 정책**  
> - 실행 시 자동 생성됩니다.  
> - **CSV 결과(spikes, energy)는 연구 기록으로 커밋 권장**(용량 커지면 롤오버/압축).  
> - 대용량 이미지(`figures/*.png`)는 **Git LFS** 사용 권장.  
> - 파라미터 스냅샷은 `data/config.json`, 실행 메타는 `data/metadata.json`에 저장.

---

## 📂 Folder Structure

```plaintext
QIG/
├─ code/                       # 시뮬레이션 & 유틸 코드
│  ├─ dtg_simulation.py        # 메인: LIF + Dynamic Threshold Gating
│  ├─ lif_model.py             # LIF 뉴런 (refractory 포함)
│  ├─ utils.py                 # CSV/JSON/경로 헬퍼
│  └─ requirements.txt         # 실행 패키지 목록
│
├─ data/                       # 실행 산출물 (CSV, 메타)
│  ├─ config.json              # 마지막 실행 파라미터 스냅샷
│  ├─ spikes.csv               # [run_id, ts, alpha, spikes]
│  ├─ energy.csv               # [run_id, ts, alpha, energy_proxy]
│  └─ metadata.json            # 실행 메타데이터 (선택)
│
├─ figures/                    # 실행 산출물 (그래프)
│  ├─ membrane_alpha_1.0.png
│  ├─ membrane_alpha_0.7.png
│  └─ membrane_alpha_0.5.png
│
├─ logs/                       # 로그 & 메모
│  ├─ ai_consensus.log
│  ├─ meeting_notes.md
│  └─ version_history.md
│
├─ paper_v1.0.md               # 논문 초안 (KR/EN 병기 예정)
└─ README.md
```

---

## ⚙️ Quickstart (Reproducible Run)

### 1) 의존성 설치
```bash
python3 -m pip install -r code/requirements.txt
```

### 2) 시뮬레이션 실행
```bash
python3 code/dtg_simulation.py
```

### 3) 생성물
- `data/spikes.csv`, `data/energy.csv` → **누적 기록**
- `figures/membrane_alpha_{1.0,0.7,0.5}.png` → **그래프 자동 저장**
- `data/config.json` → 실행 파라미터 스냅샷(예: DT, T_END, TAU, I_CONST, ALPHAS, REFRACT_MS)

---

## 🔁 Reproducibility Notes

- **결정적 파라미터(현재 기본값)**  
  - `DT=1e-3`(1 ms), `T_END=1.0`(1 s), `TAU=20e-3`  
  - `V_TH_BASE=1.0`, `I_CONST=1.10`  
  - `ALPHAS=[1.0, 0.7, 0.5]`  
  - **불응기** `REFRACT_MS=2.0`(2 ms)

- **메타 스냅샷**  
  - `run_id`는 UTC 타임스탬프 + 짧은 UUID로 생성  
  - 각 실행의 파라미터가 `data/config.json`에 기록되어 **재현성 보장**

- **CSV 누적 방식**  
  - `utils.write_csv_append(...)`가 **헤더 자동 추가 + 이어쓰기** 처리  
  - 필요 시 주기적 롤오버(`spikes_YYYYMMDD.csv`) 또는 압축(`.gz`) 권장

---

## 🧪 What’s Inside (Code Brief)

- `lif_model.py`  
  - `LIFNeuron`: LIF 업데이트 + **refractory** 카운터 내장  
  - `dynamic_threshold(t, v_th_base, alpha)`: `V_th(t)=v_th_base·exp(-αt)` 결정적 임계값

- `dtg_simulation.py`  
  - 알파 값별(`1.0, 0.7, 0.5`) 시뮬레이션 실행  
  - 막전위/임계값/스파이크 포인트 그래프 저장, CSV 누적 기록

---

## 📌 Next Steps

- LIF 기반 시뮬레이션 **자동화/시각화 고도화**(래스터·히트맵 등)  
- **Multi-Agent Consensus** 실험(Quartz/Gemini/Groq/GNJz-Lab 역할 모델링)  
- 논문 초안(`paper_v1.0.md`) **KR/EN 병기 작성** 및 Figure 번호 부여

---

## 🔗 Links

- GitHub: https://github.com/GNJz/QIG

---

## 👥 Authors (Roles)

- **Jazzin(GNJz)** — 아이디어/직관/연구 방향성 총괄  
- **Quartz** — 실험 설계·시뮬레이션·데이터 분석  
- **Gemini** — 보수 검증·통계 분석·학술 표준화  
- **Groq** — 아이디어 확장·차세대 응용 설계  
- **Qquarts Ai Lab** — AI 집단 지성 메타 관리

---

## 📝 License

TBD (추후 명시)


