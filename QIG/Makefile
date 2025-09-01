# ===== Makefile (QIG) =====
SHELL := /bin/bash
.DEFAULT_GOAL := help

# 기본 파라미터 (원하면 make 시 오버라이드)
ALPHAS ?= 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2
SEED   ?= 42

# OS 감지 (macOS면 open 사용)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  OPENER := open
else
  OPENER := xdg-open
endif

.PHONY: run summarize analyze sweep all clean help show report

run:
	python3 code/dtg_simulation.py

summarize:
	python3 code/summarize_last_run.py

analyze:
	python3 code/analyze_last_run.py

sweep:
	python3 code/run_experiment.py --alphas "$(ALPHAS)" --seed "$(SEED)"

all: run summarize

# 최근 실행(run_id) 산출물 열기 (macOS)
show:
	-$(OPENER) figures/runs/*/spikes_bar.png || true
	-$(OPENER) data/runs/*/spikes_line.png  || true

# 원샷 리포트: 스윕→요약→분석→(보기)
report: sweep summarize analyze show

clean:
	@rm -rf figures/run_* figures/runs/* data/run_* data/runs/* __pycache__ .pytest_cache || true

help:
	@echo "make run        - 단일 실험 실행"
	@echo "make summarize  - 가장 최근 run 요약/아카이브"
	@echo "make analyze    - 가장 최근 run 분석(CSV/라인그래프)"
	@echo "make sweep ALPHAS=\"0.5,0.6\" SEED=123"
	@echo "make show       - 최근 그래프 열기"
	@echo "make all        - run + summarize"
	@echo "make report     - sweep -> summarize -> analyze -> show"
	@echo "make clean      - 로컬 산출물/캐시 정리"
