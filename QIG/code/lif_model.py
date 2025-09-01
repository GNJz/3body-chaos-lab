import numpy as np

class LIFNeuron:
    def __init__(self, dt=1e-3, tau=20e-3, v_rest=0.0, v_reset=0.0,
                 v_th_base=1.0, refractory_ms=0.0):
        self.dt = dt
        self.tau = tau
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th_base = v_th_base
        self.v = v_rest

        self.refractory_steps = int(round(refractory_ms / (dt * 1e3))) if refractory_ms > 0 else 0
        self._ref_count = 0

    def reset(self):
        self.v = self.v_reset
        self._ref_count = 0

    def step(self, I, v_th):
        # 불응기 처리
        if self._ref_count > 0:
            self._ref_count -= 1
            return self.v_reset, False

        dv = (-(self.v - self.v_rest) + I) * (self.dt / self.tau)
        self.v += dv

        if self.v >= v_th:
            self.v = self.v_reset
            if self.refractory_steps > 0:
                self._ref_count = self.refractory_steps
            return self.v, True
        return self.v, False


def dynamic_threshold(t_array, v_th_base=1.0, alpha=1.0):
    # 결정적 임계값 함수: V_th(t) = v_th_base * exp(-alpha * t)
    return v_th_base * np.exp(-alpha * t_array)
