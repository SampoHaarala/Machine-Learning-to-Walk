#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brian2 SNN server with reward-modulated STDP (R-STDP) and a simple TCP protocol.

Protocol:
- On connect: send one JSON line (handshake) with obs_size, action_size, dt, etc.
- Then per step: Unity sends CSV "x1,x2,...,x_obs, reward"
- Server replies CSV "a1,...,a_action" (continuous actions in [-1,1])
"""

import json, socket, threading
import numpy as np
from brian2 import (ms, second, NeuronGroup, Synapses, Network, defaultclock,
                    run, SpikeMonitor, StateMonitor, prefs)

class SpikingNetwork:

    def __init__(self, host: str = "127.0.0.1", port: int = 9999, N_h: int = 300,
                 N_c: int = 100) -> None:
        # ---------- User parameters ----------
        self.HOST = host
        self.PORT = port

        self.OBS_SIZE    = 23       # e.g., 21 features + reward + error → we’ll read reward separately
        self.ACTION_SIZE = 21       # one action per DoF
        self.dt          = 20*ms    # controls the speed of the simulation.
        self.eta_r       = 0.005    # learning rate for reward-modulated plasticity
        self.eta_e       = 0.003    # learning rate for error-modulated plasticity
        self.tau_e       = 200*ms   # eligibility decay
        # ------------------------------------
    
        # Brian performance hint
        prefs.codegen.target = "numpy"
    
        # ====== Build network ======
        self.N_in  = self.OBS_SIZE - 1         # last obs element is reward (RPE baseline on Python)
        self.N_h   = N_h                       # tweak as needed (200–800 for CPU real-time)
        self.N_c   = N_c                       # second layer to mimick brain structure: motor cortex + input -> cerebellum -> added to output
        self.N_out = self.ACTION_SIZE * 3      # each action takes input from three neurons for stability
    
        # LIF (input/hidden/output)
        self.tau_m    = 20*ms   # membrane time constant (how quickly voltage leaks back to rest)
        self.tau_c_m  = 10*ms   # cerebellum membrane time constant (shorter = faster response)
        self.v_rest   = 0.0     # resting membrane potential (baseline voltage when no input)
        self.v_th     = 1.0     # spike threshold (voltage level that triggers a spike)
        self.v_reset  = 0.0     # reset potential (voltage after a spike fires)
        self.refrac   = 2*ms    # absolute refractory period (time neuron cannot spike again)
    
        # LIF dynamics: v decays toward v_rest with time constant tau_m,
        # but increases with input current I. Integration stops during refractory.
        self.eqs      = f"""
        dv/dt = (-(v - {self.v_rest}) + I)/{self.tau_m} : 1 (unless refractory)
        I : 1
        """
    
        # AdEx-like (cerebellum)
        self.v_t      = 0.5      # spike initiation threshold (where exponential term kicks in)
        self.deltaT   = 0.05     # slope factor (controls sharpness of spike onset; smaller = sharper)
        self.a        = 0.01     # subthreshold adaptation strength (how much w builds during depolarization)
        self.tau_w    = 200*ms   # adaptation time constant (how quickly w decays back to 0)
        self.refrac_c = 1*ms     # seperate parameters for the cerebellum layer
        self.eqs_c    = f"""
                        dv/dt = (-(v - {self.v_rest}) + {self.deltaT}*exp((v - {self.v_t})/{self.deltaT}) - w + I)/{self.tau_c_m} : 1 (unless refractory)
                        dw/dt = ({self.a}*(v - {self.v_rest}) - w)/{self.tau_w} : 1
                        I     : 1
                        w     : 1
                        """
    

        self.G_in  = NeuronGroup(self.N_in,  self.eqs,threshold=f'v>{self.v_th}', reset=f'v={self.v_reset}',refractory=self.refrac, method='euler', name='G_in')
        self.G_h   = NeuronGroup(self.N_h,   self.eqs,threshold=f'v>{self.v_th}', reset=f'v={self.v_reset}',refractory=self.refrac, method='euler', name='G_h')
        self.G_c   = NeuronGroup(self.N_c,   self.eqs_c,threshold=f'v>{self.v_th}', reset=f'v={self.v_reset}',refractory=self.refrac_c, method='euler', name='G_c')
        self.G_out = NeuronGroup(self.N_out, self.eqs,threshold=f'v>{self.v_th}', reset=f'v={self.v_reset}',refractory=self.refrac, method='euler', name='G_out')
    
        # Random input projection and recurrent/forward connections
        self.rng = np.random.default_rng(42)
            # Dense random weights, scaled
        def rand_w(shape, scale):
            return (rng.normal(0.0, 1.0, size=shape) / np.sqrt(shape[0])) * scale

        # Connections for the hidden layer
        self.Sinh  = Synapses(self.G_in,  self.G_h,  model='w:1', on_pre='v_post += w', name='Sinh')
        self.Shh  = Synapses(self.G_h,   self.G_h,  model='w:1', on_pre='v_post += w', name='Shh')
        self.Shout= Synapses(self.G_h,   self.G_out, model=(
            'w:1\n'
            'e:1'      # eligibility trace for R-STDP (decays in run_regularly)
        ), on_pre='v_post += w; e += 0.01', on_post='e -= 0.01', name='Shout')
        # Note: e increments are a simple pair-based surrogate (A+/-) for eligibility.

        # Connections for the cerebellum
        self.Sinc = Synapses(self.G_in, self.G_c, model='w:1; xi:1',on_pre='v_post += w; xi += 0.01', name='Sinc')
        self.Shc  = Synapses(self.G_h,  self.G_c, model='w:1',on_pre='v_post += w', name='Shc')
        self.Scout= Synapses(self.G_c,  self.G_out, model=(
            'w:1\n'
            'e:1'
        ), on_pre='v_post -= w; e += 0.01', on_post='e -= 0.01', name='Scout')

        self.Scout.connect(p=0.2)
        self.Shc.connect(p=0.2)
        self.Shc.w = '0.1*rand()'
        self.Sinh.connect(p=0.3); self.Shh.connect(p=0.05); self.Shout.connect(p=0.4)
        self.Sinh.w[:]   = rand_w(self.Sinh.w.shape,   0.5)
        self.Shh.w[:]   = rand_w(self.Shh.w.shape,   0.2)
        self.Shout.w[:] = rand_w(self.Shout.w.shape, 0.3)

        # Reward-modulated plasticity parameters (names in namespace for live update)
        self.wmin, self.wmax = -1.5, 1.5
        self.delta_rpe = 0.0      # will be updated each step from Unity reward - baseline

        # Eligibility decay + reward-modulated weight update each dt
        self.Shout.run_regularly(
            f"""
            e *= exp(-dt/{self.tau_e})
            w += {self.eta_r} * delta * e
            w = clip(w, {self.wmin}, {self.wmax})
            """,
            dt=self.dt, namespace={'delta': 0.0}
        )
        # Eligibility decay + error signal update each dt
        self.Scout.run_regularly(
            f"""
            e *= exp(-dt/{self.tau_e})
            w += {self.eta_e} * gamma * e
            w = clip(w, {self.wmin}, {self.wmax})
            """,
            dt=self.dt, namespace={'gamma': 0.0}
        )

        # Monitors for output spikes
        self.M_out = SpikeMonitor(self.G_out, name='M_out')

        # Compose network explicitly so we can call net.run(dt)
        self.net = Network(self.G_in, self.G_h, self.G_c, self.G_out, self.Sinh, self.Shh, self.Sinc, self.Shc, self.Scout, self.Shout, self.M_out)
        defaultclock.dt = self.dt

        # ====== Readout: EMA firing rate → continuous actions ======
        self.rate_ema = np.zeros(self.N_out, dtype=np.float32)
        self.alpha = 0.2  # EMA factor per control tick (0..1)

        self.prev_counts = np.zeros(self.N_out, dtype=np.int64)

        # ====== Baseline for RPE ======
        self.baseline_r = 0.0
        self.bl_alpha_r = 0.01

        # ====== Baseline for error ======
        self.baseline_e = 0.0
        self.bl_alpha_e = 0.01

    def decode_actions(self):
        """Compute actions in [-1,1] from output spike counts in the last tick."""
        counts = self.M_out.count[:]      # total spikes since start
        spk = counts - self.prev_counts   # spikes in this tick
        prev_counts = counts.copy()
        rate_ema = (1 - self.alpha)*self.rate_ema + self.alpha*spk.astype(np.float32)
        # Scale & squash to [-1,1] (scale can be tuned)
        scale = 0.1
        a = np.tanh(scale * rate_ema)
        return a

    # ====== Input injection ======
    def set_input_from_obs(self, obs_vec):
        """
        obs_vec length N_in; map to input currents I.
        We use a simple affine map + clamp (replace with better normalization if needed).
        """
        x = np.asarray(obs_vec, dtype=np.float32)
        # Normalize with tanh to preserve detail
        scale = 5
        x = np.tanh(x / scale)

        self.G_in.I[:]  = x      # Send input signal only to the input layer
        self.G_h.I[:]   = 0.0    # Other layers only receive information through synapses
        self.G_out.I[:] = 0.0

    def set_reward(self, r):
        """Update global delta (RPE) and expose it to Brian namespace."""
        self.baseline_r = (1 - self.bl_alpha_r)*self.baseline_r + self.bl_alpha_r*r
        self.delta_rpe = float(r - self.baseline_r)
        # Update the namespace value used by Shout.run_regularly
        self.Shout.namespace['delta'] = self.delta_rpe

    def set_error(self, e):
        """Update global gamma (error) and expose it to Brian namespace."""
        self.baseline_e = (1 - self.bl_alpha_e)*self.baseline_e + self.bl_alpha_e*e
        gamma_error = float(e - self.baseline_e)
        # Update the namespace value used by Shout.run_regularly
        self.Scout.namespace['gamma'] = gamma_error

    def step(self, obs: list[float]) -> list[float]:
        """Run the model for one tick."""
        self.set_input_from_obs(obs)
        self.net.run(self.dt)
        return self.decode_actions().tolist()
    
    def get_weights(self):
        return [self.Sinh.w[:], self.Shh.w[:], self.Shout.w[:], self.Sinc.w[:], self.Scout.w[:]]
