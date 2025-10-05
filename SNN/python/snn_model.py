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

    def __init__(self, host: str = "127.0.0.1", port: int = 9999, input_size: int = 18, N_h: int = 300,
                 N_c: int = 100) -> None:
        # ---------- User parameters ----------
        self.HOST = host
        self.PORT = port

        self.OBS_SIZE    = input_size # e.g., 16 features + reward + error → we’ll read reward separately
        self.ACTION_SIZE = input_size         # one action per DoF
        self.dt          = 20*ms      # controls the speed of the simulation.
        self.eta_r       = 0.0020     # learning rate for reward-modulated plasticity
        self.eta_e       = 0.0010     # learning rate for error-modulated plasticity
        self.tau_e       = 200*ms     # eligibility decay
        # ------------------------------------
    
        # Brian performance hint
        prefs.codegen.target = "numpy"
    
        # ====== Build network ======
        self.N_in  = (self.OBS_SIZE - 2) * 3         # last obs element is reward (RPE baseline on Python)
        self.N_h   = N_h                       # tweak as needed (200–800 for CPU real-time)
        self.N_c   = N_c                       # second layer to mimic brain structure: motor cortex + input -> cerebellum -> added to output
        self.N_out = self.ACTION_SIZE * 3      # each action takes input from three neurons for stability

        # ====== LIF (input/hidden/output) ======
        # Membrane dynamics constants
        self.tau_m   = 20 * ms   # membrane time constant (how quickly voltage leaks back to rest)
        self.v_rest  = 0.0       # resting membrane potential (baseline voltage when no input)
        self.v_th    = 0.50      # spike threshold (voltage level that triggers a spike)
        self.v_reset = 0.0       # reset potential (voltage after a spike fires)
        self.refrac  = 2 * ms    # absolute refractory period (time neuron cannot spike again)

        # LIF dynamics: v decays toward v_rest with time constant tau_m,
        # but increases with input current I. Integration stops during refractory.
        self.eqs = """
        dv/dt = (-(v - v_rest) + I)/tau_m : 1 (unless refractory)
        I : 1
        v_rest : 1
        tau_m : second
        """

        # ====== AdEx-like (cerebellum) ======
        # Adaptation dynamics parameters
        self.tau_c_m  = 10 * ms   # cerebellum membrane time constant (shorter = faster response)
        self.v_th_c   = 0.35      # spike initiation threshold (where exponential term kicks in)
        self.deltaT   = 0.05      # slope factor (sharpness of spike onset; smaller = sharper)
        self.a        = 0.01      # subthreshold adaptation strength (how much w builds during depolarization)
        self.tau_w    = 200 * ms  # adaptation time constant (how quickly w decays back to 0)
        self.refrac_c = 1 * ms    # separate refractory for the cerebellum layer

        # AdEx dynamics
        self.eqs_c = """
        dv/dt = (-(v - v_rest) + deltaT*exp((v - v_t)/deltaT) - w + I)/tau_c_m : 1 (unless refractory)
        dw/dt = (a*(v - v_rest) - w)/tau_w : 1
        I : 1
        v_rest : 1
        v_t : 1
        deltaT : 1
        a : 1
        tau_c_m : second
        tau_w : second
        """
    

        self.G_in  = NeuronGroup(self.N_in, self.eqs,threshold=f'v>{self.v_th}', reset=f'v={self.v_reset}',refractory=self.refrac, method='euler', name='G_in')
        self.G_h   = NeuronGroup(self.N_h, self.eqs,threshold=f'v>{self.v_th}', reset=f'v={self.v_reset}',refractory=self.refrac, method='euler', name='G_h')
        self.G_c   = NeuronGroup(self.N_c, self.eqs_c,threshold=f'v>{self.v_th_c}', reset=f'v={self.v_reset}',refractory=self.refrac_c, method='euler', name='G_c')
        self.G_out = NeuronGroup(self.N_out, self.eqs,threshold=f'v>{self.v_th}', reset=f'v={self.v_reset}',refractory=self.refrac, method='euler', name='G_out')

        # brian2 doesn't handle "ms" inside f-strings well so parameters need to be defined outside.
        # ====== Assign parameters to groups ======
        for g in [self.G_in, self.G_h, self.G_out]:
            g.v_rest = self.v_rest
            g.tau_m  = self.tau_m

        self.G_c.v_rest  = self.v_rest
        self.G_c.tau_c_m = self.tau_c_m
        self.G_c.v_t     = self.v_th_c
        self.G_c.deltaT  = self.deltaT
        self.G_c.a       = self.a
        self.G_c.tau_w   = self.tau_w

        # Connections for the hidden layer
        self.Sinh  = Synapses(self.G_in, self.G_h, model='w_syn:1', on_pre='v_post += w_syn', name='Sinh')
        self.Shh  = Synapses(self.G_h, self.G_h, model='w_syn:1', on_pre='v_post += w_syn', name='Shh')
        self.Shout= Synapses(self.G_h, self.G_out, model=( # Different parameters need to be defined on seperate lines for brian2
                                                         'w_syn:1\n'
                                                         'elig:1'      # eligibility trace for R-STDP (decays in run_regularly)
                                                         ), on_pre=('v_post += w_syn\n'
                                                                   'elig += 0.01'
                                                                   ), on_post='elig -= 0.01', name='Shout')
        # Note: e increments are a simple pair-based surrogate (A+/-) for eligibility

        # Copying the brain event of efference copy
        # In the brain it helps with predicting consequences of actions
        self.South = Synapses(self.G_out, self.G_h, model='w_syn:1', on_pre='v_post += w_syn', name='South')

        # Connections for the cerebellum
        self.Sinc  = Synapses(self.G_in, self.G_c, model=(
                                                        'w_syn:1\n'
                                                        'elig:1'
                                                        ),on_pre=(
                                                                 'v_post += w_syn\n' 
                                                                 'elig += 0.01'
                                                                 ), name='Sinc')
        self.Shc   = Synapses(self.G_h,  self.G_c, model='w_syn:1',on_pre='v_post += w_syn', name='Shc')
        self.Scout = Synapses(self.G_c,  self.G_out, model=(
                                                          'w_syn:1\n'
                                                          'elig:1'
                                                          ), on_pre=(
                                                                    'v_post -= w_syn\n'
                                                                    'elig += 0.01'
                                                                    ), on_post='elig -= 0.01', name='Scout')

        # ---------- Connectivity ----------
        # Feedforward & cerebellar projections
        self.Sinc.connect(p=0.20)                                # in  → cerebellum
        self.Shc.connect(p=0.05)                                 # hid → cerebellum
        self.Scout.connect(p=0.10)                               # cer → out

        # Cortex path
        self.Sinh.connect(p=0.20)                                # in  → hid
        self.Shh.connect(p=0.20, condition='i!=j')               # hid → hid (no self loops)
        self.Shout.connect(p=0.40)                               # hid → out

        # Delays to the spikes are added to keep the network stable
        self.Sinh.delay  = '1*ms + 2*ms*rand()'
        self.Shh.delay   = '1*ms + 2*ms*rand()'
        self.Shout.delay = '1*ms + 2*ms*rand()'
        self.Sinc.delay  = '1*ms + 2*ms*rand()'
        self.Shc.delay   = '1*ms + 2*ms*rand()'
        self.Scout.delay = '1*ms + 2*ms*rand()'

        # ---------- Initial weights ----------
        # in → hid (mixed sign; moderate drive)
        self.Sinh.w_syn  = '0.6 * randn() / sqrt(N_pre)'
        
        # hid → hid (recurrent very small for stability)
        self.Shh.w_syn   = '0.1 * randn() / sqrt(N_pre)'
        
        # hid → out (mixed sign; trainable readout)
        self.Shout.w_syn = '0.5 * randn() / sqrt(N_pre)'
        
        # in → cerebellum (small)
        self.Sinc.w_syn  = '0.4 * randn() / sqrt(N_pre)'
        
        # hid → cerebellum (keep weak and mostly excitatory at start)
        self.Shc.w_syn   = '0.05 * randn() / sqrt(N_pre)'
        
        # cerebellum → out (initialize as NON-NEGATIVE since Scout is inhibitory and negative weights would be excitatory)
        self.Scout.w_syn = 'abs(0.08 * randn() / sqrt(N_pre))'

        # ---------- Reward and error ----------
        # Reward-modulated plasticity parameters
        self.wmin_r, self.wmax_r = -1.5, 1.5
        self.delta_rpe = 0.0      # will be updated each step from Unity reward - baseline

        # Eligibility decay + reward-modulated weight update each dt
        self.Shout.namespace.update({
            'delta': 0.0,
            'tau_e': self.tau_e,
            'eta_r': self.eta_r,
            'wmin_r': self.wmin_r,
            'wmax_r': self.wmax_r,
        })
        self.Shout.run_regularly(
            f"""
            elig *= exp(-dt/tau_e)
            w_syn += eta_r * delta * elig
            w_syn = clip(w_syn, wmin_r, wmax_r)
            """,
            dt=self.dt
        )

        # Error-modulated plasticity parameters
        self.wmin_e, self.wmax_e = -1, 1 

        # Eligibility decay + error signal update each dt
        self.Scout.namespace.update({
            'gamma': 0.0,
            'tau_e': self.tau_e,
            'eta_e': self.eta_e,
            'wmin_e': self.wmin_e,
            'wmax_e': self.wmax_e,
        })
        self.Scout.run_regularly(
            """
            elig *= exp(-dt/tau_e)
            w_syn += eta_e * gamma * elig
            w_syn = clip(w_syn, wmin_e, wmax_e)
            """,
            dt=self.dt
        )

        # Monitors for output spikes
        self.M_out = SpikeMonitor(self.G_out, name='M_out')

        # Compose network explicitly so we can call net.run(dt)
        self.net = Network(self.G_in, self.G_h, self.G_c, self.G_out, self.Sinh, self.Shh, self.Sinc, self.Shc, self.Scout, self.Shout, self.M_out)
        defaultclock.dt = 1*ms

        # ====== Readout: EMA firing rate → continuous actions ======
        self.rate_ema = np.zeros(self.N_out, dtype=np.float32)
        self.alpha = 0.2  # EMA factor per control tick (0..1)

        self.prev_counts = np.zeros(self.N_out, dtype=np.int64)

        # ====== Baseline for RPE ======
        self.baseline_r = 0.0
        self.bl_alpha_r = 0.05  # Learning speed

        # ====== Baseline for error ======
        self.baseline_e = 0.0
        self.bl_alpha_e = 0.10 # Learning speed for cerebellum

    def decode_actions(self):
        """Compute actions in [-1,1] from output spike counts in the last tick."""
        counts = self.M_out.count[:]        # total spikes since start
        spk = counts - self.prev_counts     # spikes this tick
        self.prev_counts = counts.copy()
        self.rate_ema = (1 - self.alpha)*self.rate_ema + self.alpha*spk.astype(np.float32)
        a = np.tanh(0.5 * self.rate_ema)
        return a


    # ====== Input injection ======
    def set_input_from_obs(self, obs_vec):
        """
        obs_vec length N_in; map to input currents I.
        We use a simple affine map + clamp (replace with better normalization if needed).
        """
        x = np.tile(np.asarray(obs_vec, dtype=np.float32), 3)

        # Normalize with tanh to preserve detail
        x = (x - np.median(x)) / (np.percentile(x,75)-np.percentile(x,25)+1e-6)
        print("x: ", x)

        self.G_in.I[:]  = x
        self.G_h.I[:]   = 0.1                # hidden bias should also be able to spike
        self.G_out.I[:] = 0               # small bias; out should rely on synaptic drive

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
        spk = self.M_out.count[:] - self.prev_counts
        print("out_spikes_this_tick:", int(spk.sum()))
        return self.decode_actions().tolist()
    
    def get_weights(self):
        return [self.Sinh.w_syn[:], self.Shh.w_syn[:], self.Shout.w_syn[:], self.Sinc.w_syn[:], self.Scout.w_syn[:]]
