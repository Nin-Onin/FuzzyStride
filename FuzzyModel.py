import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyModel:
    def __init__(self):
        self._build_variables()
        self._build_membership_functions()
        self._build_rules()
        self._build_control_system()

    def _build_variables(self):
        self.heart_rate      = ctrl.Antecedent(np.arange(100, 190.1, 0.1), 'heart_rate')
        self.pacing          = ctrl.Antecedent(np.arange(3.0,  9.1,  0.1), 'pacing')
        self.distance        = ctrl.Antecedent(np.arange(0,    42.1, 0.1), 'distance')
        self.training_status = ctrl.Consequent(np.arange(0,    1.01, 0.01), 'training_status')

    # Membership Functions
    def _build_membership_functions(self):
        hr = self.heart_rate
        hr['low']      = fuzz.trimf(hr.universe, [100, 120, 135])
        hr['moderate'] = fuzz.trimf(hr.universe, [130, 145, 165])
        hr['high']     = fuzz.trimf(hr.universe, [160, 175, 190])

        pc = self.pacing
        pc['fast']     = fuzz.trimf(pc.universe, [3.0, 3.8, 4.6])
        pc['moderate'] = fuzz.trimf(pc.universe, [4.4, 5.6, 6.8])
        pc['slow']     = fuzz.trimf(pc.universe, [6.4, 7.8, 9.0])

        dist = self.distance
        dist['short']  = fuzz.trimf(dist.universe, [0,  4,  10])
        dist['medium'] = fuzz.trimf(dist.universe, [8,  18, 28])
        dist['long']   = fuzz.trimf(dist.universe, [25, 35, 42])

        ts = self.training_status
        ts['undertraining'] = fuzz.trimf(ts.universe, [0.0, 0.2, 0.5])
        ts['normal']        = fuzz.trimf(ts.universe, [0.3, 0.55, 0.8])
        ts['overtraining']  = fuzz.trimf(ts.universe, [0.6, 0.8, 1.0])

    # Rule Base
    def _build_rules(self):
        dist = self.distance
        pc   = self.pacing
        hr   = self.heart_rate
        ts   = self.training_status

        self.rules = [
            # --- Short distance ---
            ctrl.Rule(dist['short'] & pc['slow']     & hr['low'],      ts['undertraining']),
            ctrl.Rule(dist['short'] & pc['slow']     & hr['moderate'], ts['undertraining']),
            ctrl.Rule(dist['short'] & pc['slow']     & hr['high'],     ts['normal']),
            ctrl.Rule(dist['short'] & pc['moderate'] & hr['low'],      ts['undertraining']),
            ctrl.Rule(dist['short'] & pc['moderate'] & hr['moderate'], ts['normal']),
            ctrl.Rule(dist['short'] & pc['moderate'] & hr['high'],     ts['normal']),
            ctrl.Rule(dist['short'] & pc['fast']     & hr['low'],      ts['normal']),
            ctrl.Rule(dist['short'] & pc['fast']     & hr['moderate'], ts['normal']),
            ctrl.Rule(dist['short'] & pc['fast']     & hr['high'],     ts['overtraining']),
            # --- Medium distance ---
            ctrl.Rule(dist['medium'] & pc['slow']     & hr['low'],      ts['undertraining']),
            ctrl.Rule(dist['medium'] & pc['slow']     & hr['moderate'], ts['normal']),
            ctrl.Rule(dist['medium'] & pc['slow']     & hr['high'],     ts['normal']),
            ctrl.Rule(dist['medium'] & pc['moderate'] & hr['low'],      ts['normal']),
            ctrl.Rule(dist['medium'] & pc['moderate'] & hr['moderate'], ts['normal']),
            ctrl.Rule(dist['medium'] & pc['moderate'] & hr['high'],     ts['overtraining']),
            ctrl.Rule(dist['medium'] & pc['fast']     & hr['low'],      ts['normal']),
            ctrl.Rule(dist['medium'] & pc['fast']     & hr['moderate'], ts['overtraining']),
            ctrl.Rule(dist['medium'] & pc['fast']     & hr['high'],     ts['overtraining']),
            # --- Long distance ---
            ctrl.Rule(dist['long'] & pc['slow']     & hr['low'],      ts['normal']),
            ctrl.Rule(dist['long'] & pc['slow']     & hr['moderate'], ts['normal']),
            ctrl.Rule(dist['long'] & pc['slow']     & hr['high'],     ts['overtraining']),
            ctrl.Rule(dist['long'] & pc['moderate'] & hr['low'],      ts['normal']),
            ctrl.Rule(dist['long'] & pc['moderate'] & hr['moderate'], ts['overtraining']),
            ctrl.Rule(dist['long'] & pc['moderate'] & hr['high'],     ts['overtraining']),
            ctrl.Rule(dist['long'] & pc['fast']     & hr['low'],      ts['normal']),
            ctrl.Rule(dist['long'] & pc['fast']     & hr['moderate'], ts['overtraining']),
            ctrl.Rule(dist['long'] & pc['fast']     & hr['high'],     ts['overtraining']),
        ]

        self.rule_definitions = [
            ('short',  'slow',     'low',      'undertraining'),
            ('short',  'slow',     'moderate', 'undertraining'),
            ('short',  'slow',     'high',     'normal'),
            ('short',  'moderate', 'low',      'undertraining'),
            ('short',  'moderate', 'moderate', 'normal'),
            ('short',  'moderate', 'high',     'normal'),
            ('short',  'fast',     'low',      'normal'),
            ('short',  'fast',     'moderate', 'normal'),
            ('short',  'fast',     'high',     'overtraining'),
            ('medium', 'slow',     'low',      'undertraining'),
            ('medium', 'slow',     'moderate', 'normal'),
            ('medium', 'slow',     'high',     'normal'),
            ('medium', 'moderate', 'low',      'normal'),
            ('medium', 'moderate', 'moderate', 'normal'),
            ('medium', 'moderate', 'high',     'overtraining'),
            ('medium', 'fast',     'low',      'normal'),
            ('medium', 'fast',     'moderate', 'overtraining'),
            ('medium', 'fast',     'high',     'overtraining'),
            ('long',   'slow',     'low',      'normal'),
            ('long',   'slow',     'moderate', 'normal'),
            ('long',   'slow',     'high',     'overtraining'),
            ('long',   'moderate', 'low',      'normal'),
            ('long',   'moderate', 'moderate', 'overtraining'),
            ('long',   'moderate', 'high',     'overtraining'),
            ('long',   'fast',     'low',      'normal'),
            ('long',   'fast',     'moderate', 'overtraining'),
            ('long',   'fast',     'high',     'overtraining'),
        ]
    # Control System
    def _build_control_system(self):
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation     = ctrl.ControlSystemSimulation(self.control_system)