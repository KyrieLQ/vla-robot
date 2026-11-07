"""Read-only simulator labels. Never expose this object to the policy service.

Finger-pad contact is a grasp proxy, not proof of a stable grasp. Snapshots here
are diagnostic state only, not a complete restorable controller snapshot.
"""
import copy
import numpy as np


class OracleRecorder:
    def __init__(self, wrapper):
        self.env = wrapper.env
        e = self.env
        self.names = sorted(e.obj_body_id)
        self.ids = [e.obj_body_id[n] for n in self.names]
        self.geom_sets = []
        for name in self.names:
            obj = e.get_object(name)
            self.geom_sets.append(set(e.sim.model.geom_name2id(g) for g in obj.contact_geoms))
        gripper = e.robots[0].gripper
        self.pads = [set(e.sim.model.geom_name2id(g) for g in gripper.important_geoms[key])
                     for key in ('left_fingerpad', 'right_fingerpad')]
        self.goal = copy.deepcopy(e.parsed_problem['goal_state'])
        assert len(self.goal) == 1 and len(self.goal[0]) == 3
        assert self.goal[0][0].lower() == 'on', self.goal
        assert self.goal[0][1] in self.names and self.goal[0][2] in self.names
        self.meta = {
            'object_names': self.names, 'movable_objects': sorted(e.objects_dict),
            'goal_predicates': self.goal, 'target_object': self.goal[0][1],
            'destination_object': self.goal[0][2], 'quaternion_order': 'wxyz',
            'geom_names': [e.sim.model.geom_id2name(i) for i in range(e.sim.model.ngeom)],
            'bilateral_contact_definition': 'both fingerpad groups contact any geom of the object; not a stable-grasp guarantee',
            'timestamp_alignment': 'state zero is after wait; state t+1 is after action t',
            'snapshot_restorable': False,
            'policy_uses_oracle': False,
        }
        self.rows = []

    def record(self, audit=False):
        e = self.env
        if audit:
            before = e.sim.get_state().flatten().copy()
            ctrl_before = e.sim.data.ctrl.copy()
            rng_before = np.random.get_state()
        contacts = [(int(e.sim.data.contact[i].geom1), int(e.sim.data.contact[i].geom2))
                    for i in range(e.sim.data.ncon)]
        def touching(a, b):
            return any((x in a and y in b) or (x in b and y in a) for x, y in contacts)
        fingers = np.array([[touching(p, s) for p in self.pads] for s in self.geom_sets], dtype=bool)
        matrix = np.array([[touching(a, b) for b in self.geom_sets] for a in self.geom_sets], dtype=bool)
        bilateral = fingers.all(axis=1)
        if audit:
            expected = [e._check_grasp(e.robots[0].gripper, e.get_object(n).contact_geoms) for n in self.names]
            assert np.array_equal(bilateral, expected), 'Contact proxy disagrees with robosuite helper'
        goal = np.array([bool(e._eval_predicate(s)) for s in self.goal], dtype=bool)
        row = {
            'object_position': e.sim.data.body_xpos[self.ids].copy(),
            'object_quaternion': e.sim.data.body_xquat[self.ids].copy(),
            'finger_contact': fingers, 'bilateral_finger_contact': bilateral,
            'object_contact': matrix, 'goal_predicate': goal,
            'qpos': e.sim.data.qpos.copy(), 'qvel': e.sim.data.qvel.copy(),
            'control': e.sim.data.ctrl.copy(), 'sim_time': float(e.sim.data.time),
            'number_of_contacts': int(e.sim.data.ncon),
        }
        for key in ('object_position', 'object_quaternion', 'qpos', 'qvel', 'control'):
            assert np.isfinite(row[key]).all(), key
        if audit:
            assert np.array_equal(before, e.sim.get_state().flatten())
            assert np.array_equal(ctrl_before, e.sim.data.ctrl)
            rng_after = np.random.get_state()
            assert rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1]) and rng_before[2:] == rng_after[2:]
        self.rows.append(row)
        return bool(goal.all())

    def save(self, path):
        np.savez_compressed(str(path), **{key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]})
