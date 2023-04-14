import numpy as np
import torch

"""
Definition:
- torso: head, neck, rshoulder, lshoulder, pelvis, rthigh, lthigh (7 joints)
- left hand: lwrist, lelbow, lshoulder, neck, pelvis, rshoulder (6 joints)
- right hand: rwrist, relbow, rshoulder, neck, pelvis,  lshoulder (6 joints)
- left leg: lthigh, lknee, lankle, neck, pelvis,  rthigh (6 joints)
- right leg: rthigh, rknee, rankle, neck, pelvis, lthigh (6 joints)

- left hand with finger: lelbow, lwrist, lindex, lpinky (4 joints)
- right hand with finger: relbow, rwrist, rindex, rpinky (4 joints)
- left foot: lknee, lfoot, ltoestart (3 joints)
- right foot: rknee, rfoot, rtoestart (3 joints)
"""
class JointDefV4:
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['entire'] = 63
        self.n_joints_part['torso'] = 21
        self.n_joints_part['limb'] = 18
        self.n_joints_part['leftarm'] = 18
        self.n_joints_part['rightarm'] = 18
        self.n_joints_part['leftleg'] = 18
        self.n_joints_part['rightleg'] = 18
        
        self.n_joints_part['lefthand'] = 12
        self.n_joints_part['righthand'] = 12
        self.n_joints_part['leftfoot'] = 9
        self.n_joints_part['rightfoot'] = 9

        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso', 'lefthand', 'righthand', 'leftfoot', 'rightfoot']

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'lefthand':
            part_data = torch.cat((data[:, 18:24], data[:, 45:51]), axis=1)
        elif part == 'righthand':
            part_data = torch.cat((data[:, 9:15], data[:, 51:57]), axis=1)
        elif part == 'leftfoot':
            part_data = torch.cat((data[:, 39:45], data[:, 57:60]), axis=1)
        elif part == 'rightfoot':
            part_data = torch.cat((data[:, 30:36], data[:, 60:63]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'lefthand':
            part_data = np.concatenate((data[:, 18:24], data[:, 45:51]), axis=1)
        elif part == 'righthand':
            part_data = np.concatenate((data[:, 9:15], data[:, 51:57]), axis=1)
        elif part == 'leftfoot':
            part_data = np.concatenate((data[:, 39:45], data[:, 57:60]), axis=1)
        elif part == 'rightfoot':
            part_data = np.concatenate((data[:, 30:36], data[:, 60:63]), axis=1)
        return part_data

    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        l_hand = part_datas['lefthand']
        r_hand = part_datas['righthand']
        l_foot = part_datas['leftfoot']
        r_foot = part_datas['rightfoot']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], 
                                    torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:],
                                    l_hand[:, -6:], r_hand[:, -6:], l_foot[:, -3:], r_foot[:, -3:]), 1)
        
        return result

"""
Definition:
- torso: head, neck, pelvis, rshoulder, lshoulder, lthigh, rthigh (7 joints)
- left hand: lwrist, lelbow, lshoulder, neck, pelvis, rshoulder (6 joints)
- right hand: rwrist, relbow, rshoulder, neck, pelvis,  lshoulder (6 joints)
- left leg: lthigh, lknee, lankle, neck, pelvis,  rthigh (6 joints)
- right leg: rthigh, rknee, rankle, neck, pelvis, lthigh (6 joints)
"""
class JointDefV3:
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['entire'] = 45
        self.n_joints_part['torso'] = 21
        self.n_joints_part['limb'] = 18
        self.n_joints_part['leftarm'] = 18
        self.n_joints_part['rightarm'] = 18
        self.n_joints_part['leftleg'] = 18
        self.n_joints_part['rightleg'] = 18
        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        return part_data

    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], 
                                    torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
        return result

"""
Definition:
- torso: head, neck, pelvis, rshoulder, lshoulder, lthigh, rthigh (7 joints)
- left hand: lwrist, lelbow, lshoulder, neck, pelvis (5 joints)
- right hand: rwrist, relbow, rshoulder, neck, pelvis (5 joints)
- left leg:	lthigh, lknee, lankle, neck, pelvis (5 joints)
- light leg: rthigh, rknee, rankle, neck, pelvis (5 joints)
"""
class JointDefV2:
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['entire'] = 45
        self.n_joints_part['torso'] = 21
        self.n_joints_part['limb'] = 15
        self.n_joints_part['leftarm'] = 15
        self.n_joints_part['rightarm'] = 15
        self.n_joints_part['leftleg'] = 15
        self.n_joints_part['rightleg'] = 15
        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data
    
    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], 
                                    torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
        return result
    
"""
Definition:
- Torso: head, neck, pelvis (3 joints)
- Left hand: lwrist, lelbow, lshoulder, neck, pelvis (5 joints)
- Right hand: rwrist, relbow, rshoulder, neck, pelvis (5 joints)
- Left leg: lthigh, lknee, lankle, neck, pelvis (5 joints)
- Right leg: rthigh, rknee, rankle, neck, pelvis (5 joints)
"""
class JointDefV1:
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['entire'] = 45
        self.n_joints_part['torso'] = 9
        self.n_joints_part['limb'] = 15
        self.n_joints_part['leftarm'] = 15
        self.n_joints_part['rightarm'] = 15
        self.n_joints_part['leftleg'] = 15
        self.n_joints_part['rightleg'] = 15
        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:3], data[:, 3:6], data[:, 24:27]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:3], data[:, 3:6], data[:, 24:27]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data
        
    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        result = np.concatenate((torso[:, 0:6], rarm[:, 3:6], rarm[:, -6:], larm[:, 3:6], larm[:, -6:], 
                                        torso[:, 6:9], rleg[:, -9:-6], rleg[:, -6:], lleg[:, -9:-6], lleg[:, -6:]), 1)
        return result