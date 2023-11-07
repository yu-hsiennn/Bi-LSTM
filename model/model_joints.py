import numpy as np
import torch

class Joints_info:
    def __init__(self, version : str):
        self.version = version
        self.joint_15 = {
            "Head":0, "Neck":1, 
            "RightShoulder":2, "RightArm":3, "RightHand":4, 
            "LeftShoulder":5, "LeftArm":6, "LeftHand":7, 
            "Pelvis":8, 
            "RightThigh":9, "RightKnee":10,"RightAnkle":11,
            "LeftThigh":12, "LeftKnee":13, "LeftAnkle":14
        }
        
        self.joint_21 = {
            "Head":0, "Neck":1, 
            "RightShoulder":2, "RightArm":3, "RightHand":4, 
            "LeftShoulder":5, "LeftArm":6, "LeftHand":7, 
            "Pelvis":8, 
            "RightThigh":9, "RightKnee":10,"RightAnkle":11,
            "LeftThigh":12, "LeftKnee":13, "LeftAnkle":14,
            "LeftToeBase": 15, "RightToeBase": 16,
            "LeftHandIndex1": 17, "LeftHandPinky1": 18,
            "RightHandIndex1": 19, "RightHandPinky1": 20
        }
        
        self.joint_all = {
            "Head":0, "Neck":1, 
            "RightShoulder":2, "RightArm":3, "RightHand":4, 
            "LeftShoulder":5, "LeftArm":6, "LeftHand":7, 
            "Pelvis":8, 
            "RightThigh":9, "RightKnee":10,"RightAnkle":11,
            "LeftThigh":12, "LeftKnee":13, "LeftAnkle":14,
            "LeftToeBase": 15, "RightToeBase": 16,
            "LeftHandThumb1": 17, "LeftHandThumb2": 18, "LeftHandThumb3": 19,
            "LeftHandIndex1": 20, "LeftHandIndex2": 21, "LeftHandIndex3": 22,
            "LeftHandMiddle1": 23, "LeftHandMiddle2": 24, "LeftHandMiddle3": 25,
            "LeftHandRing1": 26, "LeftHandRing2": 27, "LeftHandRing3": 28,
            "LeftHandPinky1": 29, "LeftHandPinky2": 30, "LeftHandPinky3": 31,
            "RightHandThumb1": 32, "RightHandThumb2": 33, "RightHandThumb3": 34,
            "RightHandIndex1": 35, "RightHandIndex2": 36, "RightHandIndex3": 37,
            "RightHandMiddle1": 38, "RightHandMiddle2": 39, "RightHandMiddle3": 40,
            "RightHandRing1": 41, "RightHandRing2": 42, "RightHandRing3": 43,
            "RightHandPinky1": 44, "RightHandPinky2": 45, "RightHandPinky3": 46
        }
        
        self.jointsChain_15 = [
            ["Neck","Pelvis"], ["Head","Neck"],  
            ["RightShoulder", "Neck"], ["RightArm", "RightShoulder"], ["RightHand", "RightArm"],
            ["RightThigh", "Pelvis"], ["RightKnee", "RightThigh"], ["RightAnkle", "RightKnee"],
            ["LeftShoulder", "Neck"], ["LeftArm", "LeftShoulder"], ["LeftHand", "LeftArm"], 
            ["LeftThigh", "Pelvis"], ["LeftKnee", "LeftThigh"], ["LeftAnkle", "LeftKnee"]
        ]
        
        self.jointsChain_21 = [
            ["Neck","Pelvis"], ["Head","Neck"],  
            ["RightShoulder", "Neck"], ["RightArm", "RightShoulder"], ["RightHand", "RightArm"],
            ["RightThigh", "Pelvis"], ["RightKnee", "RightThigh"], ["RightAnkle", "RightKnee"],
            ["LeftShoulder", "Neck"], ["LeftArm", "LeftShoulder"], ["LeftHand", "LeftArm"], 
            ["LeftThigh", "Pelvis"], ["LeftKnee", "LeftThigh"], ["LeftAnkle", "LeftKnee"],
            ["LeftToeBase", "LeftAnkle"], ["RightToeBase", "RightAnkle"],
            ["LeftHandIndex1", "LeftHand"], ["LeftHandPinky1", "LeftHand"],
            ["RightHandIndex1", "RightHand"], ["RightHandPinky1", "RightHand"]
        ]
        
        self.jointsChain_all = [
            ["Neck","Pelvis"], ["Head","Neck"],  
            ["RightShoulder", "Neck"], ["RightArm", "RightShoulder"], ["RightHand", "RightArm"],
            ["RightThigh", "Pelvis"], ["RightKnee", "RightThigh"], ["RightAnkle", "RightKnee"],
            ["LeftShoulder", "Neck"], ["LeftArm", "LeftShoulder"], ["LeftHand", "LeftArm"], 
            ["LeftThigh", "Pelvis"], ["LeftKnee", "LeftThigh"], ["LeftAnkle", "LeftKnee"],
            ["LeftToeBase", "LeftAnkle"], ["RightToeBase", "RightAnkle"],
            ["LeftHandThumb1", "LeftHand"], ["LeftHandThumb2", "LeftHandThumb1"], ["LeftHandThumb3", "LeftHandThumb2"],
            ["LeftHandIndex1", "LeftHand"], ["LeftHandIndex2", "LeftHandIndex1"], ["LeftHandIndex3", "LeftHandIndex2"],
            ["LeftHandMiddle1", "LeftHand"], ["LeftHandMiddle2", "LeftHandMiddle1"], ["LeftHandMiddle3", "LeftHandMiddle2"],
            ["LeftHandRing1", "LeftHand"], ["LeftHandRing2", "LeftHandRing1"], ["LeftHandRing3", "LeftHandRing2"],
            ["LeftHandPinky1", "LeftHand"], ["LeftHandPinky2", "LeftHandPinky1"], ["LeftHandPinky3", "LeftHandPinky2"],
            ["RightHandThumb1", "RightHand"], ["RightHandThumb2", "RightHandThumb1"], ["RightHandThumb3", "RightHandThumb2"],
            ["RightHandIndex1", "RightHand"], ["RightHandIndex2", "RightHandIndex1"], ["RightHandIndex3", "RightHandIndex2"],
            ["RightHandMiddle1", "RightHand"], ["RightHandMiddle2", "RightHandMiddle1"], ["RightHandMiddle3", "RightHandMiddle2"],
            ["RightHandRing1", "RightHand"], ["RightHandRing2", "RightHandRing1"], ["RightHandRing3", "RightHandRing2"],
            ["RightHandPinky1", "RightHand"], ["RightHandPinky2", "RightHandPinky1"], ["RightHandPinky3", "RightHandPinky2"]
        ]
        
    def get_joint(self):
        if self.version in ["V5", "V6"]:
            return self.joint_all
        elif self.version == "V4":
            return self.joint_21
        else:
            return self.joint_15
    
    def get_joints_chain(self):
        if self.version in ["V5", "V6"]:
            return self.jointsChain_all
        elif self.version == "V4":
            return self.jointsChain_21
        else:
            return self.jointsChain_15
    
    def get_joints_index(self):
        joint_index = {}
        if self.version in ["V5", "V6"]:
            joints = self.joint_all
        elif self.version == "V4":
            joints = self.joint_21
        else:
            joints = self.joint_15
            
        for joint, idx in joints.items():
            joint_index[joint] = (idx * 3)
        
        return joint_index
    
    def get_joints_connect(self):
        joint_index = self.get_joints_index()
        joint_chain = self.get_joints_chain()
        
        return [(joint_index[joint[0]], joint_index[joint[1]]) for joint in joint_chain]

class JointDefV6:
    """
    Definition:
    - torso: Head, Neck, RightShoulder, LeftShoulder, Pelvis, RightThigh, LeftThigh (7 joints)
    - left hand: Neck, RightShoulder, LeftShoulder, Pelvis, LeftArm, LeftHand (6 joints)
    - right hand: Neck, RightShoulder, LeftShoulder, Pelvis, RightArm, RightHand (6 joints)
    - left leg: Neck, Pelvis, RightThigh, LeftThigh, LeftKnee, LeftAnkle (6 joints)
    - right leg: Neck, Pelvis, RightThigh, LeftThigh, RightKnee, RightAnkle (6 joints)

    - left hand: "LeftArm", "LeftHand", "LeftHandThumb1", "LeftHandIndex1", "LeftHandMiddle1", "LeftHandRing1", "LeftHandPinky1" (7 joints)
    - left thumb: "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3" (4 joints)
    - left index: "LeftHand", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3" (4 joints)
    - left middle: "LeftHand", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3" (4 joints)
    - left ring: "LeftHand", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3" (4 joints)
    - left pinky: "LeftHand", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3" (4 joints)

    - right hand: "RightArm", "RightHand", "RightHandThumb1", "RightHandIndex1", "RightHandMiddle1", "RightHandRing1", "RightHandPinky1" (7 joints)
    - right thumb: "RightHand", "RightHandThumb1", "RightHandThumb2", "RightHandThumb3" (4 joints)
    - right index: "RightHand", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3" (4 joints)
    - right middle: "RightHand", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3" (4 joints)
    - right ring: "RightHand", "RightHandRing1", "RightHandRing2", "RightHandRing3" (4 joints)
    - right pinky: "RightHand", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3" (4 joints)

    - left foot: LeftKnee, LeftAnkle, LeftToeBase (3 joints)
    - right foot: RightKnee, RightAnkle, RightToeBase (3 joints)
    """
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['Full_Body'] = 141

        self.n_joints_part['Torso'] = 21
        self.n_joints_part['LeftArm'] = 18
        self.n_joints_part['RightArm'] = 18
        self.n_joints_part['LeftLeg'] = 18
        self.n_joints_part['RightLeg'] = 18
        
        self.n_joints_part['LeftHand'] = 21
        self.n_joints_part['RightHand'] = 21

        self.n_joints_part['LeftThumb'] = 12
        self.n_joints_part['LeftIndex'] = 12
        self.n_joints_part['LeftMiddle'] = 12
        self.n_joints_part['LeftRing'] = 12
        self.n_joints_part['LeftPinky'] = 12
        self.n_joints_part['RightThumb'] = 12
        self.n_joints_part['RightIndex'] = 12
        self.n_joints_part['RightMiddle'] = 12
        self.n_joints_part['RightRing'] = 12
        self.n_joints_part['RightPinky'] = 12

        self.n_joints_part['LeftFoot'] = 9
        self.n_joints_part['RightFoot'] = 9

        self.part_list = ['LeftArm', 'RightArm', 'LeftLeg', 'RightLeg', 'Torso',
                        'LeftHand', 'RightHand',
                        'LeftThumb', 'LeftIndex', 'LeftMiddle', 'LeftRing', 'LeftPinky',
                        'RightThumb', 'RightIndex', 'RightMiddle', 'RightRing', 'RightPinky',
                        'LeftFoot', 'RightFoot']

    # concatenate the data it according human joints definition 
    def cat_torch(self, part, data):
        if part == 'Torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'LeftArm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='RightArm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'LeftLeg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'RightLeg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'LeftHand':
            part_data = torch.cat((data[:, 18:24], data[:, 51:54], data[:, 60:63], data[:, 69:72], data[:, 78:81], data[:, 87:90]), axis=1)
        elif part == 'RightHand':
            part_data = torch.cat((data[:, 9:15], data[:, 96:99], data[:, 105:108], data[:, 114:117], data[:, 123:126], data[:, 132:135]), axis=1)
        elif part == 'LeftFoot':
            part_data = torch.cat((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'RightFoot':
            part_data = torch.cat((data[:, 30:36], data[:, 48:51]), axis=1)
        elif part == 'LeftThumb':
            part_data = torch.cat((data[:, 21:24], data[:, 51:60]), axis=1)
        elif part == 'LeftIndex':
            part_data = torch.cat((data[:, 21:24], data[:, 60:69]), axis=1)
        elif part == 'LeftMiddle':
            part_data = torch.cat((data[:, 21:24], data[:, 69:78]), axis=1)
        elif part == 'LeftRing':
            part_data = torch.cat((data[:, 21:24], data[:, 78:87]), axis=1)
        elif part == 'LeftPinky':
            part_data = torch.cat((data[:, 21:24], data[:, 87:96]), axis=1)
        elif part == 'RightThumb':
            part_data = torch.cat((data[:, 12:15], data[:, 96:105]), axis=1)
        elif part == 'RightIndex':
            part_data = torch.cat((data[:, 12:15], data[:, 105:114]), axis=1)
        elif part == 'RightMiddle':
            part_data = torch.cat((data[:, 12:15], data[:, 114:123]), axis=1)
        elif part == 'RightRing':
            part_data = torch.cat((data[:, 12:15], data[:, 123:132]), axis=1)
        elif part == 'RightPinky':
            part_data = torch.cat((data[:, 12:15], data[:, 132:141]), axis=1)

        return part_data

    def cat_numpy(self, part, data):
        if part == 'Torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'LeftArm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='RightArm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'LeftLeg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'RightLeg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'LeftHand':
            part_data = np.concatenate((data[:, 18:24], data[:, 51:54], data[:, 60:63], data[:, 69:72], data[:, 78:81], data[:, 87:90]), axis=1)
        elif part == 'RightHand':
            part_data = np.concatenate((data[:, 9:15], data[:, 96:99], data[:, 105:108], data[:, 114:117], data[:, 123:126], data[:, 132:135]), axis=1)
        elif part == 'LeftFoot':
            part_data = np.concatenate((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'RightFoot':
            part_data = np.concatenate((data[:, 30:36], data[:, 48:51]), axis=1)
        elif part == 'LeftThumb':
            part_data = np.concatenate((data[:, 21:24], data[:, 51:60]), axis=1)
        elif part == 'LeftIndex':
            part_data = np.concatenate((data[:, 21:24], data[:, 60:69]), axis=1)
        elif part == 'LeftMiddle':
            part_data = np.concatenate((data[:, 21:24], data[:, 69:78]), axis=1)
        elif part == 'LeftRing':
            part_data = np.concatenate((data[:, 21:24], data[:, 78:87]), axis=1)
        elif part == 'LeftPinky':
            part_data = np.concatenate((data[:, 21:24], data[:, 87:96]), axis=1)
        elif part == 'RightThumb':
            part_data = np.concatenate((data[:, 12:15], data[:, 96:105]), axis=1)
        elif part == 'RightIndex':
            part_data = np.concatenate((data[:, 12:15], data[:, 105:114]), axis=1)
        elif part == 'RightMiddle':
            part_data = np.concatenate((data[:, 12:15], data[:, 114:123]), axis=1)
        elif part == 'RightRing':
            part_data = np.concatenate((data[:, 12:15], data[:, 123:132]), axis=1)
        elif part == 'RightPinky':
            part_data = np.concatenate((data[:, 12:15], data[:, 132:141]), axis=1)

        return part_data

    def combine_numpy(self, part_datas):
        torso = part_datas['Torso']
        l_arm = part_datas['LeftArm']
        l_leg = part_datas['LeftLeg']
        r_arm = part_datas['RightArm']
        r_leg = part_datas['RightLeg']
        l_hand = part_datas['LeftHand']
        r_hand = part_datas['RightHand']
        l_foot = part_datas['LeftFoot']
        r_foot = part_datas['RightFoot']
        l_thumb = part_datas['LeftThumb']
        l_index = part_datas['LeftIndex']
        l_middle = part_datas['LeftMiddle']
        l_ring = part_datas['LeftRing']
        l_pinky = part_datas['LeftPinky']
        r_thumb = part_datas['RightThumb']
        r_index = part_datas['RightIndex']
        r_middle = part_datas['RightMiddle']
        r_ring = part_datas['RightRing']
        r_pinky = part_datas['RightPinky']

        result = np.concatenate((torso[:, 0:9], r_arm[:, -6:], torso[:, 9:12], l_arm[:, -6:], 
                                    torso[:, 12:18], r_leg[:, -6:], torso[:, 18:21], l_leg[:, -6:],
                                    l_foot[:, -3:], r_foot[:, -3:],
                                    l_hand[:, 6:9], l_thumb[:, -6:], 
                                    l_hand[:, 9:12], l_index[:, -6:], 
                                    l_hand[:, 12:15], l_middle[:, -6:], 
                                    l_hand[:, 15:18], l_ring[:, -6:], 
                                    l_hand[:, 18:21], l_pinky[:, -6:], 
                                    r_hand[:, 6:9], r_thumb[:, -6:], 
                                    r_hand[:, 9:12], r_index[:, -6:], 
                                    r_hand[:, 12:15], r_middle[:, -6:], 
                                    r_hand[:, 15:18], r_ring[:, -6:], 
                                    r_hand[:, 18:21], r_pinky[:, -6:]), 1)
        
        return result

class JointDefV5:
    """
    Definition:
    - torso: Head, Neck, RightShoulder, LeftShoulder, Pelvis, RightThigh, LeftThigh (7 joints)
    - left hand: Neck, RightShoulder, LeftShoulder, Pelvis, LeftArm, LeftHand (6 joints)
    - right hand: Neck, RightShoulder, LeftShoulder, Pelvis, RightArm, RightHand (6 joints)
    - left leg: Neck, Pelvis, RightThigh, LeftThigh, LeftKnee, LeftAnkle (6 joints)
    - right leg: Neck, Pelvis, RightThigh, LeftThigh, RightKnee, RightAnkle (6 joints)

    - left hand with finger: "LeftArm", "LeftHand",
                            "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
                            "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
                            "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
                            "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
                            "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3" (17 joints)
    - right hand with finger: "RightArm", "RightHand",
                            "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
                            "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
                            "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
                            "RightHandRing1", "RightHandRing2", "RightHandRing3",
                            "RightHandPinky1", "RightHandPinky2", "RightHandPinky3" (17 joints)

    - left foot: LeftKnee, LeftAnkle, LeftToeBase (3 joints)
    - right foot: RightKnee, RightAnkle, RightToeBase (3 joints)
    """
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['Full_Body'] = 141

        self.n_joints_part['Torso'] = 21
        self.n_joints_part['LeftArm'] = 18
        self.n_joints_part['RightArm'] = 18
        self.n_joints_part['LeftLeg'] = 18
        self.n_joints_part['RightLeg'] = 18
        
        self.n_joints_part['LeftHand'] = 51
        self.n_joints_part['RightHand'] = 51
        self.n_joints_part['LeftFoot'] = 9
        self.n_joints_part['RightFoot'] = 9

        self.part_list = ['LeftArm', 'RightArm', 'LeftLeg', 'RightLeg', 'Torso', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']

    # concatenate the data it according human joints definition 
    def cat_torch(self, part, data):
        if part == 'Torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'LeftArm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='RightArm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'LeftLeg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'RightLeg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'LeftHand':
            part_data = torch.cat((data[:, 18:24], data[:, 51:96]), axis=1)
        elif part == 'RightHand':
            part_data = torch.cat((data[:, 9:15], data[:, 96:141]), axis=1)
        elif part == 'LeftFoot':
            part_data = torch.cat((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'RightFoot':
            part_data = torch.cat((data[:, 30:36], data[:, 48:51]), axis=1)

        return part_data

    def cat_numpy(self, part, data):
        if part == 'Torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'LeftArm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='RightArm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'LeftLeg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'RightLeg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'LeftHand':
            part_data = np.concatenate((data[:, 18:24], data[:, 51:96]), axis=1)
        elif part == 'RightHand':
            part_data = np.concatenate((data[:, 9:15], data[:, 96:141]), axis=1)
        elif part == 'LeftFoot':
            part_data = np.concatenate((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'RightFoot':
            part_data = np.concatenate((data[:, 30:36], data[:, 48:51]), axis=1)

        return part_data

    def combine_numpy(self, part_datas):
        torso = part_datas['Torso']
        l_arm = part_datas['LeftArm']
        l_leg = part_datas['LeftLeg']
        r_arm = part_datas['RightArm']
        r_leg = part_datas['RightLeg']
        l_hand = part_datas['LeftHand']
        r_hand = part_datas['RightHand']
        l_foot = part_datas['LeftFoot']
        r_foot = part_datas['RightFoot']

        result = np.concatenate((torso[:, 0:9], r_arm[:, -6:], torso[:, 9:12], l_arm[:, -6:], 
                                    torso[:, 12:18], r_leg[:, -6:], torso[:, 18:21], l_leg[:, -6:],
                                    l_foot[:, -3:], r_foot[:, -3:],
                                    l_hand[:, -45:], r_hand[:, -45:]), 1)
        
        return result

class JointDefV4:
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
            part_data = torch.cat((data[:, 18:24], data[:, 51:57]), axis=1)
        elif part == 'righthand':
            part_data = torch.cat((data[:, 9:15], data[:, 57:63]), axis=1)
        elif part == 'leftfoot':
            part_data = torch.cat((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'rightfoot':
            part_data = torch.cat((data[:, 30:36], data[:, 48:51]), axis=1)
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
            part_data = np.concatenate((data[:, 18:24], data[:, 51:57]), axis=1)
        elif part == 'righthand':
            part_data = np.concatenate((data[:, 9:15], data[:, 57:63]), axis=1)
        elif part == 'leftfoot':
            part_data = np.concatenate((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'rightfoot':
            part_data = np.concatenate((data[:, 30:36], data[:, 48:51]), axis=1)
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
                                    l_foot[:, -3:], r_foot[:, -3:],
                                    l_hand[:, -6:], r_hand[:, -6:]), 1)
        
        return result

class JointDefV3:
    """
    Definition:
    - torso: head, neck, pelvis, rshoulder, lshoulder, lthigh, rthigh (7 joints)
    - left hand: lwrist, lelbow, lshoulder, neck, pelvis, rshoulder (6 joints)
    - right hand: rwrist, relbow, rshoulder, neck, pelvis,  lshoulder (6 joints)
    - left leg: lthigh, lknee, lankle, neck, pelvis,  rthigh (6 joints)
    - right leg: rthigh, rknee, rankle, neck, pelvis, lthigh (6 joints)
    """
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['Full_Body'] = 45
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

class JointDefV2:
    """
    Definition:
    - torso: head, neck, pelvis, rshoulder, lshoulder, lthigh, rthigh (7 joints)
    - left hand: lwrist, lelbow, lshoulder, neck, pelvis (5 joints)
    - right hand: rwrist, relbow, rshoulder, neck, pelvis (5 joints)
    - left leg:	lthigh, lknee, lankle, neck, pelvis (5 joints)
    - light leg: rthigh, rknee, rankle, neck, pelvis (5 joints)
    """
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
    
class JointDefV1:
    """
    Definition:
    - Torso: head, neck, pelvis (3 joints)
    - Left hand: lwrist, lelbow, lshoulder, neck, pelvis (5 joints)
    - Right hand: rwrist, relbow, rshoulder, neck, pelvis (5 joints)
    - Left leg: lthigh, lknee, lankle, neck, pelvis (5 joints)
    - Right leg: rthigh, rknee, rankle, neck, pelvis (5 joints)
    """
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