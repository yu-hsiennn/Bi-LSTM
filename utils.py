class Lab_skeleton():
    def __init__(self):

        self.joint = {
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
        
        self.jointsChain = [
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

        self.jointIndex = {}
        for joint, idx in self.joint.items():
            self.jointIndex[joint] = (idx * 3)
        
        self.jointsConnect = [(self.jointIndex[joint[0]], self.jointIndex[joint[1]]) for joint in self.jointsChain]

    def get_joint(self):
        return self.joint

    def get_joints_connect(self):
        return self.jointsConnect
    
    def get_joints_index(self):
        return self.jointIndex
    
    def get_joints_chain(self):
        return self.jointsChain

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count