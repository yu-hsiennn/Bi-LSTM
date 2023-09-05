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
            "LeftHandIndex1": 17, "LeftHandPinky1": 18,
            "RightHandIndex1": 19, "RightHandPinky1": 20
        }
        
        self.jointsChain = [
            ["Neck","Pelvis"], ["Head","Neck"],  
            ["RightShoulder", "Neck"], ["RightArm", "RightShoulder"], ["RightHand", "RightArm"],
            ["RightThigh", "Pelvis"], ["RightKnee", "RightThigh"], ["RightAnkle", "RightKnee"],
            ["LeftShoulder", "Neck"], ["LeftArm", "LeftShoulder"], ["LeftHand", "LeftArm"], 
            ["LeftThigh", "Pelvis"], ["LeftKnee", "LeftThigh"], ["LeftAnkle", "LeftKnee"],
            ["LeftToeBase", "LeftAnkle"], ["RightToeBase", "RightAnkle"],
            ["LeftHandIndex1", "LeftHand"], ["LeftHandPinky1", "LeftHand"],
            ["RightHandIndex1", "RightHand"], ["RightHandPinky1", "RightHand"]
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