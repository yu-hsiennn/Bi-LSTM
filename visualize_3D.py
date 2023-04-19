import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
from matplotlib import animation
from utils import Lab_skeleton
from evaluate import kalman_filter

lab_joints = Lab_skeleton()
joint = lab_joints.get_joint()
jointChain = lab_joints.get_joints_chain()

painting = {}

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, nargs='+', required=True)# file path(pkl/pickle)
parser.add_argument('-t', '--title', type=str)
parser.add_argument('-n', '--frame_num', type=int)
parser.add_argument('-l', '--label', type=str, nargs='+', default=[])# subtitles
parser.add_argument('-o', '--output', type=str, default='result/figure.gif')
parser.add_argument('--scale', type=float, default = 2.5)
args = parser.parse_args()

fig = plt.figure()
ax = []
# label = ['CVAE', 'GroundTruth', 'ThisWork\n(Human3.6M)', 'ThisWork\n(Mixamo)']
for i in range(len(args.file)):
    ax.append(fig.add_subplot(1, len(args.file), i+1, projection="3d"))
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

def loadData():
    motionList = []
    for i, f in enumerate(args.file):
        with open(f, 'rb') as fpick:
            motion = pickle.load(fpick)
        motionList.append(motion)
    for i, _ in enumerate(motionList):
        motionList[i] = kalman_filter(motionList[i])
        motionList[i] = motionList[i].reshape(motionList[i].shape[0], int(motionList[i].shape[1]/3), 3)
        motionList[i] = motionList[i]*args.scale
        if args.frame_num:
            motionList[i] = motionList[i][:args.frame_num]
    return motionList

def init():
    for figure in ax:
        figure.set_xlabel('x')
        figure.set_ylabel('y')
        figure.set_zlabel('z')
        figure.set_xlim(-.6*args.scale, .6*args.scale)
        figure.set_ylim(-.6*args.scale, .6*args.scale)
        figure.set_zlim(-.6*args.scale, .6*args.scale)
        figure.axis('off') #hide axes
        figure.view_init(elev=130,azim=-90)
        # figure.set_xlim(-.5*args.scale, .5*args.scale)
        # figure.set_ylim(-.5*args.scale, .5*args.scale)
        # figure.set_zlim(-.5*args.scale, .5*args.scale)
        # figure.axis('off') #hide axes
        # figure.view_init(elev=130,azim=-90)

def update(i):
    for figure in ax:
        figure.lines.clear()
        figure.collections.clear()
    for f, motion in enumerate(motionList):
        for idx, chain in enumerate(jointChain):
            pre_node = joint[chain[0]]
            next_node = joint[chain[1]]
            x = np.array([motion[i, pre_node, 0], motion[i, next_node, 0]])
            y = np.array([motion[i, pre_node, 1], motion[i, next_node, 1]])
            z = np.array([motion[i, pre_node, 2], motion[i, next_node, 2]])
            if idx < 8:
                # right, red
                ax[f].plot(x, y, z, color="#e74c3c")
            elif 8 <= idx < 14:
                # left ,blue
                ax[f].plot(x, y, z, color="#3498db")
            elif idx == 14 or idx == 15:
                ax[f].plot(x, y, z, color='cyan')
            elif idx % 3 == 1:
                ax[f].plot(x, y, z, color='green')
            elif idx % 3 == 2:
                ax[f].plot(x, y, z, color='navy')
            else:
                ax[f].plot(x, y, z, color="magenta")


motionList = loadData()
ani = animation.FuncAnimation(fig, update, len(motionList[0]), interval=1,init_func=init)
f = f"{args.output[:-4]}.gif" 
writergif = animation.PillowWriter(fps=60)
ani.save(f, writer=writergif)
writervideo = animation.FFMpegWriter(fps=10)
f = f"{args.output[:-4]}.mp4" 
ani.save(f, writer=writervideo)
print("done")
