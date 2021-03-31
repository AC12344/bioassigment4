import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')

ARENA_SIDE_LENGTH = 10
NUMBER_OF_ROBOTS  = 50
STEPS             = 500
MAX_SPEED         = 0.1

radius = 2

# Positions
x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

# Velocities
vx = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))
vy = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))

# Set up the output (1024 x 768):
fig = plt.figure(figsize=(10.24, 7.68), dpi=100)
ax = plt.axes(xlim=(0, ARENA_SIDE_LENGTH), ylim=(0, ARENA_SIDE_LENGTH))
points, = ax.plot([], [], 'bo', lw=0, )


# Make the environment toroidal 
def wrap(z):    
    return z % ARENA_SIDE_LENGTH

def init():
    points.set_data([], [])
    return points,

def animate(i):
    global x, y, vx, vy
    spe_fac = seperation()
    cohe_fac = cohesion()
    al_fac = alligement()
    vx, vy = updateVel(spe_fac, cohe_fac,al_fac)
    x = np.array(list(map(wrap, x + vx)))
    y = np.array(list(map(wrap, y + vy)))
    
    points.set_data(x, y)
    print('Step ', i + 1, '/', STEPS, end='\r')
    
    return points,

def distCal(RinX,RinY, NinX, NinY):
    dy = abs(NinY - RinY)
    dx = abs(NinX - RinX)

    if dx > 0.5*ARENA_SIDE_LENGTH:
        dx = ARENA_SIDE_LENGTH - dx
    if dy > 0.5*ARENA_SIDE_LENGTH:
        dy = ARENA_SIDE_LENGTH - dy

    return np.sqrt(dx**2 + dy**2)

def seperation():
    global xs, ys, vys, vxs
    dist = 0
    seperation_fac = []
    for i in range(NUMBER_OF_ROBOTS):
        seperation_facx = 0
        seperation_facy = 0
        count = 1
        for j in range(NUMBER_OF_ROBOTS):
            dist = distCal(x[i],y[i],x[j],y[j])
            if i != j and dist < radius:
                count += 1
                if dist == 0:
                    seperation_facx += MAX_SPEED
                    seperation_facy += MAX_SPEED
                else:
                    seperation_facx += (x[i] - x[j])/dist
                    seperation_facy += (y[i] - y[j])/dist
                    
        seperation_fac.append([seperation_facx/(count), seperation_facy/(count)])
    return seperation_fac

def cohesion():
    cohesion_fac = []
    dist = 0
    for i in range(NUMBER_OF_ROBOTS):
        cohesion_facx = 0
        cohesion_facy = 0
        count = 0
        for j in range(NUMBER_OF_ROBOTS):
            dist = distCal(x[i], y[i], x[j],y[j])
            #print(dist)
            if dist < radius:
                cohesion_facx += x[j]
                cohesion_facy += y[j]
                count +=1
        
        cohesion_fac.append([x[i]-cohesion_facx/count,y[i] - cohesion_facy/count])
       # print("X : %f " % x[i])
       # print("Y : %f " % y[i])
       # print(cohesion_fac[i])
    return cohesion_fac

def alligement():
    al_fac = []
    
    for i in range(NUMBER_OF_ROBOTS):
        al_facx = 0
        al_facy = 0
        count = 0
        for j in range(NUMBER_OF_ROBOTS):
            dist = distCal(x[i], y[i], x[j],y[j])
            if dist < radius:
                al_facx += vx[j]
                al_facy += vy[j]
                count += 1
        al_fac.append([al_facx/count,al_facy/count])
    return al_fac

def updateVel(spe_fac, coh_fac,al_fac):
    
    vx_s = np.array(spe_fac)[:,0]
    vy_s = np.array(spe_fac)[:,1]
    vx_s_max = np.max(np.linalg.norm(spe_fac))
    vy_s_max = np.max(np.linalg.norm(spe_fac))
    if vx_s_max != 0:
        vx_s = vx_s/vx_s_max*MAX_SPEED
    else:
        vx_s = 0.1*MAX_SPEED
    if vy_s_max != 0:
        vy_s = vy_s / vy_s_max * MAX_SPEED
    else:
        vy_s = 0.1*MAX_SPEED
    vx_c = np.array(coh_fac)[:,0]
    vy_c = np.array(coh_fac)[:,1]
    vx_c_max = np.max(np.linalg.norm(coh_fac))
    vy_c_max = np.max(np.linalg.norm(coh_fac))
    if vy_c_max != 0:
        vx_c = vx_c/vx_c_max*MAX_SPEED
    else:
        vx_c = vx_c*MAX_SPEED
    if vy_c_max != 0:
        vy_c = vy_c/vy_c_max*MAX_SPEED
    else:
        vy_c = vy_c*MAX_SPEED
    vx_a = np.array(al_fac)[:,0]
    vy_a = np.array(al_fac)[:,1]
    vx_a_max = np.max(np.linalg.norm(al_fac))
    vy_a_max = np.max(np.linalg.norm(al_fac))
    if vx_a_max != 0:
        vx_a = vx_a/vx_a_max*MAX_SPEED
    else:
        vx_a = vx_a*MAX_SPEED
    if vy_a_max != 0:
        vy_a = vy_a/vy_a_max*MAX_SPEED
    else:
        vy_a = vy_a*MAX_SPEED
    vx = 1*vx_s + 1*vx_c + 1*vx_a 
    vy = 1*vy_s + 1*vy_c + 1*vy_a
    
    return vx, vy

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=STEPS, interval=1, blit=True)
writervideo = animation.FFMpegWriter(fps=60)
anim.save("output.mp4", writer=writervideo)
