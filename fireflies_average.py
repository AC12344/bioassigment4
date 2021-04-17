import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')

ARENA_SIDE_LENGTH = 10
NUMBER_OF_ROBOTS  = 50
STEPS             = 1000
MAX_SPEED         = 0.1

radius = 1.5
# firefly coupling constant
epsilon = 0.05

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

#circle = plt.Circle((x[0],y[0]), radius, fill = False)
#ax.add_patch(circle)
patches = [plt.Circle((x[i],y[i]),0.15, fill=True) for i in range(NUMBER_OF_ROBOTS)]
for patch in patches:
    ax.add_patch(patch)
    patch.set_color('red')
    patch.set_visible(False)
fireflies = np.random.uniform(low=0, high=1, size=(NUMBER_OF_ROBOTS,))
flashes = []
for i in range(NUMBER_OF_ROBOTS):
    flashes.append(0)

# visualize firefly synchronization
values = np.zeros((STEPS,NUMBER_OF_ROBOTS))


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
    #print('Step ', i + 1, '/', STEPS, end='\r')
    synchronize()
    values[i,:] = fireflies
    for j in range(NUMBER_OF_ROBOTS):
        if fireflies[j] >= 1:
            patches[j].center = (x[j],y[j])
            patches[j].set_visible(True)
            flashes[j] += 1
            if flashes[j] == 8:
                fireflies[j] = 0
        else:
            patches[j].center = (x[j],y[j])
            patches[j].set_visible(False)
            flashes[j] = 0
    return points,

def synchronize():
    for i in range(NUMBER_OF_ROBOTS):
        count = 0
        for j in range(NUMBER_OF_ROBOTS):
            _,_,dist = distCal(x[i],y[i],x[j],y[j])
            if i != j and dist < radius:
                if flashes[j] >= 1:
                    count += 1
        fireflies[i] = fireflies[i] + 1/60 + epsilon*count*fireflies[i]
        if fireflies[i] > 1:
            fireflies[i] = 1

def distCal(RinX,RinY, NinX, NinY):
    dy = abs(NinY - RinY)
    dx = abs(NinX - RinX)

    if dx > 0.5*ARENA_SIDE_LENGTH:
        dx = ARENA_SIDE_LENGTH - dx
    if dy > 0.5*ARENA_SIDE_LENGTH:
        dy = ARENA_SIDE_LENGTH - dy

    dist = np.sqrt(dx**2 + dy**2)
    return dx, dy, dist

def correctWarp(RinX, RinY,NinX,NinY):
    dy = NinY - RinY
    dx = NinX - RinX
    
    if dx > 0.5*ARENA_SIDE_LENGTH:
        dx  = NinX - ARENA_SIDE_LENGTH
    elif dx < -0.5*ARENA_SIDE_LENGTH:
        dx = NinX + ARENA_SIDE_LENGTH
    else:
        dx = NinX
    if dy > 0.5*ARENA_SIDE_LENGTH:
        dy  = NinY - ARENA_SIDE_LENGTH
    elif dy < -0.5*ARENA_SIDE_LENGTH:
        dy = NinY + ARENA_SIDE_LENGTH
    else:
        dy = NinY

    return dx, dy


def seperation():
    global xs, ys, vys, vxs
    dist = 0
    seperation_fac = []
    for i in range(NUMBER_OF_ROBOTS):
        seperation_facx = 0
        seperation_facy = 0
        count = 1
        for j in range(NUMBER_OF_ROBOTS):
            _,_,dist = distCal(x[i],y[i],x[j],y[j])
            if i != j and dist < radius:
                count += 1
                if dist == 0:
                    seperation_facx += np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=1)  
                    seperation_facy += np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=1)
                else:
                    dx,dy = correctWarp(x[i],y[i],x[j],y[j])
                    seperation_facx += 1/(x[i]-dx)
                    seperation_facy += 1/(y[i]-dy)
                    
        seperation_fac.append([seperation_facx/count, seperation_facy/count])
    return seperation_fac

def cohesion():
    cohesion_fac = []
    dist = 0
    for i in range(NUMBER_OF_ROBOTS):
        cohesion_facx = 0
        cohesion_facy = 0
        count = 0
        for j in range(NUMBER_OF_ROBOTS):
            _,_,dist = distCal(x[i], y[i], x[j],y[j])
            #print(dist)
            if dist < radius:
                dx,dy = correctWarp(x[i],y[i],x[j],y[j])
                cohesion_facx += dx
                cohesion_facy += dy
                count +=1
        cohesion_fac.append([cohesion_facx/count- x[i],cohesion_facy/count - y[i]])
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
            _,_,dist = distCal(x[i], y[i], x[j],y[j])
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
        vx_s = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=NUMBER_OF_ROBOTS,)

    if vy_s_max != 0:
        vy_s = vy_s / vy_s_max * MAX_SPEED
    else:
        vy_s = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=NUMBER_OF_ROBOTS,)
    
    vx_c = np.array(coh_fac)[:,0]
    vy_c = np.array(coh_fac)[:,1]
    vx_c_max = np.max(np.linalg.norm(coh_fac))
    vy_c_max = np.max(np.linalg.norm(coh_fac))
    if vy_c_max != 0:
        vx_c = vx_c/vx_c_max*MAX_SPEED
    else:
        vx_c = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=NUMBER_OF_ROBOTS,)        
        #vx_c = vx_c*MAX_SPEED
    if vy_c_max != 0:
        vy_c = vy_c/vy_c_max*MAX_SPEED
    else:
        vy_c = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=NUMBER_OF_ROBOTS,)
        #vy_c = vy_c*MAX_SPEED
    
    vx_a = np.array(al_fac)[:,0]
    vy_a = np.array(al_fac)[:,1]
    vx_a_max = np.max(np.linalg.norm(al_fac))
    vy_a_max = np.max(np.linalg.norm(al_fac))
    if vx_a_max != 0:
        vx_a = vx_a/vx_a_max*MAX_SPEED
    else:
        vx_a = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=NUMBER_OF_ROBOTS,)        
        #vx_a = vx_a*MAX_SPEED
    if vy_a_max != 0:
        vy_a = vy_a/vy_a_max*MAX_SPEED
    else:
        vy_a = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=NUMBER_OF_ROBOTS,)
        #vy_a = vy_a*MAX_SPEED
    dvx =  vx_s + 1*vx_c + 1*vx_a
    dvy =  vy_s + 1*vy_c + 1*vy_a
    
    return dvx, dvy

first_seen_sum = 0
times_sync = 0
for it in range(3):
    # Positions
    x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
    y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

    # Velocities
    vx = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))
    vy = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))
    fireflies = np.random.uniform(low=0, high=1, size=(NUMBER_OF_ROBOTS,))
    flashes = []
    for i in range(NUMBER_OF_ROBOTS):
        flashes.append(0)
    # visualize firefly synchronization
    values = np.zeros((STEPS,NUMBER_OF_ROBOTS))
    print(it)
    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=STEPS, interval=1, blit=True)
    writervideo = animation.FFMpegWriter(fps=60)
    anim.save("output_sync_avg.mp4", writer=writervideo)
    first_seen = 0
    for n in range(STEPS):
        count = 0
        for m in range(NUMBER_OF_ROBOTS):
            if values[n,m] >= 1:
                count += 1
        if count == NUMBER_OF_ROBOTS and first_seen == 0:
            first_seen = n
            first_seen_sum += first_seen
    if first_seen != 0:
        times_sync += 1

print(times_sync)
print(first_seen_sum/times_sync)