# IKKE KODEMAL
# SKREVET EGEN KODE :)
# (Python 3.9)

# Import neccesary packages
from time import time 
start_time = time() # The time of when the code starts running

# Packages used for computations
import numpy as np
import copy

# Packages from ast2000tools that we need
import ast2000tools.utils as ast
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem

# Packages for visualization of the simulation
import matplotlib.pyplot as plt

# Constants
c:float = const.c #Speed of light in vacuum
lamda0:float = 656.3 # Hydrogen-Alpha line
m_sun:float = const.m_sun # Solar mass
k:float = const.k_B # Boltzman constant
m_H:float = const.m_H2/2 # mass of hydrogen 
G:float = const.G # gravitational constant

# Getting all the info from the txt files into one singular array
files:list = ['star0_1.01.txt','star1_1.40.txt','star2_2.90.txt','star3_1.34.txt','star4_5.66.txt']
star_masses = np.array([1.01, 1.40, 2.90, 1.34, 5.66])*m_sun
stars:list[np.ndarray] = []
for file in files:
    data:np.ndarray = np.loadtxt(file) 
    stars.append(np.array(data))

# Find mean of lamda for all stars
mean_lamda = np.zeros(len(stars))
for i in range(len(stars)):
    mean_lamda[i] = np.mean(stars[i][:, 1])

old_stars_data = copy.deepcopy(stars) # Saving lamda meassurments for future use

# Changing lamda meassurments to speed meassurments and scaling it so that the speed is defined as speed towards earth realtive to the stars own centre of mass with orbiting planet
mean_velocities:np.ndarray = c*((mean_lamda - lamda0)/(lamda0))
for i in range(len(stars)):
    stars[i][:, 1] = c*((stars[i][:, 1] - lamda0)/(lamda0)) - mean_velocities[i]

# Plots for flux over time and speed towards earth relative to centre of the stars nass

'''
for i in range(len(stars)):
    plt.figure(figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.title(f'Planet {i}')
    plt.ylabel('Speed of star')
    plt.xlabel('Time (days)')
    plt.plot(stars[i][:, 0],stars[i][:, 1])


    plt.subplot(2, 1, 2)
    plt.ylabel('Relative flux')
    plt.xlabel('Time (days)')
    plt.plot(stars[i][:, 0],stars[i][:, 2])
    

    plt.show()
'''

# These measurments are very rough by-eye estimations
stars_with_planet:list = [0, 1, None, 3, None] # Stars that could be seen having orbits from velocity graphs
orbit_period:np.ndarray = np.array([4200, 4900, None, 4400, None]) # Periods for each of the orbits (days)
max_radial_velocity:np.ndarray = np.array([19, 56, None, 20, None]) # Maximun radial velocity of stars (m/s)

def min_mass(m_star, v_star, period):
    '''Function for finding the minimum mass for a planet orbiting a star'''
    return ((m_star**(2/3))*v_star*(period**(1/3))/(2*np.pi*G)**(1/3))

# Calculating the masses of the planets orbiting these stars
planet_masses:np.ndarray = np.zeros(len(stars))
for i in range(len(stars)):
    if (i in stars_with_planet):
        planet_masses[i] = min_mass(star_masses[i], max_radial_velocity[i], orbit_period[i])
    else:
        planet_masses[i] = None
    
# We will use least square method to find better values for period, and radiall velocity
# This code takes a while, so we will do it for only 1 planet
planet = 1 # Planet number

def model(t, v, t0, P): 
    '''The model we want to fit our data into'''
    return v*np.cos(((2*np.pi)/P)*(t - t0))

# By-eye estimations of intervals were v_max, t0 and P must be
points = 20
v_range = [np.linspace(6, 20, points), np.linspace(42, 60, points), None, np.linspace(5, 18, points), None] 
t0_range = [np.linspace(3100, 3600, points), np.linspace(560, 820, points), None, np.linspace(2200, 2600, points), None]
P_range = [np.linspace(4000, 4500, points), np.linspace(4500, 5000, points), None, np.linspace(4200, 4800, points), None]

differences = np.zeros((points, points, points))
index = []

# Code to find all the models and how much they diviate from the data
for i in range(points):
    for j in range(points):
        for k in range(points):
            sum = 0
            for t in range(len(stars[planet][:, 0])):
                sum += (stars[planet][t, 1] - model(t, v_range[planet][i], t0_range[planet][j], P_range[planet][k]))**2
            differences[i, j, k] = sum
            
min_difference = np.min(differences) 
min_index = np.where(differences == min_difference) # Finding (i, j, k) for model with the lowest "difference"

calculated_min_mass = min_mass(star_masses[planet], v_range[planet][i], P_range[planet][k]) # Calculating mass of planet with our new model

# This next part of code is for plotting absorbtion lines we would recive from the star
T = 6000 # Temperature (Kelvin)
FWHM = ((2e-9*lamda0)/c)*np.sqrt((2*k*T*np.log(2))/(m_H))
sigma = FWHM/np.sqrt(8*np.log(2)) # Standard diviation of the guassian distrubiton
F_min = 0.7 # Minimum flux (depth of the absorption line)
F_cont = 1 # Continuum flux
lamda = np.linspace(640, 670, 1000) 

def F(lamda, lamda0, sigma, F_min, F_cont):
    return F_cont + (F_min - F_cont)*np.exp(-((lamda - lamda0)**2)/(2*(sigma**2)))

standard_flux = F(lamda, lamda0, sigma, F_min, F_cont) # This gives the standard absorption line for hydrogen
initial_flux = F(lamda, old_stars_data[planet][0, 1], sigma, F_min, F_cont) # Absorbtion line for first data point

new_v = model(P_range[planet][min_index[2]][0]/4, v_range[planet][min_index[0]][0], t0_range[planet][min_index[1]][0], P_range[planet][min_index[2]][0])
new_lambda = lamda0 + (new_v/c)*lamda0  # new observed wavelength
later_flux = F(lamda, new_lambda, sigma, F_min, F_cont) # Absorbtion line after P/4 periods

# Code to plot spectral lines
'''
plt.figure(figsize=(8, 5))
plt.title(f'Planet {planet}')
#plt.plot(lamda, standard_flux, label='Standars absorbtion line for hydrogen')
plt.plot(lamda, initial_flux, label='Initial absorption line')
plt.plot(lamda, later_flux, label='Absorption line after one fourth period')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux')
plt.grid()
plt.legend()
plt.show()
'''


# Plot for model vs data
'''
plt.title(f'Planet {planet}')
plt.ylabel('Speed of star')
plt.xlabel('Time (days)')
plt.plot(stars[planet][:, 0],stars[planet][:, 1], label='data')
plt.plot(stars[planet][:, 0],model(stars[planet][:, 0], v_range[planet][min_index[0]], t0_range[planet][min_index[1]], P_range[planet][min_index[2]]),label='model')
plt.legend()
plt.show()
'''

print()
print(f'Results from \033[1mPlanet {planet}:\033[0m')
print(f'\033[1mBy-eye estimate:\033[0m Period = \033[1m{orbit_period[planet]}\033[0m days | Max radial velocity = \033[1m{max_radial_velocity[planet]}\033[0m m/s | Minimum mass = \033[1m{planet_masses[planet]}\033[0m kg')
print(f'\033[1mAnalyzed estimate:\033[0m Period = \033[1m{P_range[planet][min_index[2]][0]}\033[0m days | Max radial velocity = \033[1m{v_range[planet][min_index[0]][0]}\033[0m m/s | Minimum mass = \033[1m{calculated_min_mass}\033[0m kg')
print()

end_time = time() # The time of when the code stops running
print(f'The code used \033[1m{end_time - start_time}\033[0m seconds to fully execute')


