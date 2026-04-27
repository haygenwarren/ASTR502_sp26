import isochrones
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from isochrones.mist import MIST_Isochrone

mist = MIST_Isochrone()
mist.initialize()

#plot two isochrones, 1 Gyr and 5 Gyr
age1 = 1e9
age5 = 5e9

iso1 = mist.isochrone(age=np.log10(age1))
iso5 = mist.isochrone(age=np.log10(age5))

s1 = np.argsort(iso1["Teff"].to_numpy())
s5 = np.argsort(iso5["Teff"].to_numpy())

plt.figure(figsize=(6, 8))
plt.plot(iso1["logTeff"], iso1["logL"], label="1 Gyr")
plt.plot(iso5["logTeff"], iso5["logL"], label="5 Gyr")
plt.gca().invert_xaxis()
plt.xlabel('log(T_eff) (K)')
plt.ylabel('log(L / L_sun)')
plt.legend()
plt.tight_layout()
plt.savefig('gyrisochrones.png')
plt.show()

#plot two isochrones, at Fe/H=0.0 and -0.5
age_feh = 1e9  # 1 Gyr

# request two isochrones at the same age but different metallicities
iso_feh0 = mist.isochrone(age=np.log10(age_feh), feh=0.0)
iso_fehm05 = mist.isochrone(age=np.log10(age_feh), feh=-0.5)

# sort by Teff (increasing logTeff) for clean plotting
s0 = np.argsort(iso_feh0["logTeff"].to_numpy())
sm5 = np.argsort(iso_fehm05["logTeff"].to_numpy())

plt.figure(figsize=(6, 8))
plt.plot(iso_feh0["logTeff"], iso_feh0["logL"], label='[Fe/H]=0.0, 1 Gyr')
plt.plot(iso_fehm05["logTeff"], iso_fehm05["logL"], label='[Fe/H]=-0.5, 1 Gyr')
plt.gca().invert_xaxis()
plt.xlabel('log(T_eff) (K)')
plt.ylabel('log(L / L_sun)')
plt.legend()
plt.tight_layout()
plt.savefig('feh.png')
plt.show()

#create a plot with sliders to change age and metallicity of the isochrone
from matplotlib.widgets import Slider

init_age_gyr = 1.0
init_feh = 0.0

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.12, bottom=0.25)

iso = mist.isochrone(age=np.log10(init_age_gyr * 1e9), feh=init_feh)
sidx = np.argsort(iso["logTeff"])
x = iso["logTeff"]
y = iso["logL"]
line, = ax.plot(x, y, lw=2)
ax.invert_xaxis()
ax.set_xlabel('log(T_eff) (K)')
ax.set_ylabel('log(L / L_sun)')
ax.set_title(f'Isochrone: {init_age_gyr:.2f} Gyr, [Fe/H]={init_feh:.2f}')

ax_age = plt.axes([0.12, 0.12, 0.76, 0.03])
ax_feh = plt.axes([0.12, 0.06, 0.76, 0.03])

age_slider = Slider(ax_age, 'Age (Gyr)', 0.01, 14.0, valinit=init_age_gyr)
feh_slider = Slider(ax_feh, '[Fe/H]', -2.5, 0.5, valinit=init_feh)

def update(val):
    age_gyr = age_slider.val
    feh = feh_slider.val
    iso = mist.isochrone(age=np.log10(age_gyr * 1e9), feh=feh)
    sidx = np.argsort(iso["logTeff"])
    x = iso["logTeff"]
    y = iso["logL"]
    line.set_xdata(x)
    line.set_ydata(y)
    ax.set_title(f'Isochrone: {age_gyr:.2f} Gyr, [Fe/H]={feh:.2f}')
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

age_slider.on_changed(update)
feh_slider.on_changed(update)
plt.savefig('sliderisochrones.png')
plt.show()

matplotlib.interactive(True)
plt.ion()
plt.show(block=False)

try:
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
except KeyboardInterrupt:
    pass