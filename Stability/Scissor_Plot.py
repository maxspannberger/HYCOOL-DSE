import matplotlib.pyplot as plt
import numpy as np

# Constants
kn = -4                              #nacelle emperical factor (sl. 40), -4 for nacelle fwd of wing LE
Vh_V = 0.85                          #tail/wing speed ratio (sl.42), 1 for t-tail, 0.95 for fin mtd, 0.85 for fus-mtd stabilizer
etah = 0.95                          #efficiency of horizontail tail, assumed constant (comments sl.43)
CLh = -0.8                           #for adjustable horizontal tail (sl.17)
gamma = 1.4
R = 287
T = 288.15
S_M = 0.05                           #stability margin

#cg range
xcg_lower = 0.283*0.98               # Forward CG limit, taken from Luca's analysis
xcg_upper = 0.640*1.02  #0.81        # Aft CG limit, dummy value

# Parameters for original scissor plot
xac_w = 0.25                         #location of aerodynamic center of wing, ASSUMED, FIND SOURCE
bf = 2.695 #[m]                      #fuselage diameter, airport manual
hf = 2.695 #[m]                      #fuselage height, airport manual
lf = 36.57 #[m]                      #fuselage length, airport manual
lfn = 17.246 #[m]                    #from tip of fuselage to root of main wing, measured from scaled drawing in airport manual
S = 77.39 #[m]                       #wing area main wing, taken from airport planning manual
Sh = 15.91 #[m]                      #wing area horiontal stabilizer, taken from airport planning manual
b = 26.17 #[m]                       #wing span, airport manual
c = 3.48 #[m]                        #mean aerodynamic chord main wing, easa document
ct = 1.425 #[m]                      #tip chord length, measured from scaled drawing in airport manual
cr = 5.11 #[m]                       #root chrod length, airport manual
sweep25 = np.deg2rad(27.553)         #wing sweep at quarter MAC chord length of main wing, in RAD!!!,  measured from scaled drawing
sweep50 = np.deg2rad(22.676)         #wing sweep at half MAC chord length of main wing, in RAD!!!, measured from sclaed drawing
sweep25_tail = np.deg2rad(30.104)    #wing sweep at quarter MAC chord length of horizontal tail, in RAD!!!, measured from sclaed drawing
sweep50_tail = np.deg2rad(26.318)    #wing sweep at half MAC chord length of horizontal tail, in RAD!!!, measured from sclaed drawing
bn = 1.56 #[m]                       #engine outer diamater, airport manual
ln = -11.908 #[m]                    #distance from ac main wing to end of engine, NEGATIVE VALUE!!!, see slide 40 lect 7, measured from scaled drawing airport manual
bh = 8.54 #[m]                       #horizontal tail wing span, airport manual
Ah = bh**2/Sh                        #aspect ratio horizontal tail
A = b**2/S  + 1.9*1.225/b * b**2/S   #aspect ratio main wing, not purely geometric to take winglet into account
Mcruise = 0.84                       #max cruise mach number, easa document
S_net = S - bf * cr                  #S - central wing area 'inside' fuselage
lh = 16.944 #[m]                     #tail length, distance between wing ac and stabilizer ac, measured from scaled drawing airport manual
hh = 4.342 #[m]                      #vertical distance between stabilizer plane and wing plane, measured from scaled drawing airport manual
CL0 = 1.4                            #CL at 0 angle of attack for flapped wing (landing), only flaps, is for pitching moment calculations
mu1 = 0.215                          #see slides 20-21, lect 8 for calculation
mu2 = 0.79                           #see slides 20-21 for calculation
mu3 = 0.045                          #see slides 20-21 for calculation
cdash_c = 1.2917                     #see slide 22, lect 8 -> need flap delfection angle, see notes, 45 degree deflection angle
delta_cl_max = 1.8                  #increase in AIRFOIL cl due to flaps extended at landing, upper limit DATCOM method
Swf_S = 0.621113839                  #ratio between flapped wing area (with only TE flaps), and clean wing area, measured from airport manual scaled drawing
CLA_h = 1.661                        #lift coefficient of aircraft without tail, found with calculation see notes
CL = 1.661                           #wing lift coefficient at landing, assume same as CLA_h?
Cm0_airfoil = -0.11                  #moment coefficient airfoil at zero angle of attack, airfoiltools.com, NASA SC(2)-0612
Vlanding = 71.5078  #[m/s]           #max landing speed with flaps fully out (at max landing weight), taken form aiport planning manual
M_max_landing = 36968 #[kg]          #max landing weight, taken from airport planning manual

#------------------------------------- STABILITY CONDITION --------------------------------------------------

# CLalpha_h -> CL gradient for horizontail tail
betas = np.sqrt(1 - Mcruise**2)        # compressability factor for stability
CLalpha_h_s = (2*np.pi*Ah)/(2 + np.sqrt(4 + (Ah*betas/etah)**2 * (1+ np.tan(sweep50_tail)**2/betas**2)))

# CLalpha_A_h -> CL gradient for fuselage + wing minus tail
CLalpha_w_s = (2*np.pi*A)/(2 + np.sqrt(4 + (A*betas/etah)**2 * (1+ np.tan(sweep50)**2/betas**2)))
CLalpha_A_h_s = CLalpha_w_s * (1 + 2.15 * bf/b)*S_net/S + np.pi/2 * bf**2 / S
                             
# xac -> distance aerodynamic center wrt LEMAC
xacf1_s = (-1.8/CLalpha_A_h_s) * (bf*hf*lfn) / (S*c)            # fuselage destabilizing contribution
xacf2_s = (0.273/(1 + ct/cr))*((bf*S/b * (b-bf))/(c**2*(b + 2.15*bf)))*np.tan(sweep25)  # fuselage stabilizing contribution
xacn_s = 2 * (kn * (bn**2 * ln)/(S*c*CLalpha_A_h_s))     # contribution from nacelles
xac_s = xac_w + xacf1_s + xacf2_s + xacn_s

# depsilon_dalpha -> downwash gradient, slides 46 and 47 for explanation
r = 2*lh/b      #sl. 46
mtv = 2 * hh / b                  #see slide 46 lect 7 for definition
Kepsilon_sweep = (0.1124 + 0.1265 * sweep25 + 0.1766 * sweep25**2)/(r**2) + 0.1024/r + 2
Kepsilon_sweep0 = 0.1124/r**2 + 0.1024/r + 2
depsilon_dalpha = Kepsilon_sweep/Kepsilon_sweep0 * ((r/(r**2+mtv**2))*0.4876/np.sqrt(r**2 + 0.6319+mtv**2) + (1 + (r**2/(r**2+0.7915+5.0734*(mtv**2)))**0.3113)*(1 - np.sqrt(mtv**2/(1+mtv**2))))*CLalpha_w_s/(np.pi*A)  

# we need to compute Shs for each xcg
xcg = np.arange(-0.4, 1.2, 0.01)

# Stability Sh/s
ShS_s = xcg/(CLalpha_h_s/CLalpha_A_h_s * (1 - depsilon_dalpha)*lh/c * (Vh_V)**2) - (xac_s - S_M)/(CLalpha_h_s/CLalpha_A_h_s * (1 - depsilon_dalpha)*lh/c * (Vh_V)**2)
ShS_s_no_margin = xcg/(CLalpha_h_s/CLalpha_A_h_s * (1 - depsilon_dalpha)*lh/c * (Vh_V)**2) - (xac_s)/(CLalpha_h_s/CLalpha_A_h_s * (1 - depsilon_dalpha)*lh/c * (Vh_V)**2)


#------------------------------------- CONTROLLABILITY CONDITION --------------------------------------------------

# CLalpha_h -> CL gradient for horizontail tail
Mlanding = Vlanding / np.sqrt(gamma*R*T)
betac = np.sqrt(1 - Mlanding**2)        # compressability factor for stability
CLalpha_h_c = (2*np.pi*Ah)/(2 + np.sqrt(4 + (Ah*betac/etah)**2 * (1+ np.tan(sweep50_tail)**2/betac**2)))

# CLalpha_A_h -> CL gradient for fuselage + wing minus tail
CLalpha_w_c = (2*np.pi*A)/(2 + np.sqrt(4 + (A*betac/etah)**2 * (1+ np.tan(sweep50)**2/betac**2)))
CLalpha_A_h_c = CLalpha_w_c * (1 + 2.15 * bf/b)*S_net/S + np.pi/2 * bf**2 / S
                             
# xac -> distance aerodynamic center wrt LEMAC
xacf1_c = (-1.8/CLalpha_A_h_c) * (bf*hf*lfn) / (S*c)            # fuselage destabilizing contribution
xacf2_c = (0.273/(1 + ct/cr))*((bf*S/b * (b-bf))/(c**2*(b + 2.15*bf)))*np.tan(sweep25)  # fuselage stabilizing contribution
xacn_c = 2 * (kn * (bn**2 * ln)/(S*c*CLalpha_A_h_c))     # contribution from nacelles
xac_c = xac_w + xacf1_c + xacf2_c + xacn_c

# Cmac calculation sl. 19 lect 8
Cmac_w = Cm0_airfoil * (A * np.cos(sweep25)**2 / (A + 2*np.cos(sweep25))) #contribution of wing
Cmac_fus = -1.8*(1-2.5*bf/lf)*np.pi*bf*hf*lf/(4*S*c) * CL0/CLalpha_A_h_c #contribution of fuselage
delta_Cmac = mu2 * (-mu1 * delta_cl_max * cdash_c - (CL + delta_cl_max*(1-Swf_S))*1/8 * cdash_c*(cdash_c-1)) + 0.7*A*mu3/(1+2/A) * delta_cl_max*np.tan(sweep25)
Cmac_f = delta_Cmac - CL * (0.25 - xac_c) #contribution of flaps extension
Cmac = Cmac_w + Cmac_f + Cmac_fus

# Controllability Sh/S
ShS_c = xcg / (CLh/CLA_h * lh/c * (Vh_V)**2) + (Cmac/CLA_h - xac_c)/(CLh/CLA_h * lh/c * (Vh_V)**2)

print(depsilon_dalpha)


#------------------------------------- SCISSOR PLOT --------------------------------------------------

plt.figure(figsize=(10, 6))

# Plot the primary equation lines
plt.plot(xcg, ShS_s, 'b-', label='Stability Line With Safety Margin')
plt.plot(xcg, ShS_c, 'g-', label='Controllability Line')
plt.plot(xcg, ShS_s_no_margin, 'k:', label='Neutral Stability Curve')

# Shade the invalid regions red
invalid_ShS = np.maximum(ShS_s, ShS_c)
plt.fill_between(xcg, 0, invalid_ShS, color='red', alpha=0.4)

# Calculate your actual aircraft's Sh/S based on the airport manual parameters you already coded
actual_ShS = Sh / S

# Plot your actual Center of Gravity Range using the variables defined at the top
plt.plot([xcg_lower, xcg_upper], [actual_ShS, actual_ShS], 'k-', linewidth=2.5, label=f'Actual CG Range (Sh/S = {actual_ShS:.3f})')

# Optional: Add small vertical "end caps" to the CG range line to make it look highly professional
plt.plot([xcg_lower, xcg_lower], [actual_ShS - 0.005, actual_ShS + 0.005], 'k-', linewidth=2.5)
plt.plot([xcg_upper, xcg_upper], [actual_ShS - 0.005, actual_ShS + 0.005], 'k-', linewidth=2.5)

# Plot formatting
plt.title('Scissor Plot of the Aircraft')
plt.xlabel('Xcg / MAC')
plt.ylabel('Sh / S')
plt.xlim(0.0, 1.1)   # Cut off the empty space on the left and right
plt.ylim(0.0, 0.3)   # Focused the height to frame your actual Sh/S line perfectly
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


print(f'xac control {xac_c}, xac stab {xac_s}')
print(f'cruise xac wing {xac_w}, cruise xac fus {xacf1_s+xacf2_s}, cruise xac nac {xacn_s}')
print(f'appr xac wing {xac_w}, appr xac fus {xacf1_c+xacf2_c}, appr xac nac {xacn_c}')
print(f'Cmac {Cmac}')
print(f'Cmac wing {Cmac_w}, Cmac fus {Cmac_fus}, Cmac flaps {Cmac_f}')
print(f'CLh {CLh}')
print(f'CLA_h {CLA_h}')
print(f'lh {lh}')
print(f'Vh/V {Vh_V}')
print(f'CLalpha_h control {CLalpha_h_c}, CLalpha_h stab {CLalpha_h_s}')
print(f'CLalpha_a_h cruise contribution wing {CLalpha_w_s}, contribution fuselage {CLalpha_A_h_s-CLalpha_w_s}')
print(f'CLalpha_A_h control {CLalpha_A_h_c}, CLalpha_A_h stab {CLalpha_A_h_s}')
print(f'downwash gradient {depsilon_dalpha}')
print(f'wing aspect ratio {A}')




