import functions as fns  #contains all our functions, in the file "functions.py'
import numpy as np # fast numerical functions. Don't know the JS equivalent
from matplotlib import pyplot as plt# only used for sampling random colors from a colorbar
import matplotlib.gridspec as gridspec # Best way to make figures with subfigures in python
from matplotlib import rc #resource configuration. Change settings and make plots pretty.

#set the parameters for the script
rc('text', usetex = True)
rc('font', family='serif')
rc('savefig', dpi = '300')
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('xtick.major', size=2.5)
rc('ytick.major', size=2.5)
rc('xtick', direction='in')
rc('ytick', direction='in')

# define the function that we are integrating
def pf(xx, alpha, epsilon, sf):
    """
    Sum of three functions, which reproduce an analytical model for a Kadomstev
    sawtooth crash. The first line represents the field before the crash, the
    second line represents the non-axisymmetric part, and the third line
    represent the field after the crash. A complete cycle would entail the
    following for the doublet (amplitude,epsilon);
    (0,0) -> (amp_max,0) -> (amp_max,1) -> (0,1) -> (0,0).

    """
    return  fns.torodal_field_squared_q(xx, sf=sf, shear=shear, epsilon = epsilon) + \
            fns.ralf_pertII(xx, width=0.4, sf=sf, amplitude=alpha) + \
            fns.torodal_field_squared_q_final(xx, sf=sf, shear=shear, epsilon = epsilon)



#Define some parameters. These should not be changed by the user.
shear = 1 #the rate at which the field 'twists'
sf = .91  #the twisting at the very center
amp_max = 0.08 # highest value of our perturbation
line_resp1 = 100  # the number of 'pictures' in the movie
numstreams=50 #the number of field lines that are integrated (one integration can be one 'task' for the DCP
cmap =  plt.get_cmap('gnuplot') #nice colors
randoms = np.random.random(numstreams)# array full of randoms for random colors
streampoints = fns.linepoints(np.array((1.,0,.25)), np.array((1, 0, -.8)), numstreams ) #starting points (50-tuple of x,y,z coordinates) that lie on a line


# Fully parameterized plot.
parameters = np.linspace(0, -1, num=line_resp1, endpoint=False)

#alpha   =   [0.025464790894703267 - 0.016976527263135505*np.cos(4*parameter*np.pi) - 0.0033953054526271037*np.cos(8*parameter*np.pi) - 0.040000000000000036*np.sin(2*parameter*np.pi) + \
#            2.7305908429669844e-18*np.sin(4*parameter*np.pi) - 1.734723475976807e-18*np.sin(6*parameter*np.pi) - 1.843267981421152e-18*np.sin(8*parameter*np.pi) + \
#            8.673617379884035e-19*np.sin(10*parameter*np.pi) for parameter in parameters]
#epsilon =  [0.5000000000000006 - 0.4526423672846763*np.cos(2*parameter*np.pi) + 7.01943515742626e-17*np.cos(4*parameter*np.pi) - 0.022515818587186147*np.cos(6*parameter*np.pi) - \
#            3.0014068312294517e-16*np.cos(8*parameter*np.pi) - 0.008105694691387139*np.cos(10*parameter*np.pi) - 1.6653345369377348e-16*np.sin(2*parameter*np.pi) + \
#            0.053051647697298504*np.sin(4*parameter*np.pi) + 1.6653345369377348e-16*np.sin(6*parameter*np.pi) + 0.0053051647697298365*np.sin(8*parameter*np.pi) - \
#            2.220446049250313e-16*np.sin(10*parameter*np.pi) for parameter in parameters]

alphamax = 0.05
alpha =  np.linspace(0, alphamax, 0)
alpha = np.append(alpha, np.ones(20)*alphamax)
alpha = np.append(alpha, np.linspace(alphamax, 0, 20))

epsilon =  np.zeros(0)
epsilon = np.append(epsilon, np.linspace(0, 1, 20))
epsilon = np.append(epsilon, np.ones(20))


pf(np.array((1.,0.,0.)), 1., 1., 0.9)



for num, (alpha, epsilon) in enumerate(zip(alpha, epsilon)):
    print('starting poincare plot {}'.format(num))
    fig = plt.figure()
    fig.set_size_inches(10,14)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5,1], hspace =0.1)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax1.set_aspect('equal')

    xx = np.linspace(0, 0.8, 200)
    ax2.plot(xx, sf + shear*xx**2, zorder=1)
    ax2.set_yticklabels([r'$\frac{2}{3}$', r'$1$', r'$1~\frac{1}{3}$'])

    streamlines = fns.stream_multi(streampoints, vvfn=pf, tol=1e-6, sf=sf, alpha=alpha, epsilon=epsilon, iterMax=10000)

    RR = []
    zz = []
    for snum, streamline in enumerate(streamlines):
        ppoints = streamline.makePoincare(normal =  np.array([0,1,0]))
        ax1.scatter(ppoints[:,0], ppoints[:,1], s=.3, color = cmap(randoms[snum]))
        RR.append(streampoints[snum][0])
        zz.append(streampoints[snum][2])
    ax1.plot(RR, zz, color='k')

    twists = []
    locs = []
    for snum, streamline in enumerate(streamlines):
        twists.append(streamline.getTwist())
        locs.append(np.sqrt(np.sum((streampoints[snum]-np.array((1,0,0)))**2)))  # distance from equilibrium axis at
    ax2.scatter(locs, 1/np.array(twists), color=cmap(randoms), zorder = 2)
    ax1.set_ylim(-1, 1)

    ax1.text(.2, .8, r'($\alpha = {:.4f}$, $\epsilon = {:.4f}$)'.format(alpha,epsilon), fontsize=25)
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.set_xlabel(r'$R$', fontsize = 25)
    ax1.set_ylabel(r'$z$', fontsize = 25)

    ax2.set_xlabel(r'$a$', fontsize = 25)
    ax2.set_ylabel(r'$q$', fontsize = 25)
    ax2.set_ylim(0.8, 2.2)
    ax2.set_xlim(0., .8)
    ax2.set_yticks([2/3, 1, 1+1/3])
    ax2.axhline(1.0, c='k')
    ax2.axhline(2/3, c='c')

    plt.savefig('CHANGED_PERT{}.png'.format(num), bbox_inches='tight')
    plt.close()
