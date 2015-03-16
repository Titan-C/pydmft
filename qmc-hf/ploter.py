from subprocess import call
import matplotlib.pyplot as plt
import numpy as np

def plot_qmc(**kwargs):
    input_data = {
        'dtaureal' : 0.5,
        'U': 2.,
        'D': 1,
        'nloop': 4,
        'dmu': 0.01,
        'nmu': 1,
        'xmu0': 0.3,
        'if1': 1,
        'du': 1.5,
        'nu': 1,
        'nsweep': 10000,
        'nsweep0': 15000,
        'iread': 0,
        'imod': 1,
        'imet': 0,
        'ndirty': 500
        }

    if kwargs is not None:
        input_data.update(kwargs)

    text = """

{dtaureal},{U}
{D},{nloop},{dmu},{nmu},{xmu0},{if1}
{du},{nu},{nsweep},{nsweep0}
{iread},{imod},{imet}
{ndirty}

if1=1 does not compute <SzSz>
   =0 does it
nsweep0 is nsweep for last iteration
imet=1 makes a metallic seed
    =0 makes an insulatin one""".format(**input_data)

    with open('fort.50', 'w') as f:
        f.write(text)
#    exe = call(['./qmc'])

    f, (ax1, ax2) = plt.subplots(2, sharex=True)

    imG = np.loadtxt('fort.60').reshape(2,-1,2)
    reG = np.loadtxt('fort.61').reshape(2,-1,2)
    ax1.set_title(r'U={}, $\beta$=16'.format(input_data['U']))
    for i in range(2):
        ax1.plot(imG[i,:, 0], imG[i,:, 1], 'o-', label='Im G {}'.format(i))
        ax1.plot(reG[i,:, 0], reG[i,:, 1], 'o-', label='Re G {}'.format(i))
    ax1.legend()
#    ax1.set_ylim([-1.55,0])

    reS = np.loadtxt('fort.63').reshape(2,-1,2)
    imS = np.loadtxt('fort.64').reshape(2,-1,2)
    ax2.set_title('Self Energy')
    for i in range(2):
        ax2.plot(imS[i,:, 0], imS[i,:, 1], 'o-', label=r'$\mu$={} Im $\Sigma$ {}'.format(input_data['xmu0'], i))
        ax2.plot(reS[i,:, 0], reS[i,:, 1], 'o-', label=r'$\mu$={} Re $\Sigma$ {}'.format(input_data['xmu0'], i))
    ax2.legend()

    ax2.set_xlabel(r'$i\omega_n$')
#    ax2.set_xlim([-4, 8])
    f.tight_layout()
#    ax2.set_ylim([-0.8,0])

    g = plt.figure()
    gtau = np.loadtxt('fort.3').reshape(2,65,2)

    plt.semilogy(gtau[0,:,0], gtau[0,:,1],'+-', label='$\\mu$={} $G$'.format(input_data['xmu0']))
    plt.semilogy(gtau[1,:,0], gtau[1,:,1],'+-', label=r'$\mu$={} G($\tau$)'.format(input_data['xmu0']))
    plt.xlim([0,16])
#    plt.ylim([-0.6,-0.0])
    plt.title(r'U={}, $\beta$=16'.format(input_data['U']))
    plt.legend(loc=0)
    g.tight_layout()

#    return exe
plot_qmc()