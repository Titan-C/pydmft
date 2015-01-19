from subprocess import call
import matplotlib.pyplot as plt
import numpy as np

def plot_qmc(**kwargs):
    input_data = {
        'dtaureal' : 0.5,
        'U': 4.0,
        'D': 1,
        'nloop': 10,
        'dmu': 0.01,
        'nmu': 1,
        'xmu0': 0.0,
        'if1': 1,
        'du': 1.5,
        'nu': 1,
        'nsweep': 5000,
        'nsweep0': 5000,
        'iread': 0,
        'imod': 1,
        'imet': 1,
        'ndirty': 500
        }

    if kwargs is not None:
        for key, value in kwargs.iteritems():
            input_data[key] = value

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
#    exe = call(['./a.out'])

    f, (ax1, ax2) = plt.subplots(2, sharex=True)

    imG = np.loadtxt('fort.60').reshape(input_data['nmu'],-1,2)
    reG = np.loadtxt('fort.61').reshape(input_data['nmu'],-1,2)
    ax1.set_title('Green\'s Functions')
    for i in range(input_data['nmu']):
        ax1.plot(imG[i,:, 0], imG[i,:, 1], 'o-', label='Im G{}'.format(i))
        ax1.plot(reG[i,:, 0], reG[i,:, 1], 'o-', label='Re G')
    ax1.legend()

    reS = np.loadtxt('fort.63')
    imS = np.loadtxt('fort.64')
    ax2.set_title('Self Energy')
    ax2.plot(imS[:, 0], imS[:, 1], 'o-', label=r'Im $\Sigma$')
    ax2.plot(reS[:, 0], reS[:, 1], 'o-', label=r'Real $\Sigma$')
    ax2.legend()

    f.suptitle('U={}, $\beta$=16'.format(input_data['U']))
    ax2.set_xlabel(r'$i\omega_n$')

    return exe

plot_qmc()