#testing the examples before gallery
import subprocess
import dmft.plot.hf_single_site as pss

def test_example():
    command = "examples/Hirsh-Fye/single_site.py -sweeps 1000 -therm 400 -Niter 6 -ofile /tmp/testhfss"
    command = command.split()
    print(subprocess.call(command))
    pss.show_conv(4, 'U2.5', '/tmp/testhfss', xlim=8)
