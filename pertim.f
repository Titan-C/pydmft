From marcelo@physics.rutgers.edu Mon Nov 21 11:07:52 1994
Received: from nef.ens.fr by peterpan.ens.fr (4.1/88/01/19 3.0)
 	id AA01750; Mon, 21 Nov 94 11:07:49 +0100
Return-Path: <marcelo@physics.rutgers.edu>
Received-Date: Mon, 21 Nov 94 11:07:49 +0100
Received: from pion.rutgers.edu by nef.ens.fr (5.65c8/ULM-1.0)
	Id AA15965 ; Mon, 21 Nov 1994 11:07:48 +0100
Received: from electron.rutgers.edu by pion.rutgers.edu (5.59/SMI4.0/RU1.5/3.08) 
	id AA27867; Mon, 21 Nov 94 05:07:41 EST
Received: by electron.rutgers.edu (5.59/SMI4.0/RU1.5/3.08) 
	id AA25138; Mon, 21 Nov 94 05:07:40 EST
Date: Mon, 21 Nov 94 05:07:40 EST
From: marcelo@physics.rutgers.edu (Marcelo Rozenberg)
Message-Id: <9411211007.AA25138@electron.rutgers.edu>
To: marcelo@physics.rutgers.edu
Status: R

c 2nd Order Perturbation Theory (a la Yamada Yosida)
c for the oo-d Hubard Model in Matsubara space.
c Produces the local Green function and the Selfenergy.
c Also calculates the kinetic T, potential V, 
c and total E energies, and the double occupation.
c Has to be linked to IMSL
c#########################################################


 	parameter(L=16384)
 	implicit real*8(a-h,o-z)

 	double complex xi,ep,om,om1,om2,d2,one,root
 	double complex fg0(2*L),fg0b(2*L)      
 	double complex fg0t(2*L),fg(2*L)      
 	double complex sefb(2*L),seff(2*L),self(2*L)


c******* output data **********************************************
c fort.30=Imaginary part of Selfenergy
c fort.60=Imaginary part of G 
c fort.80=kinetic energy T as a function of temperature
c fort.81=potential energy V as a function of temperature
c fort.82=total energy E as a function of temperature
c fort.83=double occupation as a function of temperature
c fort.90=kinetic energy T as a function of U
c fort.91=potential energy V as a function of U
c fort.92=total energy E as a function of U
c fort.93=double occupation as a function of U


c******* input data (fort.70) **************************************
c L=number of frequency points
c t0=initial temperature
c u0=initial interaction strength
c dt=step for the temperature loop
c du=step for the interaction loop
c nl=number of temperature/interaction loops
c d=half-bandwidth
c nloop=number of iteration loops
c nl0=number of final loops to be printed-out
c imet=selects initial seed (1=metallic, 0=insulating)
c isie=flag for computation of energies (isie=1 does it, isie=0 does not)
c itu=selects type of loop (itu=0  U loop, itu=1  temperature loop)

        read(70,*)t0,u0,dt,du,nl
        read(70,*)d,nloop,nl0,imet,isie,itu

c*******define some constants*******

        ep=.0001*(1.,0.)
        d2=d*(1.,0.)
        d2=d2**2
        pi=3.141592653589793
	xi=(0.,1.)
        one=(1.,0.)
	u=u0
	t=t0

	do 301 i=1,2*L
	   fg0(i)=(0.,0.)	
	   fg(i)=(0.,0.)
 301	continue
c*******************************************************************


c*******temperature/U loop starts here***********************

	do 222 il=1,nl
	if(itu.eq.1)then
	t=t0+dt*(float(il-1))
	else
        u=u0+du*float(il-1)
	endif

	dtau=1./t/float(L)
	binv=1./dtau/float(L)



c******** the iteration loop starts here ***********************

        do 103 iloop=1,nloop


c****g0 is the non interacting GF in real frequency space******

c**on first loop compute a seed (imet=1 metallic, imet=0 insulating)

	if(iloop*iu*it.eq.1)then
        do 101 i=1,L
           om=xi*(2.*float(i)-1.-float(L))*pi*binv
           fg0(2*i)=one/om
        if(imet.eq.1)then
           root=cdsqrt((om+ep)**2-d2)
           sig=1.
           if(dimag(om)*dimag(root).lt.0.)sig=-1.
           fg0(2*i)=2.*one/(om+(cdsqrt((om+ep)**2-d2)))
        endif
 101    continue

	else

        do 102 i=1,L
           om=xi*(2.*float(i)-1.-float(L))*pi*binv
           om1=om+self(2*i)
           om2=om-self(2*i)
 	   root=cdsqrt((om2+ep)**2-d2)
	   sig=1.
 	   if(dimag(om)*dimag(root).lt.0.)sig=-1.
           fg0(2*i)=2.*one/(om1+sig*root)
	   fg(2*i)=2.*one/(om2+sig*root)
 102    continue
	
	endif

c** fg0(i) is the non interacting GF in Matsubara space **

        call dfftcb(2*L,fg0,fg0t)
 
	ex=-1.
        do 82 i=1,2*L
	ex=-ex
           fg0t(i)=binv*fg0t(i)*ex
 82      continue

	do 83 i=1,L
           fg0b(i)=fg0t(i+L)
 83      continue
        
	do 84 i=L+1,2*L
           fg0b(i)=fg0t(i-L)
 84      continue
        
c** fg0b(i) is the non interacting GF in time **



c***calculate the selfenergy in 2OPT****************

	do 520 i=1,L
	 sefb(i+L)=u**2*fg0b(i+L)**3
520	continue

	do 525 i=1,L
	 sefb(L+1-i)=-sefb(i+L)
525	continue

        call dfftcf(2*L,sefb,seff)

	ex=-1.
	do 530 i=1,2*L
	ex=-ex
           seff(i)=.5*dtau*ex*seff(i)
530      continue

	do 540 i=1,L
           self(i)=seff(i+L)
540      continue
        
	do 550 i=L+1,2*L
           self(i)=seff(i-L)
550     continue

c***self is the self-energy in Matsubara space******



c************print output*******************************

 	   x=pi/(L*dtau)
	if(iloop.ge.nloop-nl0+1)then

 	do 106 i=L-300,L+300,2
           si=dimag(self(i))
 	   g1=dimag(fg(i))
           write(30,*)real(x*(i-L-1)),real(si)
           write(60,*)real(x*(i-L-1)),real(g1)
 106      continue
 	write(30,*)'   '
 	write(60,*)'   '
 	endif

c********************************************************


c*******close iteration loop*********************************

 103    continue



c******************get the energies****************************

c** E=tum,T=tuma,V=tum-tuma***

	if(isie.eq.1)then

	sum=0.
	tum=0.
	tuma=0.
	do 111 i=1,L
           oma=(2.*float(i)-1.-float(L))*pi*binv
	sg=1.
	if(oma.lt.0.)sg=-1.
	omb=oma*2.
 	tum=tum+.5*dimag(fg(2*i))*dimag(self(2*i))
     $ -(dimag(fg(2*i))+2./(oma+sg*dsqrt(oma**2+d**2)))*oma
	tuma=tuma+dimag(fg(2*i))*dimag(self(2*i))
     $ -(dimag(fg(2*i))+2./(oma+sg*dsqrt(oma**2+d**2)))*oma

111	continue
 
	free=0.
	e=-d
	de=d/1000.
	do 666 i=1,2000
 	 free=free+de*e*dsqrt(1.-(e/d)**2)/(dexp(e/t)+1.)
	 e=e+de
666	continue	
	free=free*2./(pi*d)

	tum=t*tum+free
	tuma=t*tuma+free




	if(itu.eq.1)then

c***print-out energies as function of temperature***

c**kinetic**
 	write(80,*)real(binv),real(tuma)
c**potential**
 	write(81,*)real(binv),real(tum-tuma)
c**total**
	write(82,*)real(binv),real(tum)
c**<Nup*Ndw>=<D> double ocupation**
 	write(83,*)real(binv),real((tum-tuma)*2./u+.25)

	else

c***print-out energies as function of U***

c**kinetic**
 	write(90,*)real(u),real(tuma)
c**potential**
 	write(91,*)real(u),real(tum-tuma)
c**total**
 	write(92,*)real(u),real(tum)
c**<Nup*Ndw>=<D> double ocupation**
 	write(93,*)real(u),real((tum-tuma)*2./u+.25)

	endif

c**************************************************************

	endif




c*******close temperature/U loop********************************

222	continue

	stop
	end


