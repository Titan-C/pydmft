c idem to ht1.f code working with num rec
c speed=96Mflops
c this is the fastest code to date. note: use B-SPLINE
c agrees with the previous but only within the QMC error
c gives the same physics. the difference arises from using
c a b-spline instead of a spline. 4-16-93
c I'm modifying this code to make it faster. the good one is
c hubb.f in /mu!!!!!!!!!!!!!!!!!!!!

c 2/15/93 this program is identical to mu.f and it is
c been cleaned up and optimized a little.
c as says below the vertex calculation is not fully tested
c 12/18/92 this program calculates the paramagnetic
c solution of the hubbard model at infinite d
c also gives the charge and spin suceptibilities
c 10=Xch,20=Xz,30=Xp,40=Xx,43=Xz(ising),92=nu+nd,90=n
c it is not finished!!!!!!!!!!!!!!(means not fully tested)
c#########################################################
c***** main program ******************

      include 'parameter1.h'

c	parameter(L=32768,Lfak=8)
c	parameter(L=4096,Lfak=8)
c	parameter(L=32768,Lfak=64)
c	parameter(L=32768,Lfak=16)
c	parameter(L=65536,Lfak=128)
c 	parameter(L=32768,Lfak=128)
c	parameter(L=32768,Lfak=100)
	implicit real*8(a-h,o-z)
        double complex xi,d2,one,ome
        double complex fg0(2*L),fg0f(2*L),fg0b(2*L)
        double complex fg0t(2*L),a(2*L)
        double complex fgb(2*L),fgf(2*L),self(2*L),self0(2*L)
        double precision g0(-L:L),g(-L:L),dumg(2*L),dumg0(2*L)
        double precision dumg1(2*L),dumg01(2*L)
        double precision g00(-Lfak:Lfak),gtmp(-Lfak:Lfak)
        double precision v(Lfak)
	common/sweep/nsweep,nsweep0,ndirty
	common/fields/v
c       COMMON /WORKSP/  RWKSP
c       REAL RWKSP(16660)
c       CALL IWKIN(16660)



c******* input data ***********
        read(50,*)dtaureal,u0
        read(50,*)d,nloop,dmu,nmu,xmu0,if1
        read(50,*)du,nu,nsweep,nsweep0
        read(50,*)iread,imod,imet
	read(50,*)ndirty

        d2=d*(1.,0.)
        d2=d2**2
        dtau=dtaureal*float(Lfak)/float(L)
	if=1
c	pi=3.141592654
 	pi=datan(1.d0)*4.d0
	xi=(0.,1.)
        one=(1.,0.)
	beta=dtau*float(L)
	temp=1./beta
	xL=float(L)
	u=u0
	xmu=xmu0



	do 301 i=1,2*L
	   fg0(i)=(0.,0.)
 301	continue


c***choose or generate an input seed********
	if(iread.eq.1)then

	do i=-Lfak,Lfak
	read(4,*)x,g00(i)
	enddo

	do i=-Lfak,Lfak
	read(4,*)x,gtmp(i)
	enddo

	do i=1,Lfak
	read(4,*)v(i)
	enddo

	else

c******* first does a run of the impurity **********
c******* g0 is initialized here and the impurity ***
c******* problem is solved *************************

        do 101 i=1,L
           ome=(xi*pi/xL/dtau)*(2.*float(i)-1.-xL)
	   ome=ome-xmu
           sq=dimag(cdsqrt((ome)**2-d2))
	   w=dimag(ome)
	   sig=sq*w/dabs(sq*w)
	if(imet.eq.1)then
           fg0(2*i)=2./(ome+sig*(cdsqrt((ome)**2-d2)))
	else
         fg0(2*i)=1./ome
c          self(2*i)=u**2/4./ome
c          ome=ome-self(2*i)
c          sq=dimag(cdsqrt((ome)**2-d2))
c          w=dimag(ome)
c          sig=w*sq/dabs(w*sq)
c          fgf(2*i)=2./(ome+sig*(cdsqrt(ome**2-d2)))
c          fg0(2*i)=ome-d2*fgf(2*i)

	endif
 101    continue
	if(imet.ne.1)then
c	write(6,*),'ojo con la seed!'
	endif

c***********************************
c fg0(i) has the frecuency spectrum
c from -pi to pi
c***********************************


c for num rec
        call four1(fg0,2*L,1)
        do i=1,2*L
        fg0t(i)=fg0(i)
        enddo

        do 2 i=1,L
           fg0t(2*i)=-temp*fg0t(2*i)
           fg0t(2*i-1)=temp*fg0t(2*i-1)
 2      continue

	do 3 i=1,L
           fg0b(i)=fg0t(i+L)
	   fg0b(L+i)=fg0t(i)
 3      continue


c******************************
c  fg0b(i) has the real time GF
c  from -beta to beta
c******************************

        do 11 i=1,2*L
           g0(i-L-1)=dreal(fg0b(i))
 11     continue

c*******************************************
c here goes the trick that takes care of the
c discontinuity in g(tau)
c*******************************************
        g0(0)=g0(0)+.5
        g0(-L)=-g0(0)

c**** make sure that L can be divided by Lfak****
	call extract(g0,g00,L,Lfak)

c***initailize the pseudospin fields*******
	polar=.5
      z=dtaureal*u/2
      z=exp(z)
      xlam=log(z+sqrt(z**2-1.))
c     write(6,*)u,z,xlam
 	call initial(xlam,v,polar,Lfak)
	if2=0
        call impurity(g00,gtmp,dtaureal,u,if,if1,if2)

c**** this endif comes from the line where an input seed
c**** (g00 and gtmp) is read from file fort.4
	endif

	call interp(gtmp,g)
	call interp(g00,g0)


c**********************
c here "undo" the trick
c**********************
       g(0)=g(0)-.5
       g0(0)=g0(0)-.5
       g(-L)=-g(0)
       g0(-L)=-g0(0)

        do 12 i=1,2*L
           fgb(i)=g(i-L-1)*(1.,0.)
           fg0b(i)=g0(i-L-1)*(1.,0.)
 12     continue


        call four1(fg0b,2*L,-1)
        call four1(fgb,2*L,-1)
        do i=1,2*L
        fg0f(i)=fg0b(i)
        fgf(i)=fgb(i)
        enddo



	do 302 i=1,2*L
	   self(i)=(0.,0.)
 302	continue

	do 5 i=1,L
           fg0f(2*i)=(-dtau/2.)*fg0f(2*i)
           fgf(2*i)=(-dtau/2.)*fgf(2*i)
           self0(2*i)=one/fg0f(2*i)-one/fgf(2*i)
 5      continue

	do 407 i=1,L
           self(i)=self0(i+L)
           self(i+L)=self0(i)
           a(i)=fg0f(i+L)
           a(i+L)=fg0f(i)
 407      continue


c************************
c self(i) is the 1st self
c energy from -pi to pi
c************************





c******* here starts the u-loop without reset ************

	do 555 iu=1,nu
	u=u0+du*float(iu-1)

c	if(iu.gt.1) then
      	z=dtaureal*u/2
      	z=exp(z)
      	xlam=log(z+sqrt(z**2-1.))

	do i=1,Lfak
	v(i)=xlam*v(i)/dabs(v(i))
	enddo
c	endif

c******* here starts the mu-loop without reset ************

	do 432 imu=1,nmu
	xmu=xmu0+dmu*float(imu-1)

c******* here starts the iteration ***************

        do 103 iloop=1,nloop

c this is crucial!!!!!!!!!!
c otherwise it just gathers shit in the even freqs...
        do i=1,L
        fg0(2*i-1)=(0.d0,0.d0)
        enddo

	if(iloop.eq.nloop)if=0
c if iread=1 jumps to call impurity
	if(iread.eq.1.and.iloop.eq.1) goto 444
c	write(6,*)'paso'
        do 102 i=1,L
           ome=(xi*pi/xL/dtau)*(2.*float(i)-1.-xL)
           ome=ome-self(2*i)-xmu
           sq=dimag(cdsqrt((ome)**2-d2))
	   w=dimag(ome)
	   sig=w*sq/dabs(w*sq)
           fgf(2*i)=2./(ome+sig*(cdsqrt(ome**2-d2)))
 102    continue

c***********************************
c fgf(i) has the frecuency spectrum
c from -pi to pi
c***********************************


        do 104 i=1,L
           fg0(2*i)=one/(one/fgf(2*i)+self(2*i))
 104    continue

c***********************************
c fg0(i) has the frecuency spectrum
c from -pi to pi
c***********************************


c for num rec
        call four1(fg0,2*L,1)
        do i=1,2*L
        fg0t(i)=fg0(i)
        enddo


        do 82 i=1,L
           fg0t(2*i)=-temp*fg0t(2*i)
           fg0t(2*i-1)=temp*fg0t(2*i-1)
 82      continue

	do 83 i=1,L
           fg0b(i)=fg0t(i+L)
           fg0b(L+i)=fg0t(i)
 83      continue


c******************************
c  fg0b(i) has the real time GF
c  from -beta to beta
c******************************

        do 811 i=1,2*L
           g0(i-L-1)=dreal(fg0b(i))
 811     continue

c*********************************
c here goes the trick to take care
c of the discontinuity in g(tau)
c*********************************
	g0(0)=g0(0)+.5
        g0(-L)=-g0(0)

	call extract(g0,g00,L,Lfak)
c if iread=1 goto here directly
444	continue
	if2=iloop
        call impurity(g00,gtmp,dtaureal,u,if,if1,if2)
	call interp(gtmp,g)
	call interp(g00,g0)


c	if(mod(iloop,imod).eq.0)then
 	if(iloop.ge.nloop-1)then

 	do 1108 i=0,Lfak
           write(12,*)real(dtaureal*i),real(gtmp(i))
           write(13,*)real(dtaureal*i),real(g00(i))
 1108    continue
           write(12,*)'  '
           write(13,*)'  '

c	endif
 	endif

 	if(iloop.eq.nloop)then

 	do i=-Lfak,Lfak
           write(3,*)real(dtaureal*i),real(g00(i))
	enddo

 	do i=-Lfak,Lfak
           write(3,*)real(dtaureal*i),real(gtmp(i))
	enddo

c	do i=1,Lfak
c	write(3,*)v(i)
c	enddo
	endif

c**********************
c here "undo" the trick
c**********************
       g(0)=g(0)-.5
       g0(0)=g0(0)-.5
       g(-L)=-g(0)
       g0(-L)=-g0(0)


        do 812 i=1,2*L
           fgb(i)=g(i-L-1)*(1.,0.)
           fg0b(i)=g0(i-L-1)*(1.,0.)
 812     continue


c for num rec
        call four1(fg0b,2*L,-1)
        call four1(fgb,2*L,-1)
        do i=1,2*L
        fg0f(i)=fg0b(i)
        fgf(i)=fgb(i)
        enddo


	do 85 i=1,L
           fg0f(2*i)=(-dtau/2.)*fg0f(2*i)
           fgf(2*i)=(-dtau/2.)*fgf(2*i)
           self0(2*i)=one/fg0f(2*i)-one/fgf(2*i)
 85      continue

	do 409 i=1,L
           self(i)=self0(i+L)
           self(i+L)=self0(i)
           dumg(i)=dimag(fgf(i+L))
           dumg(i+L)=dimag(fgf(i))
           dumg0(i)=dimag(fg0f(i+L))
           dumg0(i+L)=dimag(fg0f(i))
           dumg1(i)=dreal(fgf(i+L))
           dumg1(i+L)=dreal(fgf(i))
           dumg01(i)=dreal(fg0f(i+L))
           dumg01(i+L)=dreal(fg0f(i))
 409      continue


c***************************
c self(i) is the updated
c self energy from -pi to pi
c***************************



c	if(mod(iloop,imod).eq.0)then
 	if(iloop.ge.nloop-1)then

c          write(30,*)'"mu=',real(xmu),'"'
c          write(130,*)'"mu=',real(xmu),'"'
c          write(40,*)'"mu=',real(xmu),'"'
c          write(140,*)'"mu=',real(xmu),'"'
c          write(60,*)'"mu=',real(xmu),'"'
c          write(61,*)'"mu=',real(xmu),'"'
  	do 106 i=L-4*Lfak,L+4*Lfak,2
c           si=dimag(self(i))
c           si1=real(self(i))
 	xa=real(pi/beta*float(i-L-1))
c          write(30,*)xa,real(si)
c           write(130,*)xa,real(si1)
c           write(40,*)xa,real(dumg0(i))
c           write(140,*)xa,real(dumg01(i))
            write(60,*)xa,real(dumg(i))
            write(61,*)xa,real(dumg1(i))
            write(63,*)xa,real(self(i))
            write(64,*)xa,imag(self(i))
  106      continue
c          write(30,*)'  '
c          write(130,*)'  '
c          write(40,*)'  '
c          write(140,*)'  '
           write(60,*)'  '
           write(61,*)'  '

c	endif
 	endif

c	write(61,*)iloop,real(dumg(L))

	if(iloop.ge.nloop-1)then
	   write(90,*)real(xmu),real(-gtmp(0))+.5
	end if

 103    continue

432	continue

 555	continue


	stop
	end


c###########################################################
c*** this subroutine is only used to define a g0(tau) *****
c*** if want to do the impurity problem *******************

      subroutine getg0(g0,dtau,u,L)
	implicit real*8(a-h,o-z)
      double precision g0(-L:L)

c number of sums in the discrete integration
	max=1000
	delta=.5
c de: integration interval, pmax, integration limit.
	de=.1
	pmax=max*de

	beta=L*dtau
      do 1 i=-L,L

	  dt=dtau*i

c define theta
	 if(i.ge.0)then
	  theta=1.
	 else
	  theta=0.
	 endif

	 eps=-pmax
         sum=0.
	 do 10 j=1,2*max

c assuming the fermi surface is at 0.
	  f=1./(exp(beta*eps)+1.)
c assuming symmetric case
	  width=eps**2+delta**2
	if(-eps*dt.gt.700.)then
	aintgd=0.
	else
	  aintgd=exp(-eps*dt)*(f-theta)/width
	endif
	  sum=sum-aintgd
	  eps=eps+de
10	 continue
	 g0(i)=de*delta*sum/3.14159

 1    continue

      return
      end

c#########################################################
	subroutine extract(g0,g00,L,Lfak)
	implicit real*8(a-h,o-z)
	double precision g0(-L:L),g00(-Lfak:Lfak)

c******************************************************
c be careful about  L and  Lfak
c extract from g0 which has L pts only Lfak pts for g00
c******************************************************


c     g00(0)=g0(0)
c
c     nrat=L/Lfak
c     do 20 i=1,Lfak
c g00(i)=g0(i*nrat)
c20   continue
c
c     do 21 i=1,Lfak
c g00(-i)=-g00(Lfak-i)
c21   continue

c     return
c     end

c this is a new extract routine from htnrNNnb.f that
c I took from the PRE\gabi\mu directory

      g00(0)=g0(0)
      g00(Lfak)=1.d0-g0(0)


c this takes beter care of the possible mismatch
c between L and Lfak
c     nrat=L/Lfak
        xnrat=dfloat(L)/dfloat(Lfak)

      do 20 i=1,Lfak-1
        xx=i*xnrat
        ix=idint(xx)
        g00(i)=g0(ix)
c       print*,i,g00(i),ix,g0(ix)
c        g00(i)=g0(i*nrat)
 20   continue

      do 21 i=1,Lfak
         g00(-i)=-g00(Lfak-i)
 21   continue

      return
      end


c###########################################################
      subroutine interp(gtmp,g)
	implicit real*8(a-h,o-z)
      include 'parameter1.h'

c     parameter(Lfak=128,Lfak1=128+1,Lfak3=128+3)
c     parameter(Lfak=100,Lfak1=100+1,Lfak3=100+3)
c     parameter(Lfak=80,Lfak1=80+1,Lfak3=80+3)
c     parameter(Lfak=64,Lfak1=64+1,Lfak3=64+3)
c     parameter(Lfak=16,Lfak1=16+1,Lfak3=16+3)
c     parameter(Lfak=8,Lfak1=8+1,Lfak3=8+3)
c     integer nintv
      double precision gtmp(-Lfak:Lfak),g(-L:L)
      double precision xa(Lfak1),ya(Lfak1)
c for num rec
	double precision y2(Lfak1)
c for imsl
c     double precision break(Lfak1),cscoef(4,Lfak1)
c for cray
c	dimension bcoef(Lfak3),T(Lfak3+4)
c	dimension w1(3*Lfak3),w(5*Lfak3)
c for imsl
c     external dcsint,dcsval

c*********************************************************
c interpolate gtmp which has Lfak pts to g which has L pts
c*********************************************************

      do 10 i=1,Lfak1
	 xa(i)=float(i-1)/float(Lfak)
	 ya(i)=gtmp(i-1)
 10   continue
c******note:  Lfak1=Lfak+1******

c for cray
c	call bint4(xa,ya,Lfak1,2,2,0.,0.,1,T,bcoef,Lfak3,4,w)
c for imsl
c	call dcsint(Lfak1,xa,ya,break,cscoef)
c for num rec
        call spline(xa,ya,Lfak1,0.d0,0.d0,y2)


c******* assign g(i)********

c for imsl
c	nintv=Lfak
c	inbv=1
	do 20 i=1,L
           x=float(i)/float(L)
c for cray
c         g(i)=bvalu(T,bcoef,Lfak3,4,0,x,inbv,w1)
c for imsl
c          g(i)=dcsval(x,nintv,break,cscoef)
c for num rec
	call splint(xa,ya,y2,Lfak1,x,y)
	g(i)=y
 20     continue

	g(0)=gtmp(0)

	do 40 i=1,L
           g(-i)=-g(L-i)
 40     continue

	return
	end

c############################################################
      subroutine impurity(g0,g,dtau,u,if,if1,if2)
	implicit real*8(a-h,o-z)
      include 'parameter2.h'

c     parameter(L=64)
c     parameter(L=16)
c     parameter(L=128)
c     parameter(L=100)
c     parameter(L=80)
c     parameter(L=8)
      double precision gup(L,L),gdw(L,L),v(L)
      double precision g0(-L:L),del(L,L),g(-L:L)
      double precision gstup(L,L),gstdw(L,L)
      double precision xgu(-L:L),xgd(-L:L),xg(-L:L),xga(-L:L)
	dimension gx(L,L)
        dimension xs(0:L),xgt(-L:L)
c        dimension ach(0:L),az(0:L),xgt(-L:L)
c        dimension ap(0:L),ax(0:L)
c        dimension vch(L,L),vz(L,L)
c        dimension wch(L,L),wz(L,L)
c        dimension vs(L,L),vt(L,L)
c        dimension T(33)
c*********************************************************
c       dimension gu(-L:L,3600),gd(-L:L,3600),s(2*L,3600)
c       dimension gus(-L:L,-L:L,3600),gds(-L:L,-L:L,3600)
c****if calculating vertex comment out the next 2 lines***
c****and uncomment out the previous 2 lines***************
c****this is necesary to save memory space****************
        dimension s(2*L,1)
c       dimension gu(-L:L,1),gd(-L:L,1),s(2*L,1)
c       dimension gus(-L:L,-L:L,1),gds(-L:L,-L:L,1)
c*********************************************************
c     real rnunf
c     external rnset,rnunf
	common/sweep/nsweep,nsweep0,ndirty
	common/fields/v

c     iseed=123457
      if(if2.eq.0) idum=-123457
      polar=.5
c     nsweep=3000
c***select here the number of sweeps for the last loop***
	ia=1
	if(if.eq.0)then
	ia=0
	nsweep=nsweep0
	if=1
	endif
c********************************************************
c     ndirty=1000
      ncor=2
      nwarm=500
        docc=0.d0

      do 11 j=1,L
         do 12 i=1,L
            del(i,j)=0.
            del(i,i)=1.
 12      continue
 11   continue
      do 800 j=1,L
            xgt(j)=0.
            xgt(-j)=0.
         do 800 i=1,L
            gstup(i,j)=0.
            gstdw(i,j)=0.
 800  continue
            xgt(0)=0.
      z=dtau*u/2
      z=exp(z)
      xlam=log(z+sqrt(z**2-1.))
        if(u.eq.0.)u=.001
        xxx=1./(1.-dexp(-dtau*u))
c	write(6,*)'u',u,'xxx',xxx
c initialize
c     call rnset(iseed)
      r=ran1(idum)
      g0(-L)=-g0(0)

	do 567 ir=1,L
	do 567 is=1,L
	gx(is,ir)=g0(is-ir)
567	continue

c	call initial(xlam,v,polar,L)


      call gnewclean(gup,v,gx,del,1.d0)
      call gnewclean(gdw,v,gx,del,-1.d0)


c********************************************
c does a total of nsweeps
c a clean update comes after
c ndirty dirty updates
c*********************
c kx= # of measurements
c irr= # of accepted flips (after the warm-up)
c nrat= # of negative det encountered
c********************************************


      kx=0
      irr=0
      nrat=0
      do 2 k=1,nsweep
         kk=mod(k,ndirty)
         kcor=mod(k,ncor)
         do 5 j=1,L
            dv=2.*v(j)

c****calculates the determinant ratio***********
            ratup=1.+(1.-gup(j,j))*(exp(-dv)-1.)
            ratdw=1.+(1.-gdw(j,j))*(exp(dv)-1.)
            rat=ratup*ratdw
c	print*,ratup,ratdw

            if(rat.lt.0.)then
               nrat=nrat+1
            end if
            rat=rat/(1.+rat)
c for num rec
	r=ran1(idum)
c for imsl
c	     r=rnunf()
c for cray
c        r=runif(T,32)
            if(rat.gt.r)then
               if(k.gt.nwarm)then
                  irr=irr+1
               end if
               v(j)=-v(j)
               if(kk.eq.0)then
                  call gnewclean(gup,v,gx,del,1.d0)
                  call gnewclean(gdw,v,gx,del,-1.d0)
                  goto 100
               endif
               call gnew(gup,v,j,del,1.d0)
               call gnew(gdw,v,j,del,-1.d0)
            endif
  100      continue
5        continue


c*** store the measurements*****************

	if(kcor.eq.0.and.k.gt.nwarm)then
	kx=kx+1
c******store the g's************************
	do 333 jx=1,L
           do 333 ix=1,L
              gstup(ix,jx)=gstup(ix,jx)+gup(ix,jx)
              gstdw(ix,jx)=gstdw(ix,jx)+gdw(ix,jx)
653	continue
333	continue


c***compute de double occup******
c	if(ia.eq.0)then
        do ix=1,L
       docc=docc+gup(ix,ix)*gdw(ix,ix)
        enddo
c	endif
c**********************************


	if(if1.eq.1)goto 654
c******store the Ising spins****************
        do 557 iq=1,L
        s(iq,kx)=v(iq)/abs(v(iq))
        s(iq+L,kx)=s(iq,kx)
557     continue
654	continue

 	endif
c******************************************************

 2    continue

      write(2,*)'acc. rate:',real(float(irr)/float((nsweep-nwarm)*L))
      write(2,*)'# of neg det:',nrat



c**** get g(-L:L) by averaging the matrices up & dw********
      kxmax=kx

c*** normalize the sum of the g's*********

	do 334 jx=1,L
           do 334 ix=1,L
              gstup(ix,jx)=gstup(ix,jx)/float(kxmax)
              gstdw(ix,jx)=gstdw(ix,jx)/float(kxmax)
 334  continue

c***and the double occupation*****
        docc=docc/dfloat(kxmax*L)
	write(81,*)u,docc
	if(ia.eq.0)then
	write(80,*)u,docc
	endif
c*******************

c**** wrap-around***********

        do 601 j=0,L-1
           xgd(-j)=0.
           xgu(-j)=0.
           do 602 i=1,L-j
                 xgd(-j)=xgd(-j)+gstdw(i,i+j)
                 xgu(-j)=xgu(-j)+gstup(i,i+j)
 602       continue
           xga(-j)=.5*(xgu(-j)+xgd(-j))
 601    continue

        do 604 j=1,L-1
           xgd(j)=0.
           xgu(j)=0.
           do 605 i=1,L-j
                 xgd(j)=xgd(j)+gstdw(i+j,i)
                 xgu(j)=xgu(j)+gstup(i+j,i)
 605       continue
           xga(j)=.5*(xgu(j)+xgd(j))
 604    continue
        do 606 i=1,L-1
           xg(i)=(xga(i)-xga(i-L))/float(L)
           xg(i-L)=-xg(i)
 606    continue
        xg(0)=xga(0)/float(L)
        xg(-L)=-xg(0)

c**** this is just a trick for the interpolation
        xg(L)=1.-xg(0)
        g0(L)=1.-g0(0)
c***********************************************

        do 556 i=-L,L
           g(i)=xg(i)
c       if(iloop.eq.nloop)then
c          write(85,*)i,real(g(i))
c          write(84,*)i,real(g0(i))
c	end if
 556    continue



	if(if.eq.1)goto 777

         if(if1.eq.0)then
c***********get chis***************

c***first from the ising fields****

        xsa=0.
        xs2=0.
        do 587 ii=0,L
        xs(ii)=0.
587     continue
        do 558 iq=1,kxmax
        do 559 itq=0,L-1
        do 560 il=1,L
        itk=il+itq
        xs(itq)=xs(itq)+s(itk,iq)*s(il,iq)
        xsa=xsa+s(il,iq)
        xs2=xs2+s(il,iq)**2
560     continue
559     continue
558     continue
        do 561 ii=1,L-1
        xs(ii)=xxx*xs(ii)/float(kxmax*L)
        write(43,*)float(ii)/dtau/float(L),xs(ii)
561     continue

        endif


777	continue
        return
        end



c***********dirty update of g**************

      subroutine gnew(g,v,j,del,xflag)
        implicit real*8(a-h,o-z)
      include 'parameter2.h'

c     parameter(L=64)
c     parameter(L=16)
c     parameter(L=128)
c     parameter(L=100)
c     parameter(L=80)
c     parameter(L=8)
      double precision g(L,L),v(L),del(L,L),d(L,L)

        dv=xflag*2.*v(j)
        ee=exp(dv)-1.
        a=ee/(1.+(1.-g(j,j))*ee)
        do 2 i2=1,L
           do 1 i1=1,L
             d(i1,i2)=g(i1,i2)+(g(i1,j)-del(i1,j))*(a*g(j,i2))
 1         continue
 2      continue
        do 3 i2=1,L
        do 3 i1=1,L
        g(i1,i2)=d(i1,i2)
3       continue
      return
      end

c##################################################


c******clean update of g************

      subroutine gnewclean(g,v,gx,del,xflag)
	implicit real*8(a-h,o-z)
      include 'parameter2.h'

c     parameter(L=64)
c     parameter(L=16)
c     parameter(L=128)
c     parameter(L=100)
c     parameter(L=80)
c     parameter(L=8)
      double precision g(L,L),v(L),b(L,L),binv(L,L)
      double precision del(L,L)
      double precision w(L,L)
c     double precision g0(-L:L),del(L,L)
 	dimension gx(L,L),ee(L)
c	dimension gx(L,L),ee(L),ipvt(L),z(L)
c	dimension det(2),work(L)

	do 3 i=1,L
	ee(i)=dexp(xflag*v(i))-1.
3	continue

      do 2 j=1,L
         do 1 i=1,L
            b(i,j)=del(i,j)-ee(j)*(gx(i,j)-del(i,j))
 1       continue
 2    continue
c for imsl
c     call dlinrg(l,b,L,binv,L)
c for num rec
 	call gaussj(b,L,L,w,L,L)
 	do j=1,L
 	do i=1,L
 	binv(i,j)=b(i,j)
 	enddo
 	enddo

      do 13 i1=1,L
         do 14 i2=1,L
            xdum=0.
            do 15 i=1,L
c              xdum=xdum+binv(i1,i)*g0(i-i2)
               xdum=xdum+binv(i1,i)*gx(i,i2)
 15          continue
        g(i1,i2)=xdum
 14       continue
 13    continue

c for cray
c	call sgeco(b,L,L,ipvt,rcond,z)
c	call sgedi(b,L,L,ipvt,det,work,01)
c	call sgemm('n','n',L,L,L,1.,b,L,gx,L,0.,g,L)
      return
      end

c####################################################

c****initialize the vector v of Ising fields*******

      subroutine initial(xlam,v,polar,L)
	implicit real*8(a-h,o-z)
      double precision v(L)
c     real rnunf
c     external rnset,rnunf

      idum=-123457
      r=ran1(idum)
      do 1 i=1,L
         s=1.
	  r=ran1(idum)
         if(r.gt.polar) s=-1.
         v(i)=xlam*s
 1    continue
      return
      end



c########################################################
c########################################################

      SUBROUTINE four1(data,nn,isign)
        implicit real*8(a-h,o-z)
        double precision data(2*nn)
c     INTEGER isign,nn
c     REAL data(2*nn)
c     INTEGER i,istep,j,m,mmax,n
c     REAL tempi,tempr
c     DOUBLE PRECISION theta,wi,wpi,wpr,wr,wtemp
      n=2*nn
      j=1
      do 11 i=1,n,2
        if(j.gt.i)then
          tempr=data(j)
          tempi=data(j+1)
          data(j)=data(i)
          data(j+1)=data(i+1)
          data(i)=tempr
          data(i+1)=tempi
        endif
        m=n/2
1       if ((m.ge.2).and.(j.gt.m)) then
          j=j-m
          m=m/2
        goto 1
        endif
        j=j+m
11    continue
      mmax=2
2     if (n.gt.mmax) then
        istep=2*mmax
        theta=6.28318530717959d0/(isign*mmax)
        wpr=-2.d0*dsin(0.5d0*theta)**2
        wpi=dsin(theta)
        wr=1.d0
        wi=0.d0
        do 13 m=1,mmax,2
          do 12 i=m,n,istep
            j=i+mmax
            tempr=sngl(wr)*data(j)-sngl(wi)*data(j+1)
            tempi=sngl(wr)*data(j+1)+sngl(wi)*data(j)
            data(j)=data(i)-tempr
            data(j+1)=data(i+1)-tempi
            data(i)=data(i)+tempr
            data(i+1)=data(i+1)+tempi
12        continue
          wtemp=wr
          wr=wr*wpr-wi*wpi+wr
          wi=wi*wpr+wtemp*wpi+wi
13      continue
        mmax=istep
      goto 2
      endif
      return
      END

c########################################################
c########################################################

      SUBROUTINE spline(x,y,n,yp1,ypn,y2)
      INTEGER n,NMAX
      REAL*8 yp1,ypn,x(n),y(n),y2(n)
      PARAMETER (NMAX=800)
      INTEGER i,k
      REAL*8 p,qn,sig,un,u(NMAX)
      if (yp1.gt..99e30) then
        y2(1)=0.
        u(1)=0.
      else
        y2(1)=-0.5
        u(1)=(3./(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.
        y2(i)=(sig-1.)/p
        u(i)=(6.*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
      if (ypn.gt..99e30) then
        qn=0.
        un=0.
      else
        qn=0.5
        un=(3./(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.)
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software 3^03.

c########################################################
c########################################################
      SUBROUTINE splint(xa,ya,y2a,n,x,y)
      INTEGER n
      REAL*8 x,y,xa(n),y2a(n),ya(n)
      INTEGER k,khi,klo
      REAL*8 a,b,h
      klo=1
      khi=n
1     if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.) pause 'bad xa input in splint'
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     *2)/6.
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software 3^03.


c########################################################
c########################################################
      FUNCTION ran1(idum)
      INTEGER idum,IA,IM,IQ,IR,NTAB,NDIV
      REAL*8 ran1,AM,EPS,RNMX
      PARAMETER (IA=16807,IM=2147483647,AM=1./IM,IQ=127773,IR=2836,
     *NTAB=32,NDIV=1+(IM-1)/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
      INTEGER j,k,iv(NTAB),iy
      SAVE iv,iy
      DATA iv /NTAB*0/, iy /0/
      if (idum.le.0.or.iy.eq.0) then
        idum=max(-idum,1)
        do 11 j=NTAB+8,1,-1
          k=idum/IQ
          idum=IA*(idum-k*IQ)-IR*k
          if (idum.lt.0) idum=idum+IM
          if (j.le.NTAB) iv(j)=idum
11      continue
        iy=iv(1)
      endif
      k=idum/IQ
      idum=IA*(idum-k*IQ)-IR*k
      if (idum.lt.0) idum=idum+IM
      j=1+iy/NDIV
      iy=iv(j)
      iv(j)=idum
      ran1=min(AM*iy,RNMX)
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software 3^03.
c########################################################
c########################################################
      SUBROUTINE gaussj(a,n,np,b,m,mp)
      INTEGER m,mp,n,np,NMAX
      REAL*8 a(np,np),b(np,mp)
      PARAMETER (NMAX=256)
      INTEGER i,icol,irow,j,k,l,ll,indxc(NMAX),indxr(NMAX),ipiv(NMAX)
      REAL*8 big,dum,pivinv
      do 11 j=1,n
        ipiv(j)=0
11    continue
      do 22 i=1,n
        big=0.
        do 13 j=1,n
          if(ipiv(j).ne.1)then
            do 12 k=1,n
              if (ipiv(k).eq.0) then
                if (abs(a(j,k)).ge.big)then
                  big=abs(a(j,k))
                  irow=j
                  icol=k
                endif
              else if (ipiv(k).gt.1) then
                pause 'singular matrix in gaussj'
              endif
12          continue
          endif
13      continue
        ipiv(icol)=ipiv(icol)+1
        if (irow.ne.icol) then
          do 14 l=1,n
            dum=a(irow,l)
            a(irow,l)=a(icol,l)
            a(icol,l)=dum
14        continue
          do 15 l=1,m
            dum=b(irow,l)
            b(irow,l)=b(icol,l)
            b(icol,l)=dum
15        continue
        endif
        indxr(i)=irow
        indxc(i)=icol
        if (a(icol,icol).eq.0.) pause 'singular matrix in gaussj'
        pivinv=1./a(icol,icol)
        a(icol,icol)=1.
        do 16 l=1,n
          a(icol,l)=a(icol,l)*pivinv
16      continue
        do 17 l=1,m
          b(icol,l)=b(icol,l)*pivinv
17      continue
        do 21 ll=1,n
          if(ll.ne.icol)then
            dum=a(ll,icol)
            a(ll,icol)=0.
            do 18 l=1,n
              a(ll,l)=a(ll,l)-a(icol,l)*dum
18          continue
            do 19 l=1,m
              b(ll,l)=b(ll,l)-b(icol,l)*dum
19          continue
          endif
21      continue
22    continue
      do 24 l=n,1,-1
        if(indxr(l).ne.indxc(l))then
          do 23 k=1,n
            dum=a(k,indxr(l))
            a(k,indxr(l))=a(k,indxc(l))
            a(k,indxc(l))=dum
23        continue
        endif
24    continue
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software 3^03.

c########################################################
c########################################################
