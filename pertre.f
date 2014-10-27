c tienen el glitch del output de dens(i) arreglado
c this is also known as testotro.f
c added the calculation of the density density bubble 2/97
c added the calculation of the conductivity 1/94
cthis is a good program for perturbation
c may 25
c  this one is almost running good
c  still has the problems of the discontinuities
c  in the g0 and sigma
c	parameter(L=32768)
 	parameter(L=1*8192)
	implicit real*8(a-h,o-z)
	double complex one,xi,xs,sq,g,ome
c	double complex zz
c	double complex zerfe
	double complex g0(2*L),tg0(2*L),fg0(-L:L)
	double complex sft2(2*L),sf2(2*L),sft(-L:L),sf(-L:L)
	double precision dns(2*L),xsq(2*L),ysq(2*L)
	double precision zsq(2*L),wsq(2*L),sg(2*L)
	double precision cond(10000),c(-10000:10000)
        COMMON /WORKSP/  RWKSP
C        REAL RWKSP(1572914)
C        CALL IWKIN(1572914)
c	external zerfe


c	read(22,*)dtau,d0,u,nloop
	read(22,*)f,d0,u,nloop
	read(22,*)i01,imax1,i02,imax2,nl0,imet
	read(22,*)kmin,kmax,kstep,del,ff,igauss
	read(22,*)icond,ine,ibub
	pi=3.1415927
c	f=2.*pi/dtau/dfloat(2*L)
	dtau=2.*pi/f/dfloat(2*L)
	f2=f**2
	one=(1.,0.)
	xi=(0.,1.)
	XL=float(L)
	xn=1./(u**2/pi/d0)
	epsilon=1./100000.
	xinrootpi=1./dsqrt(pi/2.)
	xinpihaf=2./pi
	L2=2*L

c	do 1 i=1,2*L
c	om=(float(i)-XL-1.)*f
c 	del(i)=d0
c	if(om.lt.1.d-6.and.om.ge.0.)om=1.d-6
c	if(om.gt.-1.d-6.and.om.le.0.)om=-1.d-6
c	sf(i-L-1)=-one/om
c1	continue

	do 100 iloop=1,nloop

	do 2 i=L+1,2*L
	om=(float(i)-XL-1.)*f
	sig=1.d0
	ome=om*one
	xs=sf(i-L-1)
	sq=cdsqrt((ome-xs)**2-d0**2*one)
	sqim=dimag(sq)
	sqre=real(sq)


	if(i.le.L+5)then
		if(sqre.gt.0..and.imet.eq.0)sig=-1.d0
	 else
c***********
c		ximp=dimag(ome+xs+sq)
c		ximm=dimag(ome+xs-sq)
c		if(ximp.lt.0.)sig=-1.d0
c		if(ximp*ximm.gt.0.1)then
c***********
		   benpr=real(2./(ome+xs+sq))
	 	   benmr=real(2./(ome+xs-sq))
	 	   benpi=dimag(2./(ome+xs+sq))
		   benmi=dimag(2./(ome+xs-sq))
		   xp=((benpr+benchr0-2.*benchr)**2
     $    	   +(benpi+benchi0-2.*benchi)**2)
		   xm=((benmr+benchr0-2.*benchr)**2
     $    	   +(benmi+benchi0-2.*benchi)**2)
		   if(xp.gt.xm)sig=-1.d0
c	        endif
        g=(2./(ome+xs+sq))
        d1=dimag(one/g)-dimag(sf(i-L-1))
        d2=real(one/g)-real(sf(i-L-1))
        d2=d2**2
        d3=d1**2
        dn=d1/(d2+d3)
        if(dn.lt.0)sig=-1.

	end if
	xsq(i)=sig*sqim
	ysq(i)=sig*sqre
	zsq(i)=ome+real(xs)
	wsq(i)=dimag(xs)
	g0(i)=2./(ome+xs+sig*sq)
	sg(i)=sig

	if(iloop.eq.1)then
	 if(dabs(om).lt.1.d-9)then
	  g0(i)=0.*one
	else
	  g0(i)=1./(ome+float(imet)*d0*xi/2.)
	endif
	endif

	benchr=real(g0(i))
	benchi=dimag(g0(i))
	benchr0=real(g0(i-1))
	benchi0=dimag(g0(i-1))
2	continue

	do 222 i=1,L-1
	g0(L+1-i)=-g0(L+1+i)
222	continue
	g0(1)=g0(2)

	g0(L+1)=0.*one

c*** g0 has the freq spec from -pi to pi

c-------------------------------------
c IMSL
c       call dfftcb(2*L,g0,tg0)
c
c num rec
        do i=1,2*L
        tg0(i)=g0(i)
        enddo
        call four1(tg0,L2,1)
c-------------------------------------


	ex=-1.
        do 3 i=1,2*L
	ex=-ex
           tg0(i)=ex*f/2./pi*tg0(i)
 3      continue

        do 4 i=1,L
           fg0(i-1)=tg0(i)
 4      continue

        do 5 i=L+1,2*L
           fg0(-2*L-1+i)=tg0(i)
 5      continue

c *** fg0(i) has the real time GF
c  from -beta to beta



	do 7 i=-L+1,L-1
	sft(i)=-u**2*fg0(i)**2*fg0(-i)
7	continue
	sft(-L)=-u**2*fg0(-L)**2*fg0(L-1)

        do 24 i=-L,L-1
	   sft2(i+L+1)=sft(i)
 24      continue


c-------------------------------------
c IMSL
c       call dfftcf(2*L,sft2,sf2)
c
c num rec
        do i=1,2*L
        sf2(i)=sft2(i)
        enddo
        call four1(sf2,L2,-1)
c-------------------------------------


	ex=-1.
	do 8 i=1,2*L
	ex=-ex
	sf2(i)=-ex*dtau*sf2(i)
8	continue

        do 34 i=1,L
           sf(i-1)=sf2(i)
34      continue

        do 35 i=L+1,2*L
           sf(-2*L-1+i)=sf2(i)
35      continue


c*** sf is the self-energy from -pi to pi
	if(iloop.ge.nl0)then
c	if(mod(iloop,2).eq.0)then
	do 111 i=1,2*L
	if(i.eq.L+1)then
	dns(i)=2.*float(imet)/d0
	else
	d1=dimag(one/g0(i))-dimag(sf(i-L-1))
	d2=real(one/g0(i))-real(sf(i-L-1))
	d2=d2**2
	d3=d1**2
	dns(i)=d1/(d2+d3)
	endif
111	continue
	if(imet.eq.0)dns(L+1)=0.
	do  i=imax2,imax1+1,-i02
        write(25,*)-real(f*float(i)),real(dns(i+L+1))
 	write(24,*)-real(f*float(i)),real(dimag(sf(-i)))
        write(23,*)-real(f*float(i)),real(real(sf(-i)))
c	write(26,*)-real(f*float(i)),real(-real(sf(-i))-f*float(i))
	enddo
        do i=imax1,0,-i01
        write(25,*)-real(f*float(i)),real(dns(i+L+1))
 	write(24,*)-real(f*float(i)),real(dimag(sf(-i)))
        write(23,*)-real(f*float(i)),real(real(sf(-i)))
c	write(26,*)-real(f*float(i)),real(-real(sf(-i))-f*float(i))
	enddo

	do 118 i=1,imax1,i01
 	write(24,*)real(f*float(i)),real(dimag(sf(i)))
 	write(23,*)real(f*float(i)),real(real(sf(i)))
c	write(26,*)real(f*float(i)),real(-real(sf(i))+f*float(i))
	write(25,*)real(f*float(i)),real(dns(i+L+1))
c	write(11,*)real(f*float(i)),real(real(g0(i+L+1)))
c	write(10,*)real(f*float(i)),real(sg(i+L+1))
c	write(12,*)real(f*float(i)),real(dimag(g0(i+L+1)))
c	write(31,*)real(f*float(i)),real(ysq(i+L+1))
cc	write(32,*)real(f*float(i)),real(xsq(i+L+1))
c	write(33,*)real(f*float(i)),real(zsq(i+L+1))
c	write(34,*)real(f*float(i)),real(wsq(i+L+1))
	
118	continue
	do 116 i=imax1+1,imax2,i02
 	write(24,*)real(f*float(i)),real(dimag(sf(i)))
 	write(23,*)real(f*float(i)),real(real(sf(i)))
c	write(26,*)real(f*float(i)),real(-real(sf(i))+f*float(i))
	write(25,*)real(f*float(i)),real(dns(i+L+1))
c	write(11,*)real(f*float(i)),real(real(g0(i+L+1)))
c	write(10,*)real(f*float(i)),real(sg(i+L+1))
c	write(12,*)real(f*float(i)),real(dimag(g0(i+L+1)))
c	write(31,*)real(f*float(i)),real(ysq(i+L+1))
c	write(32,*)real(f*float(i)),real(xsq(i+L+1))
c	write(33,*)real(f*float(i)),real(zsq(i+L+1))
c	write(34,*)real(f*float(i)),real(wsq(i+L+1))
	
116	continue
 	write(24,*)'    '
 	write(23,*)'   '
	write(25,*)'    '
c	write(11,*)' '
c	write(12,*)' '
	endif

100	continue	

c	do 234 i=L+1,2*L
c	xnp=xnp+f*dns(i)
c	en=en+f*f*float(i-L-1)*dns(i)
c234	continue

c** en is the total energy and xnp is the occupation number

c	write(1,*)real(u),en
c	write(2,*)real(u),xnp


	if(icond.eq.1)then
c****do the convolution to get the conductivity*********
	cc=0.d0
	xx=f*ff/2.d0/pi
	do 199 k=kmin,kmax,kstep
	print*,k
	xnu=dfloat(k)*f
	cond(k)=0.d0
	do 200 i=-1000,1000
	c(i)=0.d0
	ep=dfloat(i)*ff

	e2=ep**2
	rho=0.d0
	if(e2.ge.1.d0)goto 200
	if(igauss.eq.1)then
 	rho=dexp(-2.*e2)*xinrootpi
	else
	if(e2.ge.1.d0)goto 200
	rho=dsqrt(1.d0-e2)*xinpihaf
	endif
	
	do 201 j=-k,-1
	om=dfloat(j)*f

	jk=abs(j+k)
c***take abs to avoid running into bus errors
c***see if statement below
c***still the meaningful quantity is just j+k
	sr1=real(sf(j))
	si1=dabs(dimag(sf(j)))
	sr2=0.d0
	si2=0.d0
	if(jk.ge.L)goto 198
	sr2=real(sf(j+k))
	si2=dabs(dimag(sf(j+k)))
198	continue
	
	

	ar=om-ep-sr1
	ai=del+si1
c	if(si1.gt.del)ai=si1
	
	
	br=om+xnu-ep-sr2
	bi=del+si2
c       if(si2.gt.del)bi=si2

	a=ai/(ar**2+ai**2)
	b=bi/(br**2+bi**2)

	c(i)=c(i)+a*b/xnu
201	continue
	
	cond(k)=cond(k)+rho*c(i)

200	continue
	write(3,*)real(xnu),real(cond(k)*xx)
199	continue

	endif

c***compute n(e)*****
	if(ine.eq.1)then
	do i=-1000,1000,10
	ep=dfloat(i)*ff
	xne=0.d0

	do j=-10000,-1
	om=dfloat(j)*f

        sr=real(sf(j))
        si=dabs(dimag(sf(j)))
        ar=om-ep-sr
        ai=del
        if(si.gt.del)ai=si
        a=ai/(ar**2+ai**2)
	xne=xne+a*f
	enddo
	write(100,*)real(ep),real(xne/pi)
	enddo
	
	endif



	if(ibub.eq.1)then
c****do the convolution to get the density-density bubble*********
	
	do 556 ixq=0,3
	xq=dfloat(ixq)*.25
	gx=xq
	fx=(1.+xq**2)/(1.-xq**2)
	hx=1./dsqrt((1.-xq**2)/2.)
	
	xx=f*ff/2.d0/pi
	do 599 k=kmin,kmax,kstep
	print*,k
	xnu=dfloat(k)*f
	cond(k)=0.d0
	do 500 i=-1000,1000
	c(i)=0.d0
	ep=dfloat(i)*ff

	e2=ep**2*fx
	rho=0.d0
	if(e2.ge.1.d0)goto 500
 	rho=dexp(-2.*e2)*xinrootpi
c the 2 comes from the fact that for D=2t

	do 501 j=-k,-1
	om=dfloat(j)*f

	jk=abs(j+k)
c***take abs to avoid running into bus errors
c***see if statement below
c***still the meaningful quantity is just j+k
	sr1=real(sf(j))
	si1=dabs(dimag(sf(j)))
	sr2=0.d0
	si2=0.d0
	if(jk.ge.L)goto 598
	sr2=real(sf(j+k))
	si2=dabs(dimag(sf(j+k)))
598	continue
	
	

	ar=om-ep-sr1
	ai=del+si1
c	if(si1.gt.del)ai=si1
	
	
c###here goes one modification
	br=om+xnu-ep*gx-sr2
	bi=del+si2
c       if(si2.gt.del)bi=si2

	a=ai/(ar**2+ai**2)
cc	c###here goes another one
cc	c	b=bi/(br**2+bi**2)
cc	 	zz=hx*(br*one+bi*xi)
cc	 	b=zerfe(zz)
cc	 	b=dimag(xi*b)

	c(i)=c(i)+a*b
501	continue
	
	
	cond(k)=cond(k)+rho*c(i)/dsqrt(1.-xq**2)
500	continue
	write(4,*)real(xnu),real(cond(k)*xx)
599	continue
	write(4,*)'    '
556	continue

	endif




	stop
	end

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


