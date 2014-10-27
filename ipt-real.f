      program fixed mu0t
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c                      Code for the binary alloy
c                       version for fixed mu and mu0t
c
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     This program reads in data from the files parameter.in
C     and data.in.  The file data.in is supplied by the 
C     Matsubara code ba_clean-Z.f
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     This program was slightly modified for improved readability 
C     by Gunnar Palsson in July 1996
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     This program must be compiled with the files hkfourier.f and
C     hksolvimp.f
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT Double precision (A-h, o-Z)


      integer n
      real*8 pi
      parameter(n = 2**13)
      parameter(pi=3.141592653589793d0)  
    
      character ,outfile*9

      integer k,i,idata,index
      integer nloop,iloop,imin,imax,istep,inguess,ou,ndata

      real*8 u,mu,gss1,gss2,mu0t_a
      real*8 mu_a,  nocc_a, w
      real*8 nocc_ou_a,temp,t_in
      real*8 wout,rout,iout

      double complex delta(n),gafin(n), gaf_a(n),z(n)
      double complex sigaf_a(n),gaf_a1(n),gaf_a2(n) 
      double complex gaf_a3(n),gaf_a4(n), sig_av(n)
      
      real*8 domega,tomega,dtime,time(n),omega(n),t
      real*8 ttime,acc,eta,dos1(n),dos2(n),dos3(n),dos4(n)
      
      common/param/u,t,temp,domega,omega,dtime,time,tomega,ttime,z,eta

      common/outfuncs/gafin,gaf_a,sigaf_a,delta,sig_av
      
      open(6000, file='data.in',status='unknown')
      open(6001, file='parameter.in',status='unknown')

      open(6210, file='n_ab.ou',status='unknown')
      open(6220, file='doc.ou',status='unknown')

C     ------ Read parameters from command line ------

c      call getarg(1,p)
c      decode(20,'(i8)',p) ndata

c     ------ Read parameters from parameter.in ------

      read(6001,*) ou, ndata
      read(6001,*) nloop,t_in
      read(6001,*)  eta, tomega,inguess
      read(6001,*)  imin,imax,istep
      read(6001,*) acc

      t=t_in

      print*, ou, ndata
      print*, nloop,t_in
      print*, 'eta, tomega,inguess (1:m,2:ins)'
      print*, real(eta), real(tomega),inguess
      print*, 'imin,imax,istep'
      print*, imin,imax,istep
      imax=n
      if ((imin.lt.0).or.(imax.gt.n)) then
        print*, 'error: imin, imax out of range'
        stop
      endif
      print*, real(acc)

c     ------- Initialize parameters ---------
      
      call init

      print*,'domega=',domega
      print*,'dtime=',dtime
      print*,'domega*dtime=',domega*dtime    

C     ------ Here starts the ndata loop ------
C     ----- Loops over entries in data.in ----

      do idata=1,ndata
         
C     ------- Read data from data.in -------
         
         read(6000,*) mu
         read(6000,*) u,temp
         read(6000,*) mu0t_a
         read(6000,*) nocc_a
C
         print*,'u,temp'
         print*, real(mu), real(w)
         print*, real(u), real(temp)
         print*,'mu0t_a'
         print*, real(mu0t_a)
         print*,'nocc_a'
         print*, real(nocc_a)

         
C     ------ Initialize bath greens functions -----
         
         call init_GF1(gafin,0.d0,inguess)
         
         do k=1,n
            gaf_a1(k)=gafin(k)
            gaf_a2(k)=gafin(k)
            gaf_a3(k)=gafin(k)
            gaf_a4(k)=gafin(k)
            
            gaf_a(k)=gafin(k)
            
            delta(k)=t**2*gafin(k)
         enddo
         
         
         
c     ----- Here starts the iteration loop ----
         
         iloop=1
                  
c         mu_a=mu+w/2.d0
         mu_a=mu
                  
 9000    print*,'loop in progress:', iloop
                  
         if (dabs(u).gt.1d-8) then
         
            call solve_imp_problem(mu0t_a,mu_a,nocc_a,delta,gaf_a,
     *                             sigaf_a,nocc_ou_a,inguess)

         else
            
            do k=1,n
               gaf_a(k)=1.d0/(z(k)+mu_a-delta(k))
               sigaf_a(k)=0.d0
            enddo
            
         endif
                  
         do k=1,n
            gaf_a1(k)=gaf_a2(k)
            gaf_a2(k)=gaf_a3(k)
            gaf_a3(k)=gaf_a4(k)
            gaf_a4(k)=0.25d0*(gaf_a1(k)+gaf_a2(k)+gaf_a3(k)+gaf_a(k))
            gaf_a(k)=gaf_a4(k)
            
         enddo
         
         do k=1,n
            delta(k)=t**2*(gaf_a(k))
         enddo
         
         do k=1,n
            dos1(k)=dos2(k)
            dos2(k)=dos3(k)
            dos3(k)=dos4(k)
            dos4(k)=1.d0/pi*4.d0*dimag(delta(k))
         enddo
         
         gss1=0.d0
         gss2=0.d0
         do i=1,n,n/64
            gss1=gss1+(dos4(i)-dos1(i))**2
         enddo    
         do i=(n/2-200), (n/2+200),10
            gss2=gss2+(dos4(i)-dos1(i))**2
         enddo
         
         gss1=gss1/64.d0
         gss2=gss2/40.d0
        
         print*, 'overall accuracy:',dsqrt(gss1)
         print*, 'accuracy around zero:',dsqrt(gss2)
         
         if (((gss1.lt.acc**2).and.(gss2.lt.acc**2)).
     *        and.(iloop.gt.6)) then 

            print*,'Convergence after', iloop, ' iterations'

         else

            if (iloop.eq.nloop) then 
               print*, 'no convergence'
            else
               iloop=iloop+1
               goto 9000
            endif

         endif
C     ------ Here ends the iteration loop -------
         
         do k=1,n
            sig_av(k)=z(k)-delta(k)-t**2/delta(k)
         enddo

C     ----- Write results to files -------
         
         index = idata+100
         write(outfile,'(''sigma.'',i3)') index
c         open(index,file=outfile,status='new')
         open(index,file=outfile,status='unknown')
         write(index,'(4e18.10)') temp,mu,nocc_ou_a,domega
         do k = imin,imax
            wout = omega(k)
            rout = real(sigaf_a(k))
            iout = real(dimag(sigaf_a(k)))
            write(index,'(3e18.10)') wout,rout,iout
         enddo
         close(index)
         write(outfile,'(''green.'',i3)') index
         open(index,file=outfile,status='unknown')
c         open(index,file=outfile,status='new')
         write(index,'(4e18.10)') temp,mu,nocc_ou_a,domega
         do k = imin,imax
            wout = omega(k)
            rout = real(gaf_a(k))
            iout = real(dimag(gaf_a(k)))
            write(index,'(3e18.10)') wout,rout,iout
         enddo
         close(index)
         
C     -----

         write(6210,*) 'nocc_a=', nocc_a, 'nocc_ou_a=',nocc_ou_a
         write(6210,*) 'nocc=',(nocc_a),
     *        'nocc_ou=',(nocc_ou_a)
c         write(6210,*) 'nocc=',(pa*nocc_a+(1.d0-pa)*nocc_b),
c     *        'nocc_ou=',(pa*nocc_ou_a+(1.d0-pa)*nocc_ou_b)
         
C     -----

         write(6220,*) 'u=', real(u)
         write(6220,*) 'w=', real(w)
         write(6220,*) 'mu=',real(mu)
         write(6220,*) 'temp=',real(temp)
         write(6220,*) 
         write(6220,*) 'nocc_a=', real(nocc_a)
         write(6220,*) 'nocc=', real(nocc_a)
         write(6220,*)
         write(6220,*)  '*******************************'
         write(6220,*)

C     -----

         if (ou.eq.1) then
            call printfuncs(omega,imin,imax,istep)
         endif
         
      enddo
C     ------ Here ends the ndata loop ------
      
      end

C     ===================================================================      

      subroutine printfuncs(omega,imin,imax,istep)
      IMPLICIT Double precision (A-h, o-Z)


      integer n
      parameter(n = 2**13)

      integer k,imin,imax,istep

      real*8 wout,rout1,iout1,rout2,iout2

      real*8 omega(n)

      double complex gafin(n),gaf_a(n)
      double complex sigaf_a(n),delta(n),sig_av(n)

      common/outfuncs/gafin,gaf_a,sigaf_a,delta,sig_av

      open(6010, file='gafin.ou',status='unknown')
      open(6020, file='gaf_a.ou',status='unknown')
      open(6040, file='gaf_av.ou',status='unknown')


c      do k=imin,imax,istep
      do k=imin,n,istep
         
         wout = real(omega(k)) 
         
         rout1 = real(dble(gafin(k)))
         iout1 = real(dimag(gafin(k)))
         write(6010,'(3e18.10)') wout,rout1,iout1
         
         rout1 = real(dble(gaf_a(k)))
         iout1 = real(dimag(gaf_a(k)))
         rout2 = real(dble(sigaf_a(k)))
         iout2 = real(dimag(sigaf_a(k)))               
c        write(6020,'(5e18.10)') wout,rout1,iout1,rout2,iout2
         write(6020,'(5e18.10)') wout,iout1,rout1,rout2,iout2
         
         
         rout1 = real(dble(4.d0*delta(k)))
         iout1 = real(dimag(4.d0*delta(k)))
         rout2 = real(dble(sig_av(k)))
         iout2 = real(dimag(sig_av(k)))
         write(6040,'(5e18.10)') wout,rout1,iout1,rout2,iout2
         
      enddo
      
      return
      end

C     ===================================================================      
