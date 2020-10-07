%%% min || |u| ||_1+ tau ||(u+div g)-f||^2_2 + mu|| |g| ||_p
%%%    This model is for image decomposition 

function [u,txtr] = Dnoisetau(x0,p,opts)

tau = opts.tau; mu = opts.mu; 
beta1 = opts.beta1;  beta2 = opts.beta2; beta3 = opts.beta3;
MaxIt = opts.MaxIt;
[n1,n2,n3] = size(x0);
%%%%%%%%%%%%%%%%% Periodic  boundary condtion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1h  = zeros(n1,n2,n3); d1h(1,1,:) = -1; d1h(n1,1,:) = 1; d1h = fft2(d1h);
d2h  = zeros(n1,n2,n3); d2h(1,1,:) = -1; d2h(1,n2,:) = 1; d2h = fft2(d2h);
Px = @(x) [x(2:n1,:,:)-x(1:n1-1,:,:); x(1,:,:)-x(n1,:,:)]; %%\nabla_1 x 
Py = @(x) [x(:,2:n2,:)-x(:,1:n2-1,:), x(:,1,:)-x(:,n2,:)]; %%\nabla_2 y
PTx= @(x) [x(n1,:,:)-x(1,:,:); x(1:n1-1,:,:)-x(2:n1,:,:)]; %%\nabla_1^T x 
PTy= @(x) [x(:,n2,:)-x(:,1,:), x(:,1:n2-1,:)-x(:,2:n2,:)]; %%\nabla_2^T y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% initinalization
ZERO= zeros(n1,n2,n3); 
u = ZERO;  x1 = ZERO; x2 = ZERO; y = ZERO; g1 = ZERO; g2 = ZERO; txtr =ZERO; 
lbd11 = ZERO; lbd12 = ZERO; lbd2 = ZERO; 
lbd31 = ZERO; lbd32 = ZERO;  
MDu = beta1*(abs(d1h).^2+abs(d2h).^2) + beta2;
MDy = beta2 + 1;
Mg11= beta2*abs(d1h).^2 + beta3;
Mg22= beta2*abs(d2h).^2 + beta3;
Mg12= beta2*d1h.*conj(d2h);
Mg21= beta2*d2h.*conj(d1h);
Mdet= beta2*beta3*(abs(d1h).^2 + abs(d2h).^2) + beta3*beta3; 
HTx0= x0;

for k = 1:MaxIt
    
    
    %%%% \tilde u
    tep1 = beta1*x1 + lbd11 + beta2*g1;
    tep2 = beta1*x2 + lbd12 + beta2*g2;
    Temp = PTx(tep1) + PTy(tep2) + lbd2 + beta2*y;
    un   = real(ifft2(fft2(Temp)./MDu));
  
    %%%% \tilde z
    tep1= g1 + lbd31/beta3;
    tep2= g2 + lbd32/beta3;
    [poz1,poz2] = Project(tep1,tep2,mu/beta3,p);
    zn1 = tep1 - poz1;
    zn2 = tep2 - poz2;
 
    %%%% \tilde x
    dxun = Px(un);
    dyun = Py(un);
    sk1 = dxun - lbd11/beta1;
    sk2 = dyun - lbd12/beta1;
    nsk = sqrt(sk1.^2 + sk2.^2); nsk(nsk==0)=1;
    nsk = max(1 - tau./(beta1*nsk),0);
    xn1 = nsk.*sk1;
    xn2 = nsk.*sk2;
    
    %%%% \tilde g
    Temp = beta2*(un-y) - lbd2;
    b1 = Px(Temp) + beta3*zn1 - lbd31;
    b2 = Py(Temp) + beta3*zn2 - lbd32;
    nume1 = Mg22.*fft2(b1) - Mg12.*fft2(b2);
    nume2 = Mg11.*fft2(b2) - Mg21.*fft2(b1);    
    gn1 = real( ifft2(nume1./Mdet) );
    gn2 = real( ifft2(nume2./Mdet) );
    
    %%%% \tilde y
    txtr= -PTx(gn1)-PTy(gn2);
    Temp= HTx0 + beta2*(un+txtr) - lbd2;
    yn  = Temp/MDy;
    
    %%%% update lagrange multipliers
    lbdn11= lbd11 - beta1*(dxun-xn1);
    lbdn12= lbd12 - beta1*(dyun-xn2);
    lbdn2 = lbd2  - beta2*(un+txtr-yn);
    lbdn31= lbd31 - beta3*(zn1-gn1);
    lbdn32= lbd32 - beta3*(zn2-gn2);
    
    
    u = un; g1 = gn1; g2=gn2; x1=xn1; x2=xn2; y=yn; 
    lbd11 = lbdn11; lbd12 = lbdn12; lbd2 = lbdn2; lbd31 = lbdn31; 
    lbd32 = lbdn32;
    
    
end




