% This code solve the image decomposition model 
%    min tau TV(u)+ 0.5|u+div g-x0| + mu |g|_p
% by ADMM algorithm
% The implementation details can be referred to
% "M.K. Ng, X.M. Yuan, and W.X. Zhang, A coupled variational image
% decomposition and restoration model for blurred cartoon-plus-texture
% images with missing pixels,IEEE Trans. Image Process., vol. 22, 
% pp. 2233�C2246,  2013
% INPUT : 
% - x0 : 2D target image to be decomposed
% - p : The l_p norm to be used p can be {1,2,\infty}, p=\infty is better.
% - opts: 
%    opts.tau, opts.mu : parameters for the model
%    opts.beta1, opts.beta2 : parameters for the ADMM algorithm
%
% OUTPUT : 
% - u: cartoon part of image 
% - txtr: texture part of image
% written by Wenxing Zhang, wenxing84@gmail.com, July 2011


function [u,txtr] = ADM(x0,p,opts)

tau   = opts.tau;       mu    = opts.mu; 
beta1 = opts.beta1;     beta2 = opts.beta2;
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
ZERO = zeros(n1,n2,n3); 
u    = ZERO;  x1  = ZERO; x2   = ZERO; g1   = ZERO; g2 = ZERO;  
lbd11= ZERO; lbd12= ZERO; lbd21= ZERO; lbd22= ZERO; 
MDu  = beta1*(abs(d1h).^2+abs(d2h).^2) + beta2;
Mg11 = abs(d1h).^2 + beta2;
Mg22 = abs(d2h).^2 + beta2;
Mg12 = d1h.*conj(d2h);
Mg21 = d2h.*conj(d1h);
Mdet = beta2*(abs(d1h).^2+abs(d2h).^2) + beta2*beta2; 


for k = 1:MaxIt
    
    %%%% \tilde u
    tep1 = beta1*x1 + lbd11 + g1;
    tep2 = beta1*x2 + lbd12 + g2;
    Temp = PTx(tep1) + PTy(tep2) + x0;
    un   = real(ifft2(fft2(Temp)./MDu));
  
    %%%% \tilde z
    tep1= g1 + lbd21/beta2;
    tep2= g2 + lbd22/beta2;
    [poz1,poz2] = Project(tep1,tep2,mu/beta2,p);
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
    Temp = un-x0;
    b1 = Px(Temp) + beta2*zn1 - lbd21;
    b2 = Py(Temp) + beta2*zn2 - lbd22;
    nume1 = Mg22.*fft2(b1) - Mg12.*fft2(b2);
    nume2 = Mg11.*fft2(b2) - Mg21.*fft2(b1);    
    gn1 = real( ifft2(nume1./Mdet) );
    gn2 = real( ifft2(nume2./Mdet) );
   
    %%%% update lagrange multipliers
    lbdn11= lbd11 - beta1*(dxun-xn1);
    lbdn12= lbd12 - beta1*(dyun-xn2);
    lbdn21= lbd21 - beta2*(zn1-gn1);
    lbdn22= lbd22 - beta2*(zn2-gn2);
    
    u = un; g1 = gn1; g2=gn2; x1=xn1; x2=xn2;
    lbd11 = lbdn11; lbd12 = lbdn12; lbd21 = lbdn21; lbd22 = lbdn22;

end

txtr= -PTx(g1)-PTy(g2);

% fprintf('the running is over!\n')
