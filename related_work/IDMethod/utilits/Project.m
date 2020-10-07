%%%% Projection onto the l^p-ball of size alpha
function [g1,g2]=Project(g1,g2,alpha,p)
ng = sqrt(g1.^2 + g2.^2);

if p==2
    ng = norm(ng(:));
    if ng>alpha
        g1 = (alpha/ng)*g1;
        g2 = (alpha/ng)*g2;
    end
    
elseif p==Inf
    ng(ng==0)=1;
    ngc = min(1,alpha./ng); 
    g1 = ngc.*g1;
    g2 = ngc.*g2;
    
elseif p==1
    ngg = ProjectionWL1(ng,alpha);
    ng(ng==0)=1;
    ngc= ngg./ng;
    g1 = g1.*ngc;
    g2 = g2.*ngc;
    
    
end

  
function [z,sigma]=ProjectionWL1(xx,alpha)
x = xx(:);
ax= abs(x(:));
n = numel(x);
M = sum(ax);
if M<=alpha;  z=xx; sigma=0;  return; end
if alpha<=0
    z=zeros(size(xx));  sigma=infty;   return;
end
y=sort(ax); E1=M; E=E1-n*y(1); i=1;
while E>alpha && i<n
    i = i+1;
    E1= E1-y(i-1);
    Ep= E;
    E = E1-(n-i+1)*y(i);
end
if (i>1)
  a=(y(i)-y(i-1))*alpha;
  b=E*y(i-1)-abs(Ep)*y(i);
  sigma=(a+b)/(E-abs(Ep));
else
  sigma=y(1)*(M-alpha)/(M-E);    
end
z=x(:); K = find (ax<sigma, 1);
if (~isempty(K));  z(ax<sigma)=0; end
K=find(ax>=sigma);
if (~isempty(K));  z(K)=x(K)-sigma*sign(x(K)); end
z=reshape(z,size(xx));

