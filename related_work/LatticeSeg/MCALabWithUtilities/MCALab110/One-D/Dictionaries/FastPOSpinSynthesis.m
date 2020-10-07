function x = FastPOSpinSynthesis(wcb, scale, qmf, shift, scign)
% FastPOSynthesis -- Fast Synthesis Operator for Periodized-Orthognal with cycle spining
%			Wavelets Dictionary
%  Usage:
%	x = FastPOSSpinynthesis(c, L, qmf)
%  Inputs
%    c    1-d wavelet transform of x.
%    L    Coarsest Level of V_0;  L << J
%    qmf  quadrature mirror filter (orthonormal)
%  Outputs
%    x    1-d signal; length(x) = 2^J
%
%  Description
%    1. qmf filter may be obtained from MakeONFilter   
%    2. usually, length(qmf) < 2^(L+1)
% 
%  See Also
%    FastPOAnalysis, FWT_PO, IWT_PO, MakeONFilter
%

c = [wcb(1).coeff;wcb(2).coeff];
[n,J] = dyadlength(c(:,1));
if exist('scign','var') & scign, wcb(2).coeff(2^(J-scign):end,:)=0; end
c = [wcb(1).coeff;wcb(2).coeff];
if isstr(scale) scale=str2num(scale); end
x = IWT_PO(c(:,1), scale, qmf);
for i=1:shift
  x = x + circshift(IWT_PO(c(:,2*i), scale, qmf),-i);
  x = x + circshift(IWT_PO(c(:,2*i+1), scale, qmf),i);
end

x = x(:);
