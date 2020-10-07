function dwt = FastATROUAnalysis(x,L,par2,par3)
% 
% FastATROUAnalysis -- Fast Dyadic Wavelet Transform (periodized, orthogonal)
%  Usage
%    dwt = FastATROUAnalysis(x,L)
%  Inputs
%    x    	1-d signal; length(x) = 2^J = n
%    L    	Coarsest Level of V_0;  L << J
%  Outputs
%    dwt   a structure array 
%          dwt(2).coeff: an n times J-L matrix giving the wavelet transform of x at all dyadic scales.
%          dwt(1).coeff: an n column vector giving the approximation
%          coefficients
%
%  Description
%    To reconstruct use FastATROUSynthesis 
%
%  See Also
%    FWT_ATrou, IWT_ATrou, MakeATrouFilter
%

[n,J] = dyadlength(x);
if isstr(L) L=str2num(L); end
wc = FWT_ATrou(x,L);
dwt = [struct('scale',L,'coeff',wc(:,1)) ...
       struct('scale',L,'coeff',wc(:,2:J-L+1))];
    
