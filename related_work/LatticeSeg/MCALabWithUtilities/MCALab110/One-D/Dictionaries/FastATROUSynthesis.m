function s = FastATROUSynthesis(dwt,L,par2,par3);
%
% FastATROUSynthesis -- Fast Inverse Dyadic Wavelet Transform
%  Usage
%    s = FastATROUSynthesis(dwt,L)
%  Inputs
%    dwt  strcture array
%    L    Coarsest Level of V_0;  L << J
%  Outputs
%    s	original 1-d signal; length(x) = 2^J = n
%  Description
%    1. filters are obtained with MakeATrouFilter
%    2. usually, length(qmf) < 2^(L+1)
%    3. The transformed signal can be obtained by FastATROUAnalysis
%  See Also
%    FWT_ATrou, IWT_ATrou, MakeATrouFilter
%
[n,J] = dyadlength(dwt(1).coeff);
if isstr(L) L=str2num(L); end
wc = [dwt(1).coeff,dwt(2).coeff];
s = IWT_ATrou(wc,L);
s = s(:);
	
