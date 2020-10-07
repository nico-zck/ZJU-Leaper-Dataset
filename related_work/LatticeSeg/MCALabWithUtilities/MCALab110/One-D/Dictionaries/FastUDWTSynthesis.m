function x = FastUDWTSynthesis(wcb, scale, qmf, scign)
% FastUDWTSynthesis -- Synthesis operator for Undecimated DWT (TI)
%  Usage:
%	x = FastUDWTSynthesis(wcb, n, scale, qmf, scign)
%  Inputs:
%	wcb	the coefs, a structure array
%	scale	the coarsest decomposition scale
%	qmf	the quadrature mirror filter
%	scign   the number of detail scales to be ignored in the reconstruction
%  Outputs:
%	x	the synthesized signal, a column vector
%  See Also:
%	FastUDWTAnalysis, mrdwt, mirdwt
%

[n,J] = dyadlength(wcb(1).coeff);
if isstr(scale) scale=str2num(scale); end
wcb(2).coeff(:,1:scign)=0;
x = mirdwt(wcb(1).coeff,wcb(2).coeff,qmf,J-scale);
x = x(:);
