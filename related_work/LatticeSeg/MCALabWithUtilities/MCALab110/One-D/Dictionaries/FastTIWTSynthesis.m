function x = FastTIWTSynthesis(wcb, scale, qmf, scign)
% FastTIWTSynthesis -- Synthesis operator for Undecimated DWT (TI)
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
%	FastTIWTAnalysis, FWT_TI, IWT_TI
%

[n,J] = dyadlength(wcb(1).coeff);
if isstr(scale) scale=str2num(scale); end
if scign, wcb(2).coeff(:,1:scign)=0; end
x = IWT_TI(Stat2TI([wcb(1).coeff wcb(2).coeff]),qmf);
x = x(:);
