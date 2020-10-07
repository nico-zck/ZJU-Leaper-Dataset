function wcb = FastUDWTAnalysis(x, scale, qmf, par3)
% FastUDWTAnalysis -- Analysis operator for Undecimated DWT (TI)
%  Usage:
%	wcb = FastUDWTAnalysis(x, scale, qmf)
%  Inputs:
%	x	the signal, a column vector
%	scale	the coarsest decomposition scale
%	qmf	the quadrature mirror filter
%  Outputs:
%	wc	the coefs, a structure array
%  See Also:
%	FastUDWTSynthesis, mrdwt, mirdwt
%

[n,J] = dyadlength(x);
if isstr(scale) scale=str2num(scale); end
[ll,wc,L] = mrdwt(x,qmf,J-scale);
wcb =  [struct('scale', scale, 'coeff', ll) ...
	struct('scale', scale, 'coeff', wc)];


