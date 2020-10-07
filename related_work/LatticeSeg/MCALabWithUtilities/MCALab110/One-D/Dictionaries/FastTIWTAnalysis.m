function wcb = FastTIWTAnalysis(x, scale, qmf, par3)
% FastTIWTAnalysis -- Analysis operator for Undecimated DWT (TI)
%  Usage:
%	wcb = FastUDWTAnalysis(x, scale, qmf)
%  Inputs:
%	x	the signal, a column vector
%	scale	the coarsest decomposition scale
%	qmf	the quadrature mirror filter
%  Outputs:
%	wc	the coefs, a structure array
%  See Also:
%	FastTIWTSynthesis, FWT_TI, IWT_TI
%

[n,J] = dyadlength(x);
if isstr(scale) scale=str2num(scale); end
tiwt    = TI2Stat(FWT_TI(x,scale,qmf));
ll	= tiwt(:,1);
wc	= tiwt(:,2:end);
wcb =  [struct('scale', scale, 'coeff', ll) ...
	struct('scale', scale, 'coeff', wc)];


