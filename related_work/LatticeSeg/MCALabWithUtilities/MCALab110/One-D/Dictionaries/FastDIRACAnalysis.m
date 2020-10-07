function c = FastDIRAC2Analysis(x, par1, par2, par3)
% FastDIRACAnalysis -- Fast Dirac Analysis Operator
%  Usage:
%	c = FastDIRACAnalysis(x)
%  Inputs:
%	x	the signal
%  Outputs:
%	c	the coef: c = x
%  See Also:
%	FastDIRACSynthesis
%

c = [struct('coeff', []) struct('coeff', x(:))];

