function x = FastDIRAC2Synthesis(c, par1, par2, par3)
% FastDIRACSynthesis -- Fast Dirac Synthesis Operator
%  Usage:
%	x = FastDIRACSynthesis(c)
%  Inputs:
%	c	the coef
%  Outputs:
%	x	the signal: x = c
%  See Also:
%	FastDIRACAnalysis
%

x = c(2).coeff(:);

