function x = FastDSTiSynthesis(c, n, fineness, par2, par3)
% FastDSTiSynthesis -- Synthesis operator for (overcomplete) Dst dictionary
%  Usage 
%    	x = FastDSTiSynthesis(c, n, finess)
%  Inputs:
%	c	the coef
%   fineness	how many times finer the dictionary is compared to the 
%		standard one, eg: 1, 2, 4, 8 ...
%  Output:
%	x	the synthesized signal
%  See Also:
%	FastDSTiAnalysis
%


m = length(c); n = m/fineness;
x = dst_i(c(:));
x = x(1:n);
x = x(:);


