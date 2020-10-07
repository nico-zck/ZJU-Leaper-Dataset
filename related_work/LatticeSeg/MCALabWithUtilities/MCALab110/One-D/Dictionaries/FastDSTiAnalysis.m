function c = FastDSTiAnalysis(x, fineness, par2, par3)
% FastDstiAnalysis -- Analysis operator for (overcomplete) Dst_i dictionary
%  Usage 
%    	c = FastDSTiAnalysis(x, fineness)
%  Inputs:
%	x	the signal
%   fineness	how many times finer the dictionary is compared to the 
%		standard one, eg: 1, 2, 4, 8 ...
%  Output:
%	c	the coefs
%  See Also:
%	FastDSTiSynthesis
%

n = length(x);
x = x(:);
c = dst_i([x;zeros((fineness-1)*n,1)]);
c = c(:);

