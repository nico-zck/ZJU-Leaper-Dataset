function c = FastDSTAnalysis(x, fineness, par2, par3)
% FastDstAnalysis -- Analysis operator for (overcomplete) Dst dictionary
%  Usage 
%    	c = FastDSTAnalysis(x, fineness)
%  Inputs:
%	x	the signal
%   fineness	how many times finer the dictionary is compared to the 
%		standard one, eg: 1, 2, 4, 8 ...
%  Output:
%	c	the coefs
%  See Also:
%	FastDSTSynthesis
%

n = length(x);
m = n*fineness;
f = (1:(m-1))';
const = [n; .5 * (n +  (1 - cos(2*pi*f/fineness)) ./ (2*sin(pi*f/m)))];
const = const .^ .5;

n2 = 2*n;
m = n * fineness;
y = zeros(4*m,1);
y(2:2:n2) = x(:);
z = fft(y);
c = [struct('coeff',[]) struct('coeff',imag(z(1:m)) ./ const)];
