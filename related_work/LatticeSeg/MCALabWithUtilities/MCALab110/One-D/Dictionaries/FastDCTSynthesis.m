function x = FastDCTSynthesis(coeffs, fineness, par2, par3)
% FastDctAnalysis -- Synthesis operator for (overcomplete) dct dictionary
%  Usage 
%    	x = FastDCTSynthesis(coeffs, finess)
%  Inputs:
%	coeffs	the coef
%   fineness	how many times finer the dictionary is compared to the 
%		standard one, eg: 1, 2, 4, 8 ...
%  Output:
%	x	the synthesized signal
%  See Also:
%	FastDctAnalysis, dct_iv
%

c = coeffs(2).coeff;
m = length(c); n = m/fineness;
f = (1:(m-1))';
const = [n; .5 * (n +  sin(2*pi*f/fineness) ./ (2*sin(pi*f/m)))];
const = const .^ .5;

n2 = 2 * n;
c = c ./ const;
z = zeros(4*m, 1);
z(1:m) = c;
y = fft(z);
x = real(y(2:2:n2));
x = x(:) / fineness;
