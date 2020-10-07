function p = InnerProdCurveCoeff(C1,C2)
%  InnerProdCurveCoeff--Calculates the inner product of two arrays
%                                of curvelet coefficients 
% Usage:
%   p = InnerProdCurveCoeff(C1,C2)
% Inputs:
%   C1, C2  Tables of curvelet coefficients
% Outputs:
%   p       Complex number

if nargin < 2
  C2 = C1;
end


nscales = length(C1);
p = 0;

for j = 1:nscales,
  p = p + sum(sum(sum(sum( C1(j).coeff .*conj(C2(j).coeff) ))));      
end

