function C = RealPartCurveXform(D)
%  RealPartCurveXform--Takes the real part of curvelet coefficients
% Usage:
%   C = InnerProdCurveCoeff(D)
% Inputs:
%   D  Table of curvelet coefficients (real or complex)
% Outputs:
%   C  Table of curvelet coefficients (real)

nscales = length(D);
C = D;

for j = 1:nscales,
  C(j).coeff = real(D(j).coeff);
end

