function p = IsCurveletXformReal(C)
%  IsCurveletXformReal--Checks whether the array of curvelet
%  coefficients is real or not. 
% Usage:
%   p = IsCurveletXformReal(C)
% Inputs:
%   C   Table of curvelet coefficients
% Outputs:
%   p   0/1 variable, 1 if C is real


nscales = length(C);
p = 1; j = 1;

while (p == 1) && (j <= nscales)
  p = isreal(C(j).coeff);
  j = j+1;
end



