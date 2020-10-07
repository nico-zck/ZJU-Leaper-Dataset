function X = InvCoarseCurveCoeff(C);
% InvCoarseCurveCoeff: Reconstruct the low-frequency subband from
%                           the coarse scale curvelet coefficients
%  Usage:
%    X = InvCoarseCurveCoeff(C);
%  Inputs:
%    C   m by m matrix, m = 2*2^L
%  Outputs:
%    X   m by m matrix  
% See Also
%   CoaseCurveCoeff, Inv_Curvelet02Xform

X = fft2_mid0(C)/sqrt(prod(size(C)));
	 
	 
	 
	 	 