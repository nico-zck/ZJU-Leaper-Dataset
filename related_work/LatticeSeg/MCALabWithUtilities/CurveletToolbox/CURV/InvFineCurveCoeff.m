function X = InvFineCurveCoeff(C);
% InvFineCurveCoeff: Reconstruct the last frequency subband from
%                           the fine scale curvelet coefficients
%  Usage:
%    X = InvFineCurveCoeff(C);
%  Inputs:
%    C   n by n matrix of fine scale curvelet coefficients
%  Outputs:
%    X   n by n matrix  
% See Also
%   FineCurveCoeff, Inv_Curvelet02Xform
	  
	 X = fft2_mid0(C)/sqrt(prod(size(C)));
	 
	 