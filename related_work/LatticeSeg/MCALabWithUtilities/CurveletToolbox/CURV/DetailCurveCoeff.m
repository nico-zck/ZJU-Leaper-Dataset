function C = DetailCurveCoeff(X,deep,IsImageReal);
% DetailCurveCoeff: Returns the tight frame coefficients of the 
%                           intermediate scale 2^j;
%  Usage:
%    C = DetailCurveCoeff(X);
%  Inputs:
%    X    m by m matrix, m = 2*2^j. Typically, X is the matrix of
%         coefficients obtained after separating an object into a
%         series of scales. Scale is explicitely specified by the
%         size of X
% 
%    deep number of angular wedges per directional panel (cardinal point)
%  Outputs:
%    C    4 * d * 2^j * D array,
%          - d mumber of orienations per directional panel, m = 2^deep
%          - 2^j * D array of curvelet coefficients at a given scale
%             and orientation. The array index is effectively a
%             translation parameter. 
% 
%    C(1,:) coefficients associated with "W" west
%    C(2,:) coefficients associated with "E" east
%    C(3,:) coefficients associated with "N" north
%    C(4,:) coefficients associated with "S" south
%
%    The second index gives the directional panel; e.g. C(3,4,:,:) 
%    are coefficients associated with the 4th "Northern" directional panel
% See Also
%   SeparateScales, SeparateAngles, InvDetailCurveCoeff

         R = SqueezeAngularFT(SeparateAngles(X,deep,IsImageReal));
	 
	 nn = size(R);
	 C = zeros(nn); D = zeros(nn);
	 
	 for j = 1:size(R,1),
	   for m = 1:size(R,2),
	     W = squeeze(R(j,m,:,:));
	     W = ifft2(W)*sqrt(prod(size(W)));
	     C(j,m,:,:) = W;
	   end
	 end
	 
	 C = SymmetrizePair(C);
	         
         
         
	 
	 
	     
	 