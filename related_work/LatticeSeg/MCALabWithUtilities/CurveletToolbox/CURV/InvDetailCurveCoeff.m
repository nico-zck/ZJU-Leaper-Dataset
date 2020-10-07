function X = InvDetailCurveCoeff(C,IsImageReal);
% InvDetailCurveCoeff:  Reconstruct the jth dyadic frequency band
%                       from curvelet coefficients at that scale
%  Usage:
%    X = InvDetailCurveCoeff(C);
%  Inputs:
%    C    matrix of curvelet coefficients at scale 2^j
%  Outputs:
%    X    jth dyadic subband
%  See Also
%   DetailCurveCoeff, Inv_SeparateAngles, Inv_Curvelet02Xform

         nn = size(C); 
	 R = zeros(nn);
	 deep = log2(nn(2));
	 	 
	 C = InvSymmetrizePair(C);
         
	 for j = 1:size(R,1),
	   for m = 1:size(R,2),
	     W = squeeze(C(j,m,:,:));
	     W = fft2(W)/sqrt(prod(size(W)));
	     R(j,m,:,:) = W;
	   end
	 end
	 
	 MaxIts = 25; 
         X = Inv_SeparateAngles(Adj_SqueezeAngularFT(R),deep, ...
				IsImageReal,MaxIts,1e-6,[]); 
	 
	 
	 
	 