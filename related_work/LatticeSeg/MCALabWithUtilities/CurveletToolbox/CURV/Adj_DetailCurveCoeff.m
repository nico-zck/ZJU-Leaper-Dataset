function X = Adj_DetailCurveCoeff(C,IsImageReal);
% Adj_DetailCurveCoeff: Adjoint og DetailCurveCoeff  
%  Usage:
%    X = Adj_DetailCurveCoeff(C);
%  Inputs:
%    C    matrix of curvelet coefficients at scale 2^j
%  Outputs:
%    X    matrix of Fourier samples; jth dyadic subband
%  See Also
%   DetailCurveCoeff, Adj_SeparateAngles, Adj_Curvelet02Xform

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
	 
         X = Adj_SeparateAngles(Adj_SqueezeAngularFT(R),deep,IsImageReal);
	 
	 
	 
	 