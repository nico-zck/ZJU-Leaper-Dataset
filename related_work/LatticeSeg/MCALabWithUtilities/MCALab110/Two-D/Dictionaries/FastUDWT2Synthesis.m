function img = FastUDWT2Synthesis(wcb, scale, qmf, scign)
% FastUDWT2Synthesis -- Synthesis operator for 2D Undecimated DWT (TI)
%  Usage:
%	img = FastUDWT2Synthesis(wcb, n, scale, qmf, scign)
%  Inputs:
%	wcb	the coefs, a J-scale+1 structure array
%	scale	the coarsest decomposition scale (not used)
%	qmf	the quadrature mirror filter
%	scign   the number of detail scales to be ignored in the reconstruction
%  Outputs:
%	img	the synthesized image, a square matrix
%  See Also:
%	FastUDWT2Analysis, mrdwt, mirdwt
%

[n,J] = quadlength(wcb(1).coeff);
scale = wcb(1).scale;
if scign > J-scale
    error('Illegal detection scale parameter');
end
% Get the approximation coeffs from the structure
ll = wcb(1).coeff;

wc = zeros(n,(J-scale)*3*n);

% Get the details from the structure array
for i = J-scale:-1:1
 if i>scign
 	wc(:,(i-1)*3*n+1:i*3*n) = reshape(wcb(J-scale-i+2).coeff,n,3*n);
 end
end

img = mirdwt(ll,wc,qmf,J-scale);

