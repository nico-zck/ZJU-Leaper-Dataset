function wcb = FastUDWT2Analysis(img, scale, qmf, par3)
% FastUDWT2Analysis -- Analysis operator for 2D Undecimated DWT (TI)
%  Usage:
%	wcb = FastUDWT2Analysis(img, scale, qmf)
%  Inputs:
%	img	the image, a square matrix
%	scale	the coarsest decomposition scale
%	qmf	the quadrature mirror filter
%  Outputs:
%	wcb	the coeffs, a structure array
%  See Also:
%	FastUDWT2Synthesis, mrdwt, mirdwt
%

[n,J] = quadlength(img);
if isstr(scale) scale=str2num(scale); end
[ll,wc,L] = mrdwt(img,qmf,J-scale);

% Store the approximation in a structure
wcb = struct('scale', scale, 'coeff', ll);

% Store in a structure array
for i = J-scale:-1:1
 wcb = [wcb struct('scale', i,'coeff', reshape(wc(:,(i-1)*3*n+1:i*3*n),n,n,3))];
end



