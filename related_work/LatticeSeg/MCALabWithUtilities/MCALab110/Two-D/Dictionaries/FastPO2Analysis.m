function wc = FastPO2Analysis(img, scale, qmf, par3)
% FastPOSynthesis -- Fast Analysis Operator for Periodized-Orthognal
%			Wavelets Dictionary
%  Usage:
%	wc = FastPO2Analysis(x, L, qmf)
%  Inputs
%    img  2-d image; length(x) = 2^J
%    L    Coarsest Level of V_0;  L << J
%    qmf  quadrature mirror filter (orthonormal)
%  Outputs
%    c    2-d wavelet transform of x.
%
%  Description
%    1. qmf filter may be obtained from MakeONFilter   
%    2. usually, length(qmf) < 2^(L+1)
% 
%  See Also
%    FastPO2Synthesis, FWT2_PO, IWT2_PO, MakeONFilter
%

if isstr(scale),
   scale = str2num(scale);
end

[n,J] = quadlength(img);

c = FWT2_PO(img, scale, qmf);

wc = [struct('scale', scale, 'coeff', c(1:2^scale,1:2^scale)) struct('scale', scale, 'coeff', c)];
	
