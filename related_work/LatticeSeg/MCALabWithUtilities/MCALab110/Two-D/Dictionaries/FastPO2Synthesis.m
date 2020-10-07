function img = FastPO2Synthesis(wc, scale, qmf, par3)
% FastPO2Synthesis -- Fast Synthesis Operator for Periodized-Orthognal
%			Wavelets Dictionary
%  Usage:
%	img = FastPO2Synthesis(c, n, L, qmf)
%  Inputs
%    c    1-d wavelet transform of x.
%    L    Coarsest Level of V_0;  L << J
%    qmf  quadrature mirror filter (orthonormal)
%  Outputs
%    img    2-d image; length(x) = 2^J
%
%  Description
%    1. qmf filter may be obtained from MakeONFilter   
%    2. usually, length(qmf) < 2^(L+1)
% 
%  See Also
%    FastPO2Analysis, FWT2_PO, IWT2_PO, MakeONFilter
%

if isstr(scale),
   scale = str2num(scale);
end

c = wc(2).coeff;
c(1:2^scale,1:2^scale) = wc(1).coeff;
img = IWT2_PO(c, scale, qmf);
