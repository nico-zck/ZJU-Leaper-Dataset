function wcb = FastPOSpinAnalysis(x, scale, qmf, shift)
% FastPOSynthesis -- Fast Analysis Operator for Periodized-Orthognal with cycle-spinning
%			Wavelets Dictionary
%  Usage:
%	c = FastPOSpinAnalysis(x, L, qmf, shifts)
%  Inputs
%    x    1-d signal; length(x) = 2^J
%    L    Coarsest Level of V_0;  L << J
%    qmf  quadrature mirror filter (orthonormal)
%  Outputs
%    c    1-d wavelet transform of x.
%
%  Description
%    1. qmf filter may be obtained from MakeONFilter   
%    2. usually, length(qmf) < 2^(L+1)
% 
%  See Also
%    FastPOSynthesis, FWT_PO, IWT_PO, MakeONFilter
%

[n,J] = dyadlength(x);
if isstr(scale) scale=str2num(scale); end
c = FWT_PO(x(:), scale, qmf);
for i=1:shift
  c = [c FWT_PO(circshift(x(:),i), scale, qmf)];
  c = [c FWT_PO(circshift(x(:),-i), scale, qmf)];
end

wcb =  [struct('scale', scale, 'coeff', c(1:2^scale,:)) ...
	struct('scale', scale, 'coeff', c(2^scale+1:end,:))];
