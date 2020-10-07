function c = FastLWPAnalysis(x, D, qmf, par3)
% FastWPAnalysis -- Analysis operator for Wavelet Packet dictionary
%  Usage:
%	c = FastLWPAnalysis(x, D, qmf)
%  Inputs:
%	x	the signal, a column vector
%	D	the depth of wavelet packet
%	qmf	the quadrature mirror filter
%  Outputs:
%	c	the coefs, a structure array
%  See Also:
%	FastWPSynthesis, WPAnalysis, FWPSynthesis
%

if isstr(D) D=str2num(D); end
pkt = WPAnalysis(x, D, qmf);
c = [struct('depth', D, 'coeff', 0) ...
     struct('depth', D, 'coeff', pkt)];
