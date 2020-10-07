function x = FastLWPSynthesis(c, D, qmf, par3)
% FastLWPSynthesis -- Synthesis operator for Wavelet Packet dictionary
%  Usage:
%	x = FastLWPSynthesis(c, D, qmf)
%  Inputs:
%	c	the coefs, a structure array
%	D	the depth of wavelet packet
%	qmf	the quadrature mirror filter
%  Outputs:
%	x	the synthesized signal, a column vector
%  See Also:
%	FastWPAnalysis, WPAnalysis, FWPSynthesis
%

pkt = [c(2).coeff];
if isstr(D) D=str2num(D); end
x = FWPSynthesis(pkt, qmf)/(D+1);
x = x(:);
