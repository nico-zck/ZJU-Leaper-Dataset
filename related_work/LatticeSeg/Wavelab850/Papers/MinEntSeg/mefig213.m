% mefig213 -- De-Noising "Cusp"; Segmented Transform; ideal segmentation.
%
% In this display we show the viewer how one can use the segmented wavelet
% transform to de-noise an object.  Cusp, with discontinuity in
% derivative at a known point tau, is buried in noise (panel a).  The
% noisy data are then transformed using a segmented average-interpolating
% transform (panel b).  The resulting wavelet coefficients are
% thresholded with a sqrt(2 log n) threshold (panel c) (no coefficients
% survive thresholding, except at coarsest scale). The transform back
% into the original domain gives the reconstruction (panel d).
%
% Note -- this script makes calculations which are used later by mefig214.m
%
	global Cusp
	global yCusp SmoothCusp
	global L
	global t  % MRD 4/99
    global id n %AM 1-/05
%	
	object = 5*Cusp.^8;
	randn('seed',1269879768);
	yCusp = object + WhiteNoise(Cusp);
	absc  = (1:length(Cusp)) ./ length(Cusp);
%%
	%clf; 
    subplot(221)
	plot(absc,yCusp); 
	title('2.13a Noisy Cusp')
%
	L=4; D=2; t=id/n; %AM 10/05
	F2 = MakeAIFilter(D);
	E2 = MakeAIBdryFilter(D);
	swCusp  = FWT_SegAI(yCusp ,L,D,F2,E2,t);
%
	subplot(222); 
	PlotWaveCoeff(swCusp,L,0.); 
	xlabel('t'); ylabel('dyad');
	title('2.13b SWT(Noisy Cusp) ')
%%
	shwCusp = MultiVisu(swCusp,L);
%
	subplot(223)
	PlotWaveCoeff(shwCusp,L,0.);
	xlabel('t'); ylabel('dyad'); 
	title('2.13c Threshold(SWT(Noisy Cusp)) ')
%
	SmoothCusp = IWT_SegAI(shwCusp,L,D,F2,E2,t);
	subplot(224)
	plot(absc,SmoothCusp); title('2.13d Seg-DeNoise[Cusp]')
%  
% Prepared for the paper Minimum Entropy Segmentation 
% Copyright (c) 1994 David L. Donoho
%  
    
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:41 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
