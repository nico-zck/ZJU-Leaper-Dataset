% idfig06 -- Ideal Figure 06: Ideal Damping in Wavelet Domain
%
% The ideal reconstructions of figure 5 depicted in 
% the wavelet domain.  The reconstructions have a sparse
% wavelet transform.
%
global xblocks xbumps xheavi xDoppler  %AM 10/05
global yblocks ybumps yheavi yDoppler  %AM 10/05
global whblocks whbumps whheavi whDoppler
global L qmf
%
	[hbumps,whbumps]     = IdealWave(t,xbumps,   ybumps, 1.,L,qmf);  %AM 10/05
	[hblocks,whblocks]   = IdealWave(t,xblocks, yblocks, 1.,L,qmf);  %AM 10/05
	[hheavi,whheavi]     = IdealWave(t,xheavi,   yheavi, 1.,L,qmf);  %AM 10/05
	[hDoppler,whDoppler] = IdealWave(t,xDoppler,yDoppler,1.,L,qmf);  %AM 10/05

%clf; 
claxis
%
   subplot(221)
   IMJPlotWaveCoeff(whblocks,L,0.05);
   title(' 6 (a) Ideal[Blocks] ')
%
   subplot(222)
   IMJPlotWaveCoeff(whbumps,L,0.05);
   title(' 6 (b) Ideal[Bumps] ')
%
   subplot(223)
   IMJPlotWaveCoeff(whheavi,L,0.05);
   title(' 6 (c) Ideal[HeaviSine] ')
%
   subplot(224)
   IMJPlotWaveCoeff(whDoppler,L,0.05);
   title(' 6 (d) Ideal[Doppler] ')
%
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:41 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
