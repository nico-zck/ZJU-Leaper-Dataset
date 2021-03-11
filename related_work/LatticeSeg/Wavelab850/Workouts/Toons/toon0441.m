% toon0441 -- Best-m Wavelet & DCT Partial Reconstructions
%
%  Compare the accuracy of best-m reconstructions of the Doppler
%  signal for the Wavelet and DCT transforms.
%
	n       = 2048;
	t       = (1:n)./n;
	S8QMF   = MakeONFilter('Symmlet',8);
%
	Doppler = MakeSignal('Doppler',n);
	dctDopp = dct_iv(Doppler);
	wcDopp  = FWT_PO(Doppler,4,S8QMF);
%
	BigDCT = KeepBiggest(dctDopp,64);
	Bigwc  = KeepBiggest(wcDopp,64);
%
	subplot(221); PartialDCT = dct_iv(BigDCT);
	plot(t,PartialDCT)
	title('Doppler: DCT Recon. 64 largest')
	axis([ 0 1 -.5 .5])
%
	subplot(222); PartialS8 = IWT_PO(Bigwc,4,S8QMF);
	plot(t,PartialS8)
	title('Doppler: S8 Wavelet Recon. 64 largest')
	axis([ 0 1 -.5 .5])
%
	Bumps = MakeSignal('Bumps',n);
	dct = dct_iv(Bumps);
	wc  = FWT_PO(Bumps,4,S8QMF);
%
	BigDCT = KeepBiggest(dct,64);
	Bigwc  = KeepBiggest(wc,64);
%
	subplot(223); PartialDCT = dct_iv(BigDCT);
	plot(t,PartialDCT)
	title('Bumps: DCT Recon. 64 largest')
	axis([ 0 1 0 6])
%
	subplot(224); PartialS8 = IWT_PO(Bigwc,4,S8QMF);
	plot(t,PartialS8)
	title('Bumps: S8 Wavelet Recon. 64 largest')
	axis([ 0 1 0 6])
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:43 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
