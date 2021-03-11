% toon0161 -- Visualize multi-resolution decomposition
%
%  Multi-resolution Decomposition is an idea of Stephane Mallat
%  and Yves Meyer.  One groups together all the wavelets at a
%  given resolution, and displays the superposition of all the
%  wavelets in a group.
%
%  In this display, one can see that the fine-resolution components
%  of object Ramp are nearly all concentrated at the singularity,
%  and the fine-resolution components of object doppler are nearly
%  all concentrated at the early times.
%
	t     = (1:1024)./1024;
	Ramp  = MakeSignal('Ramp',1024);
%
	subplot(221); plot(t,Ramp); title('Object Ramp')
	subplot(223);
	QMF_Filter = MakeONFilter('Symmlet',8);
	wc    = FWT_PO(Ramp,3,QMF_Filter);
	PlotMultiRes(wc,3,2.,QMF_Filter);
	title('MRA[Ramp]');
%
	Doppler  = MakeSignal('Doppler',1024);
	subplot(222); plot(t,Doppler); title('Object Doppler')
	subplot(224);
	Dwc   = FWT_PO(Doppler,3,QMF_Filter);
	PlotMultiRes(Dwc,3,2.,QMF_Filter);
	title('MRA[Doppler]');


%  Revision History
%             10/1/05     AM     Changing the name of the variable QMF
    
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:43 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
